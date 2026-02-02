import os
import pandas as pd
import numpy as np
import duckdb
import math
import logging

logger = logging.getLogger(__name__)
# --- IMPORT AGENTS ---
try:
    from sub_agents.food_cost_agent import FoodCostAgent
except ImportError:
    print("⚠️ Warning: food_cost_agent.py not found in sub_agents.")
    FoodCostAgent = None

try:
    from sub_agents.lodging_cost_agent import LodgingCostAgent
except ImportError:
    print("⚠️ Warning: lodging_cost_agent.py not found in sub_agents.")
    LodgingCostAgent = None

try:
    from sub_agents.entertainment_agent import EntertainmentAgent
except ImportError:
    print("⚠️ Warning: entertainment_agent.py not found in sub_agents.")
    EntertainmentAgent = None

try:
    from sub_agents.transport_agent import TransportAgent
except ImportError:
    print("⚠️ Warning: transport_agent.py not found in sub_agents.")
    TransportAgent = None

try:
    from sub_agents.recommendation_entertainment_agent import RecommendationAgent
except ImportError:
    print("⚠️ Warning: recommendation_entertainment_agent.py not found in sub_agents.")
    RecommendationAgent = None


class DataHandler:
    def __init__(self, models_path=None):
        self.root_dir = os.path.dirname(os.path.abspath(__file__))
        # Use the explicit path passed from main.py
        self.models_dir = models_path or os.path.join(self.root_dir, "sub_agents", "models")

        # Initialize storage first so load_knowledge_base doesn't crash
        self.LARGE_FILE_THRESHOLD_MB = 100
        self.large_files = {"food": [], "entertainment": [], "lodging": [], "transport": []}
        self._buffers = {"food": [], "entertainment": [], "lodging": [], "transport": []}
        self.db = {cat: pd.DataFrame() for cat in ["food", "entertainment", "lodging", "transport"]}
        self.defaults = {"food": 50.0, "lodging": 150.0, "transport": 20.0, "entertainment": 30.0}
        self.trans_dir = os.path.abspath("./tmp/transportation_data")
        # --- 1. Connect Agents using models_dir ---
        if FoodCostAgent:
            pkl = os.path.join(self.models_dir, "food_cost_predictor_v1.pkl")
            if os.path.exists(pkl):
                try:
                    self.food_ai = FoodCostAgent(pkl)
                except Exception as e:
                    print(f"Food Error: {e}")

        if LodgingCostAgent:
            pkl = os.path.join(self.models_dir, "lodging_cost_predictor_v1.pkl")
            if os.path.exists(pkl):
                try:
                    self.lodging_ai = LodgingCostAgent(pkl)
                except Exception as e:
                    print(f"Lodging Error: {e}")

        if EntertainmentAgent:
            pkl = os.path.join(self.models_dir, "entertainment_cost_model.pkl")
            csv = os.path.join(self.models_dir, "city_profiles.csv")
            if os.path.exists(pkl):
                try:
                    self.ent_ai = EntertainmentAgent(model_path=pkl, city_data_path=csv)
                except Exception as e:
                    print(f"Ent Error: {e}")

        if RecommendationAgent:
            pkl = os.path.join(self.models_dir, "recommendation_engine.pkl")
            if os.path.exists(pkl):
                try:
                    self.rec_ai = RecommendationAgent(pkl)
                except Exception as e:
                    print(f"Rec Error: {e}")

        # Inside travel_agent.py -> DataHandler.__init__
        try:
            # Use the local paths where get_data_local() saved the cloud files
            state_json = os.path.join(self.models_dir, "transportation_state.json")
            city_csv = os.path.join(self.models_dir, "transportation_data.csv")

            # Instantiate with the two required local files
            self.transport_ai = TransportAgent(
                state_json_path=state_json,
                city_csv_path=city_csv
            )
            print("  [ML] TransportAgent connected with local data.")
        except Exception as e:
            print(f"  [Load] Failed transportation agent: {e}")

        # Now it is safe to scan the CSVs
        self.load_knowledge_base()

    def load_knowledge_base(self):
        print("--- Scanning Knowledge Base ---")
        for root, dirs, files in os.walk(self.root_dir):
            if any(p.startswith('.') for p in root.split(os.sep)): continue
            for file in files:
                if file.lower().endswith(('.csv', '.parquet')):
                    file_path = os.path.join(root, file)
                    try:
                        if os.path.getsize(file_path) / (1024 * 1024) > self.LARGE_FILE_THRESHOLD_MB:
                            continue
                    except:
                        continue

                    cat = self._identify_category(os.path.basename(root).lower(), file.lower())
                    if not cat: continue

                    if cat == "reference":
                        self._load_cost_of_living(file_path)
                    else:
                        self._ingest_small_file(file_path, file.lower(), cat)

        for cat, buf in self._buffers.items():
            if buf:
                try:
                    self.db[cat] = pd.concat(buf, ignore_index=True)
                except:
                    pass
        print("--- Knowledge Base Ready ---")

    # --- HELPERS ---
    def _identify_category(self, folder, filename):
        if 'cost_of_living' in filename or 'menu' in filename: return "reference"
        if 'transport' in folder or 'transit' in filename: return "transport"
        if 'lodging' in folder or 'hotel' in filename: return "lodging"
        if 'restaurant' in folder or 'food' in filename or 'yelp' in filename: return "food"
        if 'entertainment' in folder or 'venues' in filename: return "entertainment"
        return None

    def _ingest_small_file(self, path, filename, category):
        try:
            df = pd.read_parquet(path) if path.endswith('.parquet') else pd.read_csv(path, low_memory=False)
            if category == "food" and 'priceRange' in df.columns:
                self._process_food_df(df)
            elif category == "entertainment":
                self._generic_processor(df, 'entertainment', ['price', 'cost', 'fee'])
            elif category == "transport":
                self._generic_processor(df, 'transport', ['fare', 'price', 'cost'])
            elif category == "lodging":
                self._generic_processor(df, 'lodging', ['price', 'rate', 'cost', 'night'])
        except:
            pass

    def _process_food_df(self, df):
        def clean_price(p):
            try:
                p = str(p).replace('$', '').strip()
                if '-' in p: return np.mean([float(x) for x in p.split('-')])
                return float(p)
            except:
                return np.nan

        df['avg_price'] = df['priceRange'].apply(clean_price)
        df = df.dropna(subset=['avg_price'])
        if not df.empty:
            self._buffers['food'].append(
                pd.DataFrame({'city': 'any', 'state': 'USA', 'avg_price': df['avg_price']}))

    def _generic_processor(self, df, category, keywords):
        df.columns = [c.lower() for c in df.columns]
        price = next((c for c in df.columns if any(k in c for k in keywords)), None)
        city = next((c for c in df.columns if 'city' in c), None)
        if price and city:
            temp = df[[city, price]].copy()
            temp.columns = ['city', 'avg_cost']
            temp['city'] = temp['city'].astype(str).str.lower()
            temp['state'] = 'usa'
            temp['avg_cost'] = pd.to_numeric(temp['avg_cost'], errors='coerce')
            temp = temp[temp['avg_cost'] > 0]
            if not temp.empty: self._buffers[category].append(temp)

    def _load_cost_of_living(self, path):
        try:
            df = pd.read_csv(path)
            if 'Meal at an Inexpensive Restaurant' in df.columns:
                self.defaults['food'] = float(df['Meal at an Inexpensive Restaurant'].mean() * 3)
            if 'One-Way Ticket (Local Transport)' in df.columns:
                self.defaults['transport'] = float(df['One-Way Ticket (Local Transport)'].mean() * 4)
        except:
            pass

    def _map_budget_to_rating(self, budget_str):
        b = str(budget_str).lower()
        if 'cheap' in b: return 3.0
        if 'luxury' in b: return 5.0
        return 4.5

    # --- MAIN ENGINE ---
    def get_trip_estimate(self,
                          city,
                          state,
                          days,
                          travelers=1,
                          cuisine="American",
                          budget_level="Moderate",
                          demographics=None,
                          eating_out_ratio=0.5):
        """
        Main entry point for generating trip cost estimates.

        Args:
            city (str): Destination city.
            state (str): Destination state code.
            days (int): Duration of trip.
            travelers (int): Total number of travelers (Legacy).
            cuisine (str or list): Food preferences.
            budget_level (str): 'Cheap', 'Moderate', 'Luxury'.
            demographics (dict, optional): Breakdown {'males': M, 'females': F, 'children': C}.
                                           If provided, overrides 'travelers' count for other agents.
            eating_out_ratio (float): 0.0 (all grocery) to 1.0 (all restaurant). Default 0.5.
        """
        city = city.lower().strip()
        state = state.lower().strip()
        estimates = {}

        # --- PRE-PROCESS DEMOGRAPHICS & TRAVELER COUNT ---
        # 1. If demographics is provided, calculate total travelers from it
        if demographics and isinstance(demographics, dict):
            calculated_travelers = sum(demographics.values())
            if calculated_travelers > 0:
                travelers = calculated_travelers
            else:
                # Fallback if empty dict passed
                demographics = {"males": travelers, "females": 0, "children": 0}

        # 2. If demographics NOT provided, generate from 'travelers' int
        else:
            males = travelers // 2
            females = travelers - males
            demographics = {"males": males, "females": females, "children": 0}

        # 1. FOOD (ML + Multi-Cuisine Logic)
        food_result = None
        if self.food_ai:
            try:
                vibe = self._map_budget_to_rating(budget_level)

                if isinstance(cuisine, list):
                    cuisines_list = [str(c).strip() for c in cuisine if str(c).strip()]
                else:
                    cuisines_list = [c.strip() for c in str(cuisine).replace('/', ',').replace(' and ', ',').split(',')
                                     if c.strip()]

                if not cuisines_list: cuisines_list = ["American"]

                total_trip_cost_accum = 0

                # We calculate cost for every cuisine and average them
                for c in cuisines_list:
                    # NEW: Passing demographics and ratio to food agent
                    res = self.food_ai.predict_cost(
                        location=state,
                        days=days,
                        demographics=demographics,
                        eating_out_ratio=eating_out_ratio,
                        cuisine=c,
                        vibe_rating=vibe,
                        budget_level=budget_level
                    )
                    total_trip_cost_accum += res['total_cost']

                avg_trip_cost = total_trip_cost_accum / len(cuisines_list)

                food_result = {
                    "total_cost": avg_trip_cost,
                    "food_options": {
                        "grocery_heavy": avg_trip_cost * (0.6 if eating_out_ratio > 0.5 else 1.0),
                        # Approximate relative scaling
                        "balanced": avg_trip_cost,
                        "restaurant_only": avg_trip_cost * (1.4 if eating_out_ratio < 0.5 else 1.0)
                    }
                }
            except Exception as e:
                print(f"  [Food ML Error] {e}")

        if not food_result:
            print(f"  [Fallback] Calculating Food averages for {city}...")
            avg = self._get_category_avg("food", city, state)
            if avg <= 0: avg = 50.0

            # Simple fallback ratio logic
            # If high ratio (mostly restaurants), full price. If low ratio (groceries), 40% price.
            ratio_multiplier = (eating_out_ratio * 1.0) + ((1 - eating_out_ratio) * 0.4)

            total_fallback = float(avg * 3 * days * travelers * ratio_multiplier)
            food_result = {
                "total_cost": total_fallback,
                "food_options": {
                    "grocery_heavy": total_fallback * 0.5,
                    "balanced": total_fallback,
                    "restaurant_only": total_fallback * 1.5
                }
            }
        estimates["Food"] = food_result["total_cost"]

        # 2. LODGING (ML)
        lodging_total = None
        lodging_details = {}

        if self.lodging_ai:
            try:
                l_result = self.lodging_ai.predict_cost(
                    state=state,
                    travelers=travelers,  # Uses the unified traveler count
                    nights=days,
                    luxury_level=budget_level
                )
                lodging_total = l_result['total_cost']
                lodging_details = l_result['details']
            except Exception as e:
                print(f"  [Lodging ML Error] {e}")

        if lodging_total is None:
            print(f"  [Fallback] Calculating Lodging averages for {city}...")
            avg = self._get_category_avg("lodging", city, state)
            if avg <= 0: avg = 150.0

            multiplier = 1.0
            if "cheap" in budget_level.lower(): multiplier = 0.7
            if "luxury" in budget_level.lower(): multiplier = 2.5

            rooms = np.ceil(travelers / 2)
            lodging_total = float(avg * days * rooms * multiplier)
            lodging_details = {"room_type_used": "Standard (Fallback)", "amenities_assumed": 0}

        estimates["Lodging"] = float(round(lodging_total, 2))

        # 3. ENTERTAINMENT (ML)
        ent_total = None
        if self.ent_ai:
            try:
                ent_total = self.ent_ai.predict_cost(city, state, travelers, days, budget_level=budget_level)
            except Exception as e:
                print(f"  [Ent ML Error] {e}")

        if ent_total is None:
            print(f"  [Fallback] Calculating Entertainment averages for {city}...")
            ent_unit = self._get_category_avg("entertainment", city, state)
            multiplier = 1.0
            if "cheap" in budget_level.lower(): multiplier = 0.7
            if "luxury" in budget_level.lower(): multiplier = 2.5
            ent_total = float(ent_unit * days * travelers * multiplier)

        estimates["Entertainment"] = float(round(ent_total, 2))

        # Map budget_level → transport parameters
        b = budget_level.lower()

        if "cheap" in b:
            travel_style = 0  # Budget
            lifestyle = "budget"
        elif "luxury" in b:
            travel_style = 2  # Expensive / Comfortable
            lifestyle = "comfortable"
        else:
            travel_style = 1  # Medium
            lifestyle = "typical"

        estimates["Transportation"] = 0

        if self.transport_ai:
            try:
                estimates["Transportation"] = self.transport_ai.predict_cost(
                    city=city,
                    state=state,
                    days=days,
                    travelers=travelers,
                    travel_mode=travel_style,
                    lifestyle=lifestyle
                )
            except Exception as e:
                print(f"  [Transport calculation Error just giving 0] {e}")
        raw_result = {
            "total": float(round(sum(estimates.values()), 2)),
            "breakdown": estimates,
            "food_options": food_result["food_options"],
            "lodging_details": lodging_details
        }
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(i) for i in obj]
            elif isinstance(obj, float):
                # Convert NaN or Inf to 0.0 to prevent JSON crash
                return 0.0 if math.isnan(obj) or math.isinf(obj) else round(obj, 2)
            return obj

        # Insert this into DataHandler.get_trip_estimate before the return
        for category, value in estimates.items():
            if isinstance(value, float) and math.isnan(value):
                logger.error(f"🚨 DEBUG: {category} Agent returned NaN for {city}, {state}!")
                estimates[category] = 0.0  # Force fix for this run

        # Check food options too
        for opt, val in food_result.get("food_options", {}).items():
            if math.isnan(val):
                logger.error(f"🚨 DEBUG: Food Option '{opt}' is NaN!")
                food_result["food_options"][opt] = 0.0
                
        return sanitize(raw_result)


    def _get_category_avg(self, category, city, state):
        vals = []
        df = self.db.get(category, pd.DataFrame())
        if not df.empty:
            match = df[(df['city'] == city) & (df['state'] == state)]
            if not match.empty:
                vals.append(match['avg_cost'].mean())
            elif not df[df['city'] == city].empty:
                vals.append(df[df['city'] == city]['avg_cost'].mean())

        for file in self.large_files.get(category, []):
            try:
                q = f"SELECT AVG(COALESCE(try_cast(price as float), try_cast(cost as float), try_cast(fare as float), try_cast(rate as float), try_cast(avg_cost as float))) FROM '{file}' WHERE LOWER(city) = '{city}'"
                res = duckdb.sql(q).fetchone()[0]
                if res: vals.append(res)
            except:
                pass

        return float(sum(vals) / len(vals)) if vals else float(self.defaults.get(category, 100.0))

    # --- NEW: Recommendation Wrapper ---
    def get_recommendations(self, city, state, preferences):
        """
        Calls the RecommendationAgent to get venue suggestions.
        preferences expected: {'interests': [], 'budget': 'medium', 'travelers': 1}
        """
        if self.rec_ai:
            try:
                return self.rec_ai.recommend(city, state, preferences)
            except Exception as e:
                print(f"  [Rec Error] {e}")
                return []
        return []