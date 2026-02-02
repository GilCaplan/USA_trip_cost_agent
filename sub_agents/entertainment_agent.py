import pickle
import pandas as pd
import os
import numpy as np


class EntertainmentAgent:
    def __init__(self,
                 model_path="sub_agents/models/entertainment_cost_model.pkl",
                 city_data_path="city_profiles.csv"):

        # 1. Load the "Brain" (Model + Config)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Error: Could not find model at '{model_path}'.")

        print(f"Loading Entertainment Agent from {model_path}...")
        try:
            with open(model_path, 'rb') as f:
                artifacts = pickle.load(f)

            self.model = artifacts["model"]
            self.features = artifacts["features"]
            self.style_map = artifacts["travel_style_map"]

            self.numeric_cols = [f for f in self.features if
                                 f not in ['travel_style_encoded', 'num_people', 'num_days']]

            print("✅ Entertainment Model & Config loaded.")
        except Exception as e:
            print(f"❌ Failed to load model artifacts: {e}")
            raise e

        # 2. Load City Knowledge Base
        if os.path.exists(city_data_path):
            try:
                if city_data_path.endswith('.parquet'):
                    self.city_db = pd.read_parquet(city_data_path)
                else:
                    self.city_db = pd.read_csv(city_data_path)

                self.city_db['city_lower'] = self.city_db['city'].astype(str).str.lower().str.strip()
                self.city_db['state_lower'] = self.city_db['state'].astype(str).str.lower().str.strip()
                print(f"✅ Loaded city profiles for {len(self.city_db)} cities.")

            except Exception as e:
                print(f"⚠️ Warning: Could not load city profiles: {e}")
                self.city_db = pd.DataFrame()
        else:
            print(f"⚠️ Warning: City data not found at {city_data_path}. Predictions will rely on averages.")
            self.city_db = pd.DataFrame()

        # State Mapping Dictionary
        self.state_map = {
            "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar", "california": "ca",
            "colorado": "co", "connecticut": "ct", "delaware": "de", "florida": "fl", "georgia": "ga",
            "hawaii": "hi", "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
            "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me", "maryland": "md",
            "massachusetts": "ma", "michigan": "mi", "minnesota": "mn", "mississippi": "ms", "missouri": "mo",
            "montana": "mt", "nebraska": "ne", "nevada": "nv", "new hampshire": "nh", "new jersey": "nj",
            "new mexico": "nm", "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
            "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri", "south carolina": "sc",
            "south dakota": "sd", "tennessee": "tn", "texas": "tx", "utah": "ut", "vermont": "vt",
            "virginia": "va", "washington": "wa", "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
            "district of columbia": "dc"
        }

    def _get_city_features(self, city, state):
        if self.city_db.empty:
            return None

        target_city = city.lower().strip()
        target_state = state.lower().strip()

        # Map full state name to abbreviation if needed (e.g. "Nevada" -> "nv")
        if target_state in self.state_map:
            target_state = self.state_map[target_state]

        # Candidate names
        candidates = [target_city]
        if "new york" in target_city and "city" not in target_city:
            candidates.append(target_city + " city")
        elif "washington" in target_city and "dc" not in target_city:
            candidates.append("washington")

        for name in candidates:
            # 1. Exact Match (City + State)
            if target_state:
                match = self.city_db[
                    (self.city_db['city_lower'] == name) &
                    (self.city_db['state_lower'] == target_state)
                    ]
                if not match.empty:
                    return match.iloc[0].to_dict()

            # 2. Loose Match (City Only)
            match = self.city_db[self.city_db['city_lower'] == name]
            if not match.empty:
                return match.iloc[0].to_dict()

        # 3. State Fallback
        if target_state:
            state_neighbors = self.city_db[self.city_db['state_lower'] == target_state]
            if not state_neighbors.empty:
                print(f"  [Ent ML] City '{city}' not found. Using average for state '{target_state.upper()}'.")
                return state_neighbors.mean(numeric_only=True).fillna(0).to_dict()

        return None

    def _get_national_averages(self):
        if self.city_db.empty:
            return {'venue_count': 50, 'avg_price': 30.0}
        return self.city_db.mean(numeric_only=True).to_dict()

    def predict_cost(self, city, state, travelers, days, budget_level="Moderate"):
        # 1. Get City/State Stats
        city_stats = self._get_city_features(city, state)

        if not city_stats:
            print(f"  [Ent ML] Location '{city}, {state}' not found. Using national averages.")
            city_stats = self._get_national_averages()

        # 2. Map Travel Style
        style_key = "medium"
        b_lower = budget_level.lower()
        if "cheap" in b_lower:
            style_key = "budget"
        elif "luxury" in b_lower:
            style_key = "luxury"
        elif "expensive" in b_lower:
            style_key = "expensive"

        encoded_style = self.style_map.get(style_key, 1)

        # 3. Construct Input Vector
        input_data = {}

        for feature in self.features:
            if feature in city_stats:
                input_data[feature] = city_stats[feature]
            elif feature == "num_people":
                input_data[feature] = travelers
            elif feature == "num_days":
                input_data[feature] = days
            elif feature == "travel_style_encoded":
                input_data[feature] = encoded_style
            else:
                input_data[feature] = 0.0

        input_df = pd.DataFrame([input_data])
        input_df = input_df[self.features]

        # 4. Predict
        try:
            predicted_cost = float(self.model.predict(input_df)[0])
            min_cost = 5.0 * travelers * days
            final_cost = max(predicted_cost, min_cost)
            return float(round(final_cost, 2))

        except Exception as e:
            print(f"  [Ent ML Error] Prediction failed: {e}")
            base = 30.0 * (0.6 if style_key == 'budget' else 2.5 if style_key == 'luxury' else 1.0)
            return float(round(base * travelers * days, 2))