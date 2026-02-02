import json
import os
import pandas as pd
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransportAgent:
    def __init__(self, state_json_path="transportation_state.json",
                 city_csv_path="transportation_data.csv"):
        # 1. Source of Truth for Geographic Mapping
        self.state_map = {
            'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DC': 'District of Columbia', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
            'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota',
            'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana', 'NC': 'North Carolina',
            'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia',
            'VT': 'Vermont', 'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming',
            'PR': 'Puerto Rico', 'GU': 'Guam', 'VI': 'Virgin Islands'
        }

        # Bi-directional Normalizer: Ensures "Arizona", "ARIZONA", and "AZ" all map to "AZ"
        self.name_to_abbr = {v.upper(): k.upper() for k, v in self.state_map.items()}
        for abbrev in self.state_map.keys():
            self.name_to_abbr[abbrev.upper()] = abbrev.upper()

        # 2. Load State Data (JSON)
        self.state_lookup = {}
        if os.path.exists(state_json_path):
            try:
                with open(state_json_path, "r") as f:
                    raw = json.load(f)
                for e in raw:
                    # Standardize JSON key to the 2-letter abbreviation
                    raw_state = str(e["state"]).upper().strip()
                    ls_type = str(e["lifestyle"]).lower().strip()
                    cost = float(e["cost"])

                    # Convert "Arizona" -> "AZ". If it's already "AZ", it stays "AZ".
                    clean_state = self.name_to_abbr.get(raw_state, raw_state)
                    self.state_lookup[(clean_state, ls_type)] = cost
                logger.info("✅ State JSON loaded (Keys normalized to abbreviations).")
            except Exception as e:
                logger.error(f"❌ Failed to load JSON: {e}")

            # 3. Load City Data (CSV)
            self.city_df = None
            if os.path.exists(city_csv_path):
                try:
                    # Use engine='python' to handle weird delimiters/trailing commas better
                    df = pd.read_csv(city_csv_path, engine='python')

                    # LAYER 1: Force columns to be a list of strings and clean them
                    # This prevents the 'Series' object attribute error
                    df.columns = [str(c).strip().lower() for c in df.columns.tolist()]

                    # LAYER 2: Drop junk rows and fill numeric holes
                    threshold = len(df.columns) // 2
                    df = df.dropna(thresh=threshold)

                    # Impute numeric columns only
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

                    # LAYER 3: Robust Column Cleaning
                    if 'abbreviation' in df.columns and 'city' in df.columns:
                        # Use .str accessor for safety on the series data itself
                        df['city_clean'] = df['city'].astype(str).str.lower().str.strip()
                        df['state_clean'] = df['abbreviation'].astype(str).str.upper().str.strip()
                        self.city_df = df
                        logger.info(f"✅ City CSV loaded. Standardized {len(df)} profiles.")
                    else:
                        logger.warning(f"⚠️ CSV missing required columns. Found: {df.columns.tolist()}")

                except Exception as e:
                    logger.error(f"❌ Failed to load CSV: {e}")

    def _city_transport_cost(self, row, days, travelers, travel_mode):
        def safe_float(val, default):
            try:
                if isinstance(val, float) and math.isnan(val): return default
                return float(val) if val is not None else default
            except:
                return default

        ticket = safe_float(row.get("one-way ticket (local transport)"), 2.50)
        monthly = safe_float(row.get("monthly public transport pass (regular price)"), 80.00)
        taxi_start = safe_float(row.get("taxi start (standard tariff)"), 3.50)
        taxi_dist = safe_float(row.get("taxi 1 km (standard tariff)") or row.get("taxi 1 mile (standard tariff)"), 2.00)

        taxis_needed = math.ceil(travelers / 4)
        if days >= 30:
            ticket_cost = monthly * travelers
        else:
            mult = 4 if travel_mode == 0 else 2
            ticket_cost = ticket * mult * travelers * days

        if travel_mode == 0: return round(ticket_cost, 2)
        if travel_mode == 1:
            taxi_cost = (taxi_start + taxi_dist * 5) * taxis_needed * (days / 2)
            return round(ticket_cost + taxi_cost, 2)

        taxi_cost = (taxi_start + taxi_dist * 10) * 2 * taxis_needed * days
        return round(taxi_cost, 2)

    def predict_cost(self, state, days, travelers, travel_mode, city=None, lifestyle="typical"):
        # 1. Normalize identifiers
        st_input = str(state).strip().upper()
        st_abbr = self.name_to_abbr.get(st_input, st_input)
        ls = str(lifestyle).lower().strip()

        # LOG 1: Check normalization
        logger.info(f"🔍 DEBUG [Transport]: Input State: '{state}' -> Normalized: '{st_abbr}'")

        # 2. Try City Match
        if city and self.city_df is not None:
            city_clean = city.lower().strip()
            match = self.city_df[
                (self.city_df["city_clean"] == city_clean) &
                (self.city_df["state_clean"] == st_abbr)
                ]
            if not match.empty:
                logger.info(f"✅ DEBUG [Transport]: City match found for {city_clean}, {st_abbr}")
                return self._city_transport_cost(match.iloc[0], days, travelers, travel_mode)

        # 3. State Fallback
        # LOG 2: Check exactly what key we are looking for
        lookup_key = (st_abbr, ls)
        cost = self.state_lookup.get(lookup_key)

        logger.info(f"🔍 DEBUG [Transport]: State Lookup Key: {lookup_key} | Raw Value Found: {cost}")

        # 4. The NaN/None Trap
        if cost is None:
            logger.warning(f"⚠️ DEBUG [Transport]: Key {lookup_key} NOT FOUND in state_lookup. Check JSON keys.")
            return round(15.0 * days * travelers, 2)

        if isinstance(cost, float) and math.isnan(cost):
            logger.error(f"🚨 DEBUG [Transport]: Key {lookup_key} found, but value is NaN in JSON!")
            return round(15.0 * days * travelers, 2)

        # 5. Calculation
        try:
            daily_rate = float(cost) / 30
            logger.info(f"💰 DEBUG [Transport]: Success. Monthly: {cost}, Daily: {daily_rate:.2f}")
            return round(daily_rate * days * travelers, 2)
        except Exception as e:
            logger.error(f"❌ DEBUG [Transport]: Calculation error: {e}")
            return round(15.0 * days * travelers, 2)