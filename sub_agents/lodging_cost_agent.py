import pandas as pd
import joblib
import os
import numpy as np


class LodgingCostAgent:
    def __init__(self, model_path="sub_agents/models/lodging_cost_predictor_v1.pkl"):
        # 1. Load the Brain
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Error: Could not find '{model_path}'.")

        print(f"Loading Lodging Model from {model_path}...")
        try:
            artifacts = joblib.load(model_path)
            self.model = artifacts["lodging_model"]
            self.features = artifacts["model_columns"]  # The specific columns XGBoost expects

            # 2. DECIPHER THE BRAIN
            # The model was trained with 'clean_state_X'. We need to know what 'X' is.
            # We extract the list of states the model actually learned as distinct features.
            self.known_states_in_model = [
                col.replace("clean_state_", "")
                for col in self.features
                if col.startswith("clean_state_")
            ]

            print(f"✅ Lodging Model loaded. It knows {len(self.known_states_in_model)} distinct states.")

        except Exception as e:
            print(f"❌ Failed to load lodging artifacts: {e}")
            raise e

        # 3. STATIC MAPPER (Helper)
        # Helps us bridge the gap if user says "NY" but model knows "New York" (or vice versa)
        self.abbr_to_name = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
            'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
            'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
            'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming',
            'DC': 'District of Columbia'
        }
        # Create reverse map too
        self.name_to_abbr = {v.lower(): k for k, v in self.abbr_to_name.items()}

    def _resolve_state_feature(self, user_input):
        """
        Matches user input (e.g. 'NY') to the exact column name in the model (e.g. 'clean_state_New York').
        Returns the suffix string or None.
        """
        raw = str(user_input).strip()

        # Candidate 1: Exact Match (e.g. Model has 'NY', input is 'NY')
        if raw in self.known_states_in_model:
            return raw

        # Candidate 2: Try Full Name (e.g. Model has 'New York', input is 'NY')
        if raw.upper() in self.abbr_to_name:
            full_name = self.abbr_to_name[raw.upper()]
            if full_name in self.known_states_in_model:
                return full_name

        # Candidate 3: Try Abbr (e.g. Model has 'NY', input is 'New York')
        if raw.lower() in self.name_to_abbr:
            abbr = self.name_to_abbr[raw.lower()]
            if abbr in self.known_states_in_model:
                return abbr

        # Candidate 4: Case-insensitive scan
        for s in self.known_states_in_model:
            if raw.lower() == s.lower():
                return s

        return None

    def predict_cost(self, state, travelers, nights, room_type="Entire Home/Apt", rating=4.8, luxury_level="Moderate"):
        """
        Predicts using the EXACT XGBoost model structure from your training code.
        """

        # 1. Map Inputs to Training Features
        # Your training code used these defaults:
        # defaults = {"rating": 4.5, "review_count": 0, "amenities_count": 5, "host_score": 4.5, "desc_length": 100}

        amenities = 5
        host_score = 4.5
        desc_len = 100
        review_count = 50  # Assume established listing

        # Adjust based on 'Luxury Level' vibe
        if "cheap" in luxury_level.lower():
            amenities = 3
            desc_len = 50
            if room_type == "Entire Home/Apt": room_type = "Private Room"  # Downgrade logic

        elif "luxury" in luxury_level.lower():
            amenities = 25
            host_score = 4.9
            desc_len = 800
            review_count = 150

        # 2. Build the Input Vector (DataFrame)
        # We start with 0s for everything
        input_df = pd.DataFrame(0, index=[0], columns=self.features)

        # Set Numerical Columns (Names match your training code exactly)
        input_df['guests'] = travelers
        input_df['rating'] = rating
        input_df['review_count'] = review_count
        input_df['amenities_count'] = amenities
        input_df['host_score'] = host_score
        input_df['desc_length'] = desc_len

        # 3. Set One-Hot Encoded Room Type
        # Training used: pd.get_dummies(..., drop_first=True)
        # We map standard terms to the likely column names
        target_room = "Private Room"  # Default
        r_lower = room_type.lower()

        if "entire" in r_lower or "home" in r_lower or "apt" in r_lower:
            target_room = "Entire Home/Apt"
        elif "hotel" in r_lower:
            target_room = "Hotel"
        elif "shared" in r_lower:
            target_room = "Shared Room"

        # The training code generates columns like "room_type_Hotel", "room_type_Private Room"
        room_col = f"room_type_{target_room}"
        if room_col in input_df.columns:
            input_df[room_col] = 1

        # 4. Set One-Hot Encoded State (The Critical Fix)
        state_suffix = self._resolve_state_feature(state)

        if state_suffix:
            state_col = f"clean_state_{state_suffix}"
            input_df[state_col] = 1
            # print(f"  [Lodging] Matched '{state}' to model column '{state_col}'")
        else:
            # If state not found, we leave all state columns 0.
            # In get_dummies(drop_first=True), 0s everywhere means "Reference Category" (often 'Other')
            # print(f"  [Lodging] State '{state}' unknown to model. Using baseline/Other.")
            pass

        # 5. Predict
        try:
            # XGBoost predict
            price_per_night = float(self.model.predict(input_df)[0])

            # Sanity check: If model predicts negative or near-zero (possible with regression)
            min_price = 10.0 * travelers
            price_per_night = max(price_per_night, min_price)

        except Exception as e:
            # print(f"  [Lodging Error] {e}")
            price_per_night = 80.0 * travelers  # Emergency fallback

        total_cost = price_per_night * nights

        return {
            "total_cost": float(round(total_cost, 2)),
            "per_night": float(round(price_per_night, 2)),
            "details": {
                "room_type": target_room,
                "state_feature": state_suffix if state_suffix else "Other/Baseline"
            }
        }