import pandas as pd
import joblib
import os
import xgboost
import numpy as np

# USDA Food Plan Data (Monthly Costs)
# Updated to include approximate child costs (Avg of age 6-11 brackets)
GROCERY_PLANS = {
    "low": {"male": 371.0, "female": 323.0, "child": 240.0},
    "moderate": {"male": 465.0, "female": 392.0, "child": 300.0},
    "liberal": {"male": 566.0, "female": 499.0, "child": 375.0}
}


class FoodCostAgent:
    def __init__(self, model_path="sub_agents/models/food_cost_predictor_v1.pkl"):
        """
        Initializes the agent by loading the v1 XGBoost model and USDA/Index maps.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Error: Could not find '{model_path}'.")

        print(f"Loading Food Model from {model_path}...")
        try:
            artifacts = joblib.load(model_path)

            # 1. Unpack Artifacts
            self.model = artifacts["xgb_model"]
            self.features = artifacts["model_columns"]
            self.grocery_map = artifacts["grocery_index_map"]

            # Handle naming variation
            self.state_map = artifacts.get("state_map_helper", artifacts.get("state_map", {}))

            # Calculate National Avg if not explicitly saved
            self.national_avg_index = artifacts.get("national_avg_index", np.mean(list(self.grocery_map.values())))

            # Extract supported cuisines for validation
            self.known_cuisines = [
                c.replace("clean_cuisine_", "")
                for c in self.features if c.startswith("clean_cuisine_")
            ]
            print("✅ Food Model loaded successfully.")

        except Exception as e:
            print(f"❌ Failed to load food model artifacts: {e}")
            raise e

    def get_supported_cuisines(self):
        return self.known_cuisines

    def _get_state_code(self, user_input):
        """Standardizes input to 2-letter state code."""
        clean = str(user_input).lower().strip()
        if len(clean) == 2:
            return clean.upper()
        for name, code in self.state_map.items():
            if name in clean:
                return code
        return "Unknown"

    def predict_cost(self,
                     location,
                     days,
                     demographics=None,
                     eating_out_ratio=0.5,
                     cuisine="American",
                     vibe_rating=4.5,
                     budget_level="moderate"):
        """
        Calculates food costs combining:
        1. XGBoost Model (Restaurants)
        2. USDA Food Plans (Groceries) with Demographic breakdown
        3. Custom Eating Out Ratio

        Args:
            location (str): State or location name.
            days (int): Duration of the trip.
            demographics (dict): {'males': int, 'females': int, 'children': int}. Defaults to 1 Male.
            eating_out_ratio (float): 0.0 (all grocery) to 1.0 (all restaurant). Default 0.5.
            cuisine (str): Preferred cuisine type.
            vibe_rating (float): Expected quality/rating of restaurants.
            budget_level (str): 'low', 'moderate', or 'liberal'.
        """
        state_code = self._get_state_code(location)

        # Default demographics if None provided
        if demographics is None:
            demographics = {"males": 1, "females": 0, "children": 0}

        total_people = sum(demographics.values())
        if total_people == 0:
            total_people = 1  # Fallback
            demographics["males"] = 1

        # --- 1. RESTAURANT PREDICTION (XGBoost) ---
        input_df = pd.DataFrame(0, index=[0], columns=self.features)
        input_df['rating'] = vibe_rating
        input_df['review_count'] = 150

        # One-Hot Encoding: State
        if f"state_{state_code}" in input_df.columns:
            input_df[f"state_{state_code}"] = 1

        # One-Hot Encoding: Cuisine
        target_cuisine = "Other"
        for k in self.known_cuisines:
            if k.lower() == cuisine.lower():
                target_cuisine = k
                break

        if f"clean_cuisine_{target_cuisine}" in input_df.columns:
            input_df[f"clean_cuisine_{target_cuisine}"] = 1

        try:
            pred_price = float(self.model.predict(input_df)[0])
            price_restaurant_meal = max(pred_price, 7.00)
        except Exception as e:
            print(f"  [ML Warning] Prediction failed, using fallback. Error: {e}")
            price_restaurant_meal = 15.00

        # --- 2. GROCERY PREDICTION (USDA Logic) ---
        plan_key = budget_level.lower()
        if plan_key not in GROCERY_PLANS:
            plan_key = "moderate"

        plan_costs = GROCERY_PLANS[plan_key]

        # Calculate Monthly Base based on specific demographics
        monthly_base_national = (
                (demographics.get("males", 0) * plan_costs["male"]) +
                (demographics.get("females", 0) * plan_costs["female"]) +
                (demographics.get("children", 0) * plan_costs["child"])
        )

        # Convert to Weekly Base
        weekly_base_national = monthly_base_national / 4.33

        # Apply State Index Multiplier
        state_idx = float(self.grocery_map.get(state_code, self.national_avg_index))
        state_multiplier = state_idx / 100.0

        # Full Grocery Cost (assuming 100% cooking)
        weekly_grocery_local_total = weekly_base_national * state_multiplier

        # --- 3. SCENARIO CALCULATIONS ---
        weeks = days / 7.0

        # Validate ratio (clamp between 0 and 1)
        eating_out_ratio = max(0.0, min(1.0, eating_out_ratio))

        # A. Custom User Scenario (The requested output)
        # Assume 21 meals per week (3 per day)
        total_meals = 21
        meals_restaurant = total_meals * eating_out_ratio

        # Grocery Factor Logic:
        # If eating out 100%, grocery is not 0% (snacks, water, breakfast), floor at 10%
        # If eating out 0%, grocery is 100%
        grocery_utilization = max(0.1, 1.0 - eating_out_ratio)

        cost_user_custom = (price_restaurant_meal * meals_restaurant * weeks * total_people) + \
                           (weekly_grocery_local_total * weeks * grocery_utilization)

        # B. Standard Benchmarks (for comparison in UI)
        # Saver: Eat out 10% of time (approx 2 meals/week)
        cost_saver = (price_restaurant_meal * (21 * 0.1) * weeks * total_people) + \
                     (weekly_grocery_local_total * weeks * 0.9)

        # Balanced: Eat out 30% of time (approx 6 meals/week)
        cost_balanced = (price_restaurant_meal * (21 * 0.3) * weeks * total_people) + \
                        (weekly_grocery_local_total * weeks * 0.7)

        # Foodie: Eat out 70% of time (approx 15 meals/week)
        cost_foodie = (price_restaurant_meal * (21 * 0.7) * weeks * total_people) + \
                      (weekly_grocery_local_total * weeks * 0.3)

        return {
            "total_cost": float(round(cost_user_custom, 2)),
            "parameters": {
                "ratio_used": eating_out_ratio,
                "demographics": demographics
            },
            "breakdown": {
                "unit_restaurant_meal": float(round(price_restaurant_meal, 2)),
                "weekly_grocery_full_cost": float(round(weekly_grocery_local_total, 2)),
                "state_index": state_idx
            },
            "food_options": {
                "grocery_heavy": float(round(cost_saver, 2)),
                "balanced": float(round(cost_balanced, 2)),
                "restaurant_only": float(round(cost_foodie, 2))
            }
        }