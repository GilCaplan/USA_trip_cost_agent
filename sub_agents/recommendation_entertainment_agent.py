import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import normalize


class RecommendationAgent:
    def __init__(self, model_path="sub_agents/models/recommendation_engine.pkl"):
        if not os.path.exists(model_path):
            print(f"⚠️ Recommendation model not found at {model_path}")
            self.model_loaded = False
            return

        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            self.venues_df = data['venues_df']

            # --- FIX 1: Load and Cap Data immediately ---
            # Use float32 to save memory and reduce complexity
            raw_matrix = np.array(data['feature_matrix'], dtype=np.float32)

            # 1. HARD CAP: Replace anything > 10,000 with 10,000
            #    This fixes the "overflow" issue if you have a price of $1,000,000
            raw_matrix[raw_matrix > 10000.0] = 10000.0

            # 2. Safety: Replace NaN/Infinity with 0 or max cap
            raw_matrix = np.nan_to_num(raw_matrix, nan=0.0, posinf=10000.0, neginf=-10000.0)

            # 3. Normalize (L2): Compresses all vectors to length of 1.0
            #    (Math becomes crash-proof after this)
            self.feature_matrix = normalize(raw_matrix, axis=1, norm='l2')

            self.feature_columns = data['feature_columns']
            self.config = data['config']

            if 'price_avg' not in self.venues_df.columns:
                self.venues_df['price_avg'] = 50.0

            self.model_loaded = True
            print("  [ML] RecommendationAgent loaded. Matrix shape:", self.feature_matrix.shape)

        except Exception as e:
            print(f"❌ Error loading RecommendationAgent: {e}")
            self.model_loaded = False

    def _create_preference_vector(self, preferences):
        """Converts user text preferences into a mathematical vector."""
        vector = np.zeros(len(self.feature_columns), dtype=np.float32)
        cols = self.feature_columns
        known_cats = self.config.get('CATEGORIES', [])

        # 1. Interests
        user_interests = preferences.get('interests', [])
        matched = False
        for interest in user_interests:
            for cat in known_cats:
                if cat in interest.lower() or interest.lower() in cat:
                    col_name = f'cat_{cat}'
                    if col_name in cols:
                        vector[cols.index(col_name)] = 1.0
                        matched = True

        # Fallback
        if not matched:
            for cat in known_cats:
                col_name = f'cat_{cat}'
                if col_name in cols:
                    vector[cols.index(col_name)] = 0.2

        # 2. Budget
        budget = preferences.get('budget', 'medium').lower()
        mapping = {'cheap': 'budget', 'expensive': 'expensive', 'luxury': 'luxury'}
        budget_key = next((k for k, v in mapping.items() if k in budget), 'medium')

        price_col = f'price_{budget_key}'
        if price_col in cols:
            vector[cols.index(price_col)] = 1.0

        # 3. Group
        travelers = preferences.get('travelers', 1)
        group_type = 'all' if travelers > 2 else 'adults'
        aud_col = f'aud_{group_type}'
        if aud_col in cols:
            vector[cols.index(aud_col)] = 1.0

        # Normalize vector
        vector = vector.reshape(1, -1)
        vector = normalize(vector, axis=1, norm='l2')
        return vector

    def recommend(self, city, state, preferences, top_n=5):
        if not self.model_loaded:
            return []

        # --- 1. FILTER LOCATIONS ---
        city_mask = (self.venues_df['city'].str.lower() == city.lower())
        state_mask = (self.venues_df['state'].str.lower() == state.lower()) if state else city_mask

        location_mask = city_mask & state_mask if state else city_mask

        # Fallback
        if location_mask.sum() < 2 and state:
            print(f"  [RecAgent] Few results for {city}. Expanding search to all of {state}.")
            location_mask = state_mask

        if location_mask.sum() == 0:
            return []

        # --- 2. BUDGET FILTER ---
        current_indices = np.where(location_mask)[0]
        max_spend = preferences.get('max_spend', None)

        if max_spend is not None and not np.isnan(max_spend):
            limit = max_spend * 1.2
            budget_mask = (self.venues_df['price_avg'] <= limit)
            combined_mask = location_mask & budget_mask

            if combined_mask.sum() > 0:
                current_indices = np.where(combined_mask)[0]
            else:
                print(f"  [RecAgent] Budget ${max_spend} too low. Showing cheapest.")

        # --- 3. PREPARE DATA ---
        relevant_features = self.feature_matrix[current_indices].copy()
        relevant_venues = self.venues_df.iloc[current_indices].copy()

        # --- FIX 2: Runtime Sanitize (Defense Layer 2) ---
        # Explicitly cap values > 10000 again on this subset
        relevant_features[relevant_features > 10000.0] = 10000.0
        relevant_features = np.nan_to_num(relevant_features, nan=0.0, posinf=10000.0, neginf=-10000.0)

        user_vector = self._create_preference_vector(preferences)
        user_vector = np.nan_to_num(user_vector, nan=0.0)

        # --- 4. CALCULATE SCORES ---
        try:
            # Safe Dot Product
            scores = np.dot(relevant_features, user_vector.T).flatten()
        except Exception as e:
            print(f"⚠️ [RecAgent] Matrix Math Failed ({e}). Using Manual Loop.")
            scores = []
            u_vec = user_vector.flatten()
            for row in relevant_features:
                scores.append(np.dot(row, u_vec))
            scores = np.array(scores)

        # --- 5. RANK & RETURN ---
        relevant_venues['match_score'] = scores
        relevant_venues = relevant_venues.sort_values(
            by=['match_score', 'popularity_score'],
            ascending=[False, False]
        ).head(top_n)

        recommendations = []
        for _, row in relevant_venues.iterrows():
            rec = {
                "name": row['venue_name'],
                "category": row['category'],
                "price": row['price_tier'],
                "avg_cost": row.get('price_avg', 'N/A'),
                "rating": row.get('rating', 'N/A')
            }
            recommendations.append(rec)

        return recommendations