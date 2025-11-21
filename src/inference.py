import requests
import pandas as pd
import joblib
import json
import numpy as np
from sklearn.pipeline import Pipeline

from feature_builder import FeatureBuilder

import warnings
warnings.filterwarnings("ignore")

class RecommenderEngine:
    def __init__(self, model_dir="src/models", movies_file="data/raw_data/movies.csv"):
        """Initialize service by loading model and movies data"""

        preproc_path = f"{model_dir}/preprocessor.joblib"
        model_path = f"{model_dir}/xgb_model.joblib"
        preprocessor = joblib.load(preproc_path)
        xgb_model = joblib.load(model_path)
        self.model = Pipeline([("preprocessor", preprocessor), ("model", xgb_model)])

        self.movies = pd.read_csv(movies_file)
        self.base_user = "http://fall2025-comp585.cs.mcgill.ca:8080/user/"

    def get_user_info(self, user_id):
        """Fetch user info from API"""
        r = requests.get(self.base_user + str(user_id), timeout=5)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"[WARN] User {user_id} not found → cold start")
            return {"user_id": user_id, "age": -1, "occupation": "other or not specified", "gender": "U"} # default

    def build_inference_df(self, user_data):
        """Combine one user with all movies into inference dataframe"""
        user_df = pd.DataFrame([user_data])
        user_df["rating"] = 0  

        # Join user info with every movie
        movies = self.movies.copy()
        movies["user_id"] = user_data["user_id"]
        movies.rename(columns={"id": "movie_id"}, inplace=True)
        movies = movies.merge(user_df, on="user_id", how="left")
        movies = movies.sample(frac=1).reset_index(drop=True) #shuffle

        return movies

    def recommend(self, user_id, top_n=20):
        """Main entrypoint: recommend movies for a user"""
        user_data = self.get_user_info(user_id)
        candidate_df = self.build_inference_df(user_data)

        # Run through FeatureBuilder to get features
        fb = FeatureBuilder(mode="inference")
        # features = fb.build(df_override=candidate_df)

        try:
            features = fb.build(df_override=candidate_df)
            preds = self.model.predict(features)

            preds = self.model.predict(features)
            top_movies = candidate_df.assign(pred_score=preds).sort_values("pred_score", ascending=False).head(top_n)

            # Rank + return top_n
            top_movies = candidate_df.assign(pred_score=preds)
            top_movies = top_movies.sort_values("pred_score", ascending=False).head(top_n)

            return ", ".join(top_movies["movie_id"].tolist())
        except Exception as e:
            print(f"[ERROR] recommend() failed: {e}")
            return ""


if __name__ == "__main__":
    import time
    service = RecommenderEngine()
    user_id = 17588 #39387  # example cold-start
    time_start = time.time()
    recs = service.recommend(user_id)
    time_end = time.time()
    print(f"[INFO] Recommendation time: {time_end - time_start:.2f}s")
    print(f"[RECOMMENDATIONS] {recs}")
