import requests
import pandas as pd
import joblib
import json
import numpy as np

from feature_builder import FeatureBuilder

class RecommenderEngine:
    def __init__(self, model_path="models/xgb_recommender.joblib", movies_file="../data/raw_data/movies.csv", repo_id=None, mode='prod'):
        """Initialize service by loading model and movies data"""
        if mode != 'dev' and repo_id is not None:
            from huggingface_hub import hf_hub_download
            filename = model_path.split("/")[-1]
            print(f"[INFO] Downloading model {filename} from HF repo {repo_id}")
            model_path = hf_hub_download(repo_id=repo_id, filename=filename)
            print(f"[INFO] Model downloaded to {model_path}")

        self.model = joblib.load(model_path)
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
        candidate_features = fb.build(df_override=candidate_df)

        preds = self.model.predict(candidate_features)
        candidate_df["pred_score"] = preds

        # Rank + return top_n
        top_movies = candidate_df.assign(pred_score=preds)
        top_movies = top_movies.sort_values("pred_score", ascending=False).head(top_n)

        return ",".join(top_movies["movie_id"].tolist())


if __name__ == "__main__":
    service = RecommenderEngine(mode="dev")
    user_id = 13262  # example cold-start
    recs = service.recommend(user_id)
    print(f"[RECOMMENDATIONS] {recs}")
