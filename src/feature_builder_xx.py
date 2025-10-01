import pandas as pd
import numpy as np
from configs import LANGUAGE_MAP

import warnings
warnings.filterwarnings("ignore")

class FeatureBuilder:
    def __init__(self, movies_file=None, ratings_file=None, users_file=None, mode="train"):
        self.mode = mode
        self.movies = pd.read_csv(movies_file) if movies_file else None
        self.ratings = pd.read_csv(ratings_file) if ratings_file else None
        self.users = pd.read_csv(users_file) if users_file else None
        self.df = None

    def build(self, df_override=None):
        """
        Build final feature dataframe.
        If mode == 'train': merge ratings, users, and movies as before.
        If mode == 'inference': expect df with user+movie candidates (no ratings).
        """

        if self.mode == "train":
            # Merge ratings + users
            df = self.ratings.merge(self.users, on="user_id", how="left")
            df = df.merge(self.movies, left_on="movie_id", right_on="id", how="left")
        elif self.mode == "inference":
            if df_override is None:
                raise ValueError("Inference mode requires df_override (user+movie candidates).")
            df = df_override.copy()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # --- Handle Missing Values ---
        # Numeric
        df["age"] = df["age"].fillna(df["age"].median())
        df["runtime"] = df["runtime"].fillna(df["runtime"].median())
        df["popularity"] = df["popularity"].fillna(0)
        df["vote_average"] = df["vote_average"].fillna(0)
        df["vote_count"] = df["vote_count"].fillna(0)

        # Categorical
        df["occupation"] = df["occupation"].fillna("unknown")
        df["gender"] = df["gender"].fillna("U")

        # Release year
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        df["release_year"] = df["release_year"].fillna(-1).astype(int)

        # --- Feature Engineering ---
        # Age binning
        df["age_bin"] = pd.cut(
            df["age"], 
            bins=[0, 18, 25, 35, 50, 100], 
            labels=["0-18", "19-25", "26-35", "36-50", "50+"]
        )

        # Genres multi-hot
        for g in df["genres"].dropna().unique():
            if not isinstance(g, str):
                continue
            for genre in g.split(","):
                genre = genre.strip()
                if genre:
                    df[f"genre_{genre}"] = df["genres"].fillna("").str.contains(genre).astype(int)

        # Production countries multi-hot
        for c in df["production_countries"].dropna().unique():
            if not isinstance(c, str):
                continue
            for country in c.split(","):
                country = country.strip()
                if country:
                    df[f"country_{country}"] = df["production_countries"].fillna("").str.contains(country).astype(int)

        # Languages
        df["original_language"] = df["original_language"].fillna("unknown")

        def normalize_languages(lang_str):
            if pd.isna(lang_str):
                return []
            langs = [l.strip() for l in lang_str.split(",") if l.strip()]
            return set([LANGUAGE_MAP.get(l, l) for l in langs])  # Default to original if not found

        df["normalized_langs"] = df["spoken_languages"].apply(normalize_languages)
        all_langs = sorted({lang for langs in df["normalized_langs"] for lang in langs})
        for lang in all_langs:
            df[f"lang_{lang}"] = df["normalized_langs"].apply(lambda x: int(lang in x))

        # One-hot encode occupation and gender
        df = pd.get_dummies(df, columns=["occupation", "gender"], prefix=["occ", "gender"])

        # Final selection (rating only included in training)
        base_cols = ["user_id", "age", "age_bin", "movie_id", "runtime",
                     "popularity", "vote_average", "vote_count", "release_year", "original_language"]

        if self.mode == "train":
            df = df.dropna(subset=["rating"])
            base_cols.append("rating")

        # Final selection
        self.df = df[
            base_cols
            + [col for col in df.columns if col.startswith("genre_")]
            + [col for col in df.columns if col.startswith("lang_")]
            + [col for col in df.columns if col.startswith("country_")]
            + [col for col in df.columns if col.startswith("occ_")]
            + [col for col in df.columns if col.startswith("gender_")]
        ]

        print(f"[INFO] Final feature dataframe shape ({self.mode}): {self.df.shape}")
        return self.df


if __name__ == "__main__":
    data_dir = "data/raw_data/"
    fb = FeatureBuilder(f"{data_dir}/movies.csv", f"{data_dir}/ratings.csv", f"{data_dir}/users.csv", mode="train")
    final_df = fb.build()
    final_df.to_csv("data/training_data.csv", index=False)
    print("[INFO] Saved training_data.csv")
