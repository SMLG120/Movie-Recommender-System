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
        If mode == 'train': merge ratings, users, and movies.
        If mode == 'inference': expect df with user+movie candidates.
        """

        if self.mode == "train":
            df = self.ratings.merge(self.users, on="user_id", how="left")
            df = df.merge(self.movies, left_on="movie_id", right_on="id", how="left")
        elif self.mode == "inference":
            if df_override is None:
                raise ValueError("Inference mode requires df_override (user+movie candidates).")
            df = df_override.copy()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # --- Coerce Types ---
        for c in ["age", "runtime", "popularity", "vote_average", "vote_count"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        for c in ["age_bin","occupation","gender","original_language"]:
            if c not in df.columns:
                df[c] = "unknown"
            df[c] = df[c].astype(str)

        # --- Handle Missing Values ---
        df["age"] = df["age"].fillna(df["age"].median())
        df["runtime"] = df["runtime"].fillna(df["runtime"].median())
        df["popularity"] = df["popularity"].fillna(0)
        df["vote_average"] = df["vote_average"].fillna(0)
        df["vote_count"] = df["vote_count"].fillna(0)

        df["occupation"] = df["occupation"].fillna("unknown")
        df["gender"] = df["gender"].fillna("U")
        df["original_language"] = df["original_language"].fillna("unknown")

        # Release year
        df["release_year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
        df["release_year"] = df["release_year"].fillna(-1).astype(int)

        # Age binning
        df["age_bin"] = pd.cut(
            df["age"],
            bins=[0, 18, 25, 35, 50, 100],
            labels=["0-18", "19-25", "26-35", "36-50", "50+"]
        )
        # Ensure keys
        if "user_id" not in df:  df["user_id"] = -1
        if "movie_id" not in df: df["movie_id"] = df.get("id", -1)

        # --- Light outlier checks ---
        # keeps data, just clips into valid ranges and prints a summary
        def _clip_with_flag(s, low=None, high=None, name="col"):
            orig = s.copy()
            if low is not None:  s = s.clip(lower=low)
            if high is not None: s = s.clip(upper=high)
            clipped = (orig != s).sum()
            if clipped:
                total = len(s)
                pct = 100.0 * clipped / total
                print(f"[INFO] clipped {name}: {clipped}/{total} ({pct:.2f}%)")
            else:
                print("no clipping needed for", name)
            return s
    
        print("[INFO] Outlier clipping summary:")
        if "age" in df: df["age"] = _clip_with_flag(df["age"], 5, 100, "age")
        if "runtime" in df: df["runtime"] = _clip_with_flag(df["runtime"], 30, 300, "runtime")
        if "vote_count" in df: df["vote_count"] = _clip_with_flag(df["vote_count"], 0, None, "vote_count")
        if "release_year" in df:
            this_year = pd.Timestamp.now().year
            df["release_year"] = _clip_with_flag(df["release_year"], 1900, this_year + 1, "release_year")

        # Genres multi-hot (keep as binary features)
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

        # Spoken languages → normalized multi-hot
        def normalize_languages(lang_str):
            if pd.isna(lang_str):
                return []
            langs = [l.strip() for l in lang_str.split(",") if l.strip()]
            return set([LANGUAGE_MAP.get(l, l) for l in langs])

        df["normalized_langs"] = df["spoken_languages"].apply(normalize_languages)
        all_langs = sorted({lang for langs in df["normalized_langs"] for lang in langs})
        for lang in all_langs:
            df[f"lang_{lang}"] = df["normalized_langs"].apply(lambda x: int(lang in x))
        
        # Final selection
        base_cols = [
            "user_id", "age", "age_bin", "occupation", "gender",
            "movie_id", "runtime", "popularity", "vote_average", "vote_count",
            "release_year", "original_language"
        ]

        if self.mode == "train":
            df = df.dropna(subset=["rating"])
            base_cols.append("rating")

        self.df = df[
            base_cols
            + [col for col in df.columns if col.startswith("genre_")]
            + [col for col in df.columns if col.startswith("lang_")]
            + [col for col in df.columns if col.startswith("country_")]
        ]

        print(f"[INFO] Final feature dataframe shape ({self.mode}): {self.df.shape}")
        return self.df


if __name__ == "__main__":
    data_dir = "data/raw_data/"
    fb = FeatureBuilder(
        f"{data_dir}/movies.csv",
        f"{data_dir}/ratings.csv",
        f"{data_dir}/users.csv",
        mode="train"
    )
    final_df = fb.build()
    final_df.to_csv("data/training_data.csv", index=False)
    print("[INFO] Saved training_data.csv")
