import joblib, gc
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Callable
from configs import LANGUAGE_MAP

import warnings
warnings.filterwarnings("ignore")


class FeatureBuilder:
    def __init__(
        self,
        movies_file: Optional[str] = None,
        ratings_file: Optional[str] = None,
        users_file: Optional[str] = None,
        user_explicit_factors: Optional[str] = None,
        movie_explicit_factors: Optional[str] = None,
        user_implicit_factors: Optional[str] = None,
        movie_implicit_factors: Optional[str] = None,
        test_ids: Optional[list] = None,
        mode: str = "train",
        reader: Callable[[str], pd.DataFrame] = None,   # inject for tests
        logger: Optional[Callable[[str], None]] = None,
    ):

        self.mode = mode
        self._read_csv = reader or pd.read_csv
        self._log = logger or (lambda msg: print(msg))

        # Load data (mockable)
        self.test_ids = test_ids or []
        self.movies = self._safe_read(movies_file)
        self.ratings = self._safe_read(ratings_file)
        self.users = self._safe_read(users_file)
        self.user_explicit = self._safe_read(user_explicit_factors)
        self.movie_explicit = self._safe_read(movie_explicit_factors)
        self.user_implicit = self._safe_read(user_implicit_factors)
        self.movie_implicit = self._safe_read(movie_implicit_factors)
        self.df = None

    def _safe_read(self, path):
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            self._log(f"[WARN] File not found: {path}")
            return None
        data = self._read_csv(path)
        data = data[~(data['user_id'].isin(self.test_ids))] if 'user_id' in data.columns else data
        return data


    def build(self, df_override=None):
        """Build the final feature dataframe."""
        df = self._prepare_base(df_override)
        df = self._coerce_types(df)
        print(f"[INFO] Dataframe shape after base preparation ({self.mode}): {df.shape}")
        df = self._fill_missing(df)
        df = self._handle_dates_and_bins(df)
        print(f"[INFO] Dataframe shape after filling missing and handling dates ({self.mode}): {df.shape}")
        df = self._countries_and_languages(df)
        df = self._clip_outliers(df)
        print(f"[INFO] Dataframe shape after clipping outliers ({self.mode}): {df.shape}")
        df = self._encode_features(df)
        df = self._merge_embeddings(df)
        print(f"[INFO] Dataframe shape after merging embeddings ({self.mode}): {df.shape}")
        self.df = self._select_final_columns(df)
        self._log(f"[INFO] Final feature dataframe shape ({self.mode}): {self.df.shape}")
        return self.df


    def _prepare_base(self, df_override):
        if self.mode == "train":
            self.users = self.users.drop_duplicates(subset=["user_id"])
            self.movies = self.movies.drop_duplicates(subset=["id"])
            self.ratings.dropna(subset=["user_id", "movie_id"], inplace=True)
            self.ratings["user_id"] = pd.to_numeric(self.ratings["user_id"], errors="coerce").astype("Int64")
            self.users["user_id"] = pd.to_numeric(self.users["user_id"], errors="coerce").astype("int64")
            if self.ratings is None or self.users is None or self.movies is None:
                raise ValueError("Missing one of ratings/users/movies for training mode.")
            df = self.ratings.merge(self.users, on="user_id", how="left")
            df = df.merge(self.movies, left_on="movie_id", right_on="id", how="left")
            print(f"[INFO] Merged training data shape: {df.shape}")
        elif self.mode == "inference":
            if df_override is None:
                raise ValueError("Inference mode requires df_override.")
            df = df_override.copy()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        return df

    def _coerce_types(self, df):
        num_cols = ["age", "runtime", "popularity", "vote_average", "vote_count"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _fill_missing(self, df):
        if "age" in df.columns:
            df["age"] = df["age"].fillna(df["age"].median())

        for col in ["runtime", "popularity", "vote_average", "vote_count"]:
            if col in df.columns:
                df[col] = df[col].fillna(0)

        for cat in ["occupation", "gender", "original_language"]:
            if cat in df.columns:
                df[cat] = df[cat].fillna("unknown").astype(str)
        return df

    def _handle_dates_and_bins(self, df):
        df["release_year"] = pd.to_datetime(df.get("release_date"), errors="coerce").dt.year
        df["release_year"] = df["release_year"].fillna(df["release_year"].median())
        df["age_bin"] = pd.cut(
            df["age"],
            bins=[0, 18, 25, 35, 50, 100],
            labels=["0-18", "19-25", "26-35", "36-50", "50+"],
        )
        return df

    def _clip_outliers(self, df):
        def _clip(s, low=None, high=None):
            if s is None:
                return s
            return s.clip(lower=low, upper=high)

        if "age" in df: df["age"] = _clip(df["age"], 5, 100)
        if "runtime" in df: df["runtime"] = _clip(df["runtime"], 30, 720)
        if "vote_count" in df: df["vote_count"] = _clip(df["vote_count"], 0)
        if "release_year" in df:
            df["release_year"] = _clip(df["release_year"], 1500, pd.Timestamp.now().year + 1)
        return df

    def _encode_features(self, df):
        """Encode genres, countries, and languages as multi-hot features."""
        df["genres"] = df["genres"].fillna("unknown")
        for g in df["genres"].dropna().unique():
            if isinstance(g, str):
                for genre in [x.strip() for x in g.split(",") if x.strip()]:
                    df[f"genre_{genre}"] = df["genres"].str.contains(genre).astype(int)
        return df


    def _countries_and_languages(self, df):
        # Production countries multi-hot 
        df["production_countries"] = df["production_countries"].fillna("unknown") 
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
            lang_str = lang_str.strip().replace(" ", "")
            lang_str = lang_str.replace(",", ",").replace(";", ",")
            langs = [l.strip() for l in lang_str.split(",") if l.strip()] 
            return set([LANGUAGE_MAP.get(l, l) for l in langs]) 
        
        df["normalized_langs"] = df["spoken_languages"].fillna(df["original_language"])
        df["normalized_langs"] = df["spoken_languages"].apply(normalize_languages) 
        df["normalized_langs"] = df["normalized_langs"].fillna("und")
        
        all_langs = sorted({lang for langs in df["normalized_langs"] for lang in langs}) 
        for lang in all_langs: 
            df[f"lang_{lang}"] = df["normalized_langs"].apply(lambda x: int(lang in x))
        df.drop(columns=["normalized_langs"], inplace=True)
        return df

    def _merge_embeddings(self, df, mean_path="src/models/v2/mean_embeddings.joblib"):
        def _safe_merge(base, other, on, prefix):
            if other is None:
                return base
            rename_map = {c: f"{prefix}_{c}" for c in other.columns if c != on}
            return base.merge(other.rename(columns=rename_map), on=on, how="left")

        if self.mode == "train":
            # --- TRAIN MODE ---
            df = _safe_merge(df, self.user_explicit, "user_id", "exp_user")
            df = _safe_merge(df, self.movie_explicit, "movie_id", "exp_movie")
            df = _safe_merge(df, self.user_implicit, "user_id", "imp_user")
            df = _safe_merge(df, self.movie_implicit, "movie_id", "imp_movie")

        else:
            # --- INFERENCE MODE ---
            mean_embeds = joblib.load(mean_path)

            # Apply global mean user embeddings
            for col, val in mean_embeds["exp_user"].items():
                df[f"exp_user_{col}"] = val
            for col, val in mean_embeds["imp_user"].items():
                df[f"imp_user_{col}"] = val

            # Apply global mean movie embeddings
            for col, val in mean_embeds["exp_movie"].items():
                df[f"exp_movie_{col}"] = val
            for col, val in mean_embeds["imp_movie"].items():
                df[f"imp_movie_{col}"] = val

        return df

    def _select_final_columns(self, df):
        cols = [
            "user_id", "age", "age_bin", "occupation", "gender",
            "movie_id", "runtime", "popularity", "vote_average", "vote_count",
            "release_year", "original_language",
        ]
        if self.mode == "train" and "rating" in df:
            cols.append("rating")
        # add all embedding + genre features dynamically
        extra = [c for c in df.columns if c.startswith(("exp_", "imp_", "genre_", "lang_", "country_"))]
        return df[[c for c in cols if c in df] + extra]


if __name__ == "__main__":
    print("[INFO] Saved training_data.csv")
    data_dir = "data/raw_data"
    embedding_dir = "data/embeddings"
    fb = FeatureBuilder(
                movies_file=f"{data_dir}/movies.csv",
                ratings_file=f"{data_dir}/ratings.csv",
                users_file=f"{data_dir}/users.csv",
                user_explicit_factors=f"{embedding_dir}/user_factors_explicit.csv",
                movie_explicit_factors=f"{embedding_dir}/movie_factors_explicit.csv",
                user_implicit_factors=f"{embedding_dir}/user_factors_implicit.csv",
                movie_implicit_factors=f"{embedding_dir}/movie_factors_implicit.csv",
                mode="train"
    )
    final_df = fb.build()
    final_df.to_parquet("data/training_data.parquet", index=False)
    del final_df, fb
    gc.collect()
    print("[INFO] Saved training_data.parquet")
