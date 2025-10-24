import os
import json
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from scipy import sparse
from surprise import Dataset, Reader, SVD
import implicit

import argparse
import warnings

warnings.filterwarnings("ignore")


class CFTrainer:
    def __init__(
        self,
        config: dict,
        reader: callable = pd.read_csv,
        writer: callable = None,
        json_writer: callable = None,
        svd_cls=SVD,
        als_cls=implicit.als.AlternatingLeastSquares,
        logger: logging.Logger | None = None,
    ):
        """
        Collaborative Filtering trainer.
        Allows dependency injection for I/O, models, and logging.
        """
        self.config = config
        self.reader = reader
        self.writer = writer or self._save_csv
        self.json_writer = json_writer or self._save_json
        self.svd_cls = svd_cls
        self.als_cls = als_cls
        self.logger = logger or logging.getLogger("CFTrainer")

        self.out_dir = config.get("out_dir", "data/embeddings")
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(self.out_dir, "maps")).mkdir(parents=True, exist_ok=True)

    def _save_csv(self, df, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        self.logger.info(f"Saved CSV → {path}")

    def _save_json(self, obj, path):
        """Safely write JSON (with key normalization)."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        def fix_keys(d):
            if isinstance(d, dict):
                return {str(k): fix_keys(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [fix_keys(i) for i in d]
            elif isinstance(d, np.generic):
                return d.item()
            return d

        with open(path, "w") as f:
            json.dump(fix_keys(obj), f, indent=2)
        self.logger.info(f"Saved JSON → {path}")

    # Explicit CF (SVD) 
    def train_explicit(self):
        cfg = self.config
        ratings_csv = cfg["ratings_csv"]
        self.logger.info(f"[EXPLICIT] Loading ratings from {ratings_csv}")
        ratings = self.reader(ratings_csv).dropna(subset=["user_id", "movie_id", "rating"])
        ratings["user_id"] = ratings["user_id"].astype(str)
        ratings["movie_id"] = ratings["movie_id"].astype(str)

        reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
        data = Dataset.load_from_df(ratings[["user_id", "movie_id", "rating"]], reader)
        trainset = data.build_full_trainset()

        algo = self.svd_cls(
            n_factors=cfg.get("svd_factors", 50),
            n_epochs=cfg.get("svd_epochs", 30),
            lr_all=cfg.get("svd_lr", 0.005),
            reg_all=cfg.get("svd_reg", 0.02),
            random_state=cfg.get("seed", 42),
        )
        self.logger.info("[EXPLICIT] Training SVD model...")
        algo.fit(trainset)

        user_map = {trainset.to_raw_uid(i): int(i) for i in trainset.all_users()}
        item_map = {trainset.to_raw_iid(j): int(j) for j in trainset.all_items()}

        user_f = pd.DataFrame(
            [
                [uid] + list(algo.pu[user_map[uid]])
                for uid in ratings["user_id"].unique() if uid in user_map
            ],
            columns=["user_id"] + [f"exp_f{i+1}" for i in range(algo.pu.shape[1])]
        )
        item_f = pd.DataFrame(
            [
                [iid] + list(algo.qi[item_map[iid]])
                for iid in ratings["movie_id"].unique() if iid in item_map
            ],
            columns=["movie_id"] + [f"exp_f{i+1}" for i in range(algo.qi.shape[1])]
        )

        self.writer(user_f, f"{self.out_dir}/user_factors_explicit.csv")
        self.writer(item_f, f"{self.out_dir}/movie_factors_explicit.csv")
        self.json_writer({"user_map": user_map, "item_map": item_map},
                         f"{self.out_dir}/maps/explicit_maps.json")
        self.logger.info("[EXPLICIT] Finished SVD training")

    # Implicit CF (ALS)
    def _build_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute weighted implicit confidence scores."""
        alpha, w1, w2 = self.config.get("alpha", 80.0), 0.7, 0.25
        df["interaction_count"] = pd.to_numeric(df["interaction_count"], errors="coerce").fillna(0)
        df["max_minute_reached"] = pd.to_numeric(df["max_minute_reached"], errors="coerce").fillna(0)
        df["movie_duration"] = pd.to_numeric(df["movie_duration"], errors="coerce").fillna(100)

        df["completion"] = (df["max_minute_reached"] / df["movie_duration"]).clip(0, 1)
        df["freq_norm"] = np.log1p(df["interaction_count"]) / np.log1p(df["movie_duration"].clip(lower=1))
        df["score"] = w1 * df["completion"] + w2 * df["freq_norm"]
        df["confidence"] = 1 + alpha * df["score"]
        return df[["user_id", "movie_id", "confidence"]]

    def train_implicit(self):
        cfg = self.config
        watch = self.reader(cfg["watch_csv"])
        movies = self.reader(cfg["movies_csv"], usecols=["id", "runtime"]).dropna(subset=["id"])

        watch = (
            watch.merge(movies, how="left", left_on="movie_id", right_on="id", validate="many_to_one")
            .rename(columns={"runtime": "movie_duration"})
            .drop(columns="id")
        )

        watch.dropna(subset=["movie_duration"], inplace=True)
        conf = self._build_confidence(watch)

        users = conf["user_id"].unique()
        items = conf["movie_id"].unique()
        u2i = {u: i for i, u in enumerate(users)}
        i2i = {m: i for i, m in enumerate(items)}

        conf["row"] = conf["user_id"].map(u2i)
        conf["col"] = conf["movie_id"].map(i2i)
        conf.dropna(subset=["row", "col"], inplace=True)
        conf["confidence"] = conf["confidence"].fillna(0)

        rows, cols = conf["row"].astype(int), conf["col"].astype(int)
        mat = sparse.coo_matrix((conf["confidence"], (cols, rows)), shape=(len(i2i), len(u2i))).tocsr()

        model = self.als_cls(
            factors=cfg.get("als_factors", 50),
            regularization=cfg.get("als_reg", 0.01),
            iterations=cfg.get("als_iters", 20),
            use_cg=True,
            use_native=True,
        )

        self.logger.info("[IMPLICIT] Training ALS model...")
        model.fit(mat.T.tocsr())

        user_f = pd.DataFrame(model.user_factors, columns=[f"imp_f{i+1}" for i in range(model.factors)])
        item_f = pd.DataFrame(model.item_factors, columns=[f"imp_f{i+1}" for i in range(model.factors)])

        user_f.insert(0, "user_id", list(u2i.keys()))
        item_f.insert(0, "movie_id", list(i2i.keys()))

        self.writer(user_f, f"{self.out_dir}/user_factors_implicit.csv")
        self.writer(item_f, f"{self.out_dir}/movie_factors_implicit.csv")
        self.json_writer({"user_map": u2i, "item_map": i2i},
                         f"{self.out_dir}/maps/implicit_maps.json")
        self.logger.info("[IMPLICIT] Finished ALS training")

    def run(self, run_explicit=True, run_implicit=True):
        """Run full CF training with optional components."""
        self.logger.info("==== Starting Collaborative Filtering Training ====")
        if run_explicit:
            try:
                self.train_explicit()
            except Exception as e:
                self.logger.warning(f"[WARN] Explicit CF failed: {e}")
        if run_implicit:
            try:
                self.train_implicit()
            except Exception as e:
                self.logger.warning(f"[WARN] Implicit CF failed: {e}")
        self.logger.info("==== All CF models complete ====")


# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Collaborative Filtering models (SVD + ALS).")

    # ---- Paths ----
    parser.add_argument("--ratings_csv", type=str, default="data/raw_data/ratings.csv", help="Path to ratings CSV file.")
    parser.add_argument("--watch_csv", type=str, default="data/raw_data/watch_time.csv", help="Path to watch time CSV file.")
    parser.add_argument("--movies_csv", type=str, default="data/raw_data/movies.csv", help="Path to movies CSV file.")
    parser.add_argument("--out_dir", type=str, default="data/embeddings", help="Output directory for embeddings.")

    # ---- Explicit CF (SVD) ----
    parser.add_argument("--svd_factors", type=int, default=50, help="Number of latent factors for SVD.")
    parser.add_argument("--svd_epochs", type=int, default=30, help="Number of training epochs for SVD.")
    parser.add_argument("--svd_lr", type=float, default=0.005, help="Learning rate for SVD.")
    parser.add_argument("--svd_reg", type=float, default=0.02, help="Regularization term for SVD.")

    # ---- Implicit CF (ALS) ----
    parser.add_argument("--als_factors", type=int, default=50, help="Number of latent factors for ALS.")
    parser.add_argument("--als_iters", type=int, default=20, help="Number of ALS iterations.")
    parser.add_argument("--als_reg", type=float, default=0.01, help="Regularization term for ALS.")
    parser.add_argument("--alpha", type=float, default=80.0, help="Confidence scaling factor for implicit feedback.")
    parser.add_argument("--w1", type=float, default=0.7, help="Weight for completion ratio in confidence score.")
    parser.add_argument("--w2", type=float, default=0.3, help="Weight for interaction frequency in confidence score.")

    # ---- General ----
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    config = vars(args)  # convert argparse Namespace → dict
    CFTrainer(config).run()
