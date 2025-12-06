import os, gc, psutil, shutil
import json, subprocess
import logging
import pandas as pd
from typing import Dict, Optional
from download_data import DataPipeline
from cf_trainer import CFTrainer
from feature_builder import FeatureBuilder
from trainer import Trainer

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from evaluation.Offline import offline_eval

PRIMARY_METRIC = "rmse"
LOWER_IS_BETTER = True
EPS = 1e-6
STATE_PATH = "data/retrain_state.json"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class DataRefreshManager:
    """
    Data refresh + reporting manager (DVC-compatible).

    Responsibilities:
    - Run existing DataPipeline
    - Compute dataset growth metrics
    - Emit refresh report (JSON)
    - Let DVC/Git handle versioning
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw_data")
        self.report_dir = os.path.join(data_dir, "data_refresh_reports")
        os.makedirs(self.report_dir, exist_ok=True)
        self.logger = logging.getLogger("DataRefreshManager")

    # ---------- helpers ----------

    def _safe_read_csv(self, path: str) -> Optional[pd.DataFrame]:
        if not os.path.exists(path):
            return None
        return pd.read_csv(path)

    def _count_rows(self, path: str) -> int:
        df = self._safe_read_csv(path)
        return 0 if df is None else len(df)

    def _count_unique(self, csv_path: str, key: str) -> int:
        df = self._safe_read_csv(csv_path)
        if df is None or key not in df.columns:
            return 0
        return df[key].astype(str).nunique()

    def _current_stats(self) -> Dict[str, int]:
        return {
            "users": self._count_unique(
                os.path.join(self.raw_dir, "users.csv"),
                "user_id",
            ),
            "movies": self._count_unique(
                os.path.join(self.raw_dir, "movies.csv"),
                "id",
            ),
            "interactions": self._count_rows(
                os.path.join(self.raw_dir, "watch_time.csv")
            ),
            "ratings": self._count_rows(
                os.path.join(self.raw_dir, "ratings.csv")
            ),
        }

    def _load_last_report(self) -> Optional[Dict]:
        reports = sorted(
            f for f in os.listdir(self.report_dir)
            if f.endswith(".json")
        )
        if not reports:
            return None
        with open(os.path.join(self.report_dir, reports[-1])) as f:
            return json.load(f)

    # main entrypoint

    def refresh(self, pipeline: DataPipeline) -> Dict:
        """
        Orchestrate:
        1. Run ingestion pipeline
        2. Compute dataset stats
        3. Diff vs previous report
        4. Emit refresh report
        """

        # 1. Pull new data
        pipeline.run()

        # 2. Compute current stats
        current = self._current_stats()

        # 3. Load previous stats
        prev_report = self._load_last_report()
        prev = prev_report["stats"] if prev_report else None

        deltas = {}
        if prev:
            for k in current:
                deltas[k] = current[k] - prev.get(k, 0)
        else:
            deltas = current.copy()

        # 4. Build report
        report = {
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "stats": current,
            "deltas": deltas,
        }

        # 5. Persist report
        report_name = f"refresh_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(os.path.join(self.report_dir, report_name), "w") as f:
            json.dump(report, f, indent=2)

        return report

class RetrainingManager:
    """
    Model retraining manager.

    Responsibilities:
    - Retrain collaborative filtering embeddings
    - Rebuild feature store
    - Retrain main model
    """

    def __init__(self, data_dir: str = "data", test_users: str = "data/test_data/users.csv"):
        self.data_dir = data_dir
        self.logger = logging.getLogger("RetrainingManager")
        test_df = pd.read_csv(test_users)
        self.test_ids = test_df['user_id'].unique().astype(str).tolist()

    @staticmethod
    def log_memory(step_name):
        """Log current memory usage."""
        mem = psutil.virtual_memory()
        print(f"[MEMORY] {step_name}: {mem.available / (1024**3):.2f} GB available / {mem.total / (1024**3):.2f} GB total ({mem.percent}% used)")

    @staticmethod
    def load_metric(path, metric):
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)['regression_metrics'][metric]


    # split int0 subprocesses to reduce memory bloat
    def run(self):
        print(f"Starting model retraining {pd.Timestamp.utcnow().isoformat()}...")
        import subprocess
        
        subprocess.run(["python", "-c", "from retrain_manager import RetrainingManager; RetrainingManager().retrain_cf()"],
                        cwd=BASE_DIR,
                        check=True
                        )
        gc.collect()
        self.log_memory("After CF")
        
        subprocess.run(["python", "-c", "from retrain_manager import RetrainingManager; RetrainingManager().rebuild_features()"],
                        cwd=BASE_DIR,
                        check=True)
        gc.collect()
        self.log_memory("After Features")
        

        # Tuning subprocess
        subprocess.run(
            ["python", "-c",
            "from src.retrain_manager import RetrainingManager; RetrainingManager().tune_xgboost()"],
            cwd=BASE_DIR,
            check=True,
        )

        self.log_memory("After Tuning subprocess")

        # Training subprocess (clean memory)
        subprocess.run(
            ["python", "-c",
            "from src.retrain_manager import RetrainingManager; RetrainingManager().train_xgboost()"],
            cwd=BASE_DIR,
            check=True,
        )

        self.log_memory("After Training subprocess")

        # evaluate the model
        print("Starting model evaluation...")
        subprocess.run(["python", "-c", "from retrain_manager import RetrainingManager; RetrainingManager().evaluate_model()"])
        self.log_memory("After Evaluation")
        print("Model evaluation complete.")
        self.ensure_dvc_initialized()
        # track model artifact
        subprocess.run(
            ["dvc", "add", "src/models/candidate"],
            cwd=BASE_DIR,
            check=True,
        )

        # track training data
        subprocess.run(
            ["dvc", "add", "data/training_data.parquet"],
            cwd=BASE_DIR,
            check=True,
        )

        self.register_candidate_provenance()
        promoted = self.promote_if_better()
        return bool(promoted)

    @staticmethod
    def ensure_dvc_initialized():
        if not os.path.exists(".dvc"):
            subprocess.run(["dvc", "init", "--no-scm", "-f"], check=True)
            subprocess.run(["dvc", "config", "core.no_scm", "true"], check=True)

    def retrain_cf(self):
        config = {
            "ratings_csv": f"{self.data_dir}/raw_data/ratings.csv",
            "watch_csv": f"{self.data_dir}/raw_data/watch_time.csv",
            "movies_csv": f"{self.data_dir}/raw_data/movies.csv",
            "test_ids": self.test_ids or [],
            "out_dir": f"{self.data_dir}/embeddings",
            "mean_embed_out": "src/models/candidate/mean_embeddings.joblib",
            # SVD
            "svd_factors": 50, "svd_epochs": 30, "svd_lr": 0.005, "svd_reg": 0.02,
            # ALS
            "als_factors": 50, "als_iters": 20, "als_reg": 0.01, "alpha": 80.0, "w1": 0.7, "w2": 0.3,
            "seed": 42,
        }
        self.cf_trainer = CFTrainer(config)
        self.cf_trainer.run()
        del self.cf_trainer
        gc.collect()
    
    def rebuild_features(self):
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
                    test_ids=self.test_ids or [],
                    mode="train"
        )
        final_df = fb.build()
        final_df.to_parquet('data/training_data.parquet')
        print(f"[INFO] Saved data/training_data.parquet with shape: {final_df.shape}")
        # Clean up
        del final_df, fb
        gc.collect()

    def tune_xgboost(self):
        train_data = os.path.join(BASE_DIR, "data/training_data.parquet")
        output_dir = os.path.join(BASE_DIR, "src/models/candidate")
        os.makedirs(output_dir, exist_ok=True)

        trainer = Trainer(data_file=train_data, target="rating")
        print("Starting hyperparameter tuning...")

        df = trainer.load_data()
        print(f"[INFO] Loaded training data with shape: {df.shape}")
        df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)

        best_params = trainer.tune(
            tuning_file=os.path.join(output_dir, "tuning_results.json"),
            tune_df=df,
        )

        del df, trainer
        gc.collect()

        print("Tuning finished")


    def train_xgboost(self):
        train_data = os.path.join(BASE_DIR, "data/training_data.parquet")
        output_dir = os.path.join(BASE_DIR, "src/models/candidate")
        tuning_file = os.path.join(output_dir, "tuning_results.json")

        # with open(tuning_file) as f:
        #     best_params = json.load(f)
        # model_params = {k.replace("model__", ""): v for k, v in best_params.items()}

        trainer = Trainer(data_file=train_data, target="rating")
        trainer.train(tuning_file=tuning_file)
        trainer.save(output_dir=output_dir)

        del trainer
        gc.collect()

        print("Model training completed")

    
    def evaluate_model(self):
        # Paths to models, preprocessor, and data
        preproc_path = os.path.join("src", "models", "candidate", "preprocessor.joblib")
        model_path = os.path.join("src", "models", "candidate", "xgb_model.joblib")
        eval_data = os.path.join("data", "test_data", "offline_eval_data.parquet")
        results_path = os.path.join("src", "models", "candidate", "evaluation_results.json")

        # Call offline_eval's evaluate function
        results, y_test, preds = offline_eval.evaluate(
            preproc_path=preproc_path,
            model_path=model_path,
            eval_data=eval_data,
            results_path=results_path
        )

        print("\nOffline evaluation finished from evaluation.offline")
        print("Results preview:", results)

    
    def promote_if_better(self):
        metric = PRIMARY_METRIC

        cand = self.load_metric("src/models/candidate/evaluation_results.json", metric)
        v2   = self.load_metric("src/models/v2/evaluation_results.json", metric)
        v1   = self.load_metric("src/models/v1/evaluation_results.json", metric)

        assert cand is not None, "Candidate model has no eval results"

        def better(a, b):
            if b is None:
                return True
            if LOWER_IS_BETTER:
                return a < b - EPS
            return a > b + EPS

        # ---- Decision tree ----
        if better(cand, v2):
            self.logger.info("Candidate beats v2 → promote to v2")
            self._shift_models(new="candidate", to_v2=True)
            return True
        elif better(cand, v1):
            self.logger.info("Candidate beats v1 only → promote to v1")
            self._shift_models(new="candidate", to_v2=False)
            return True
        else:
            self.logger.info("Candidate rejected (no improvement)")
            return False

    @staticmethod
    def _shift_models(new, to_v2):
        if to_v2:
            if os.path.exists("src/models/v2"):
                shutil.rmtree("src/models/v1", ignore_errors=True)
                shutil.move("src/models/v2", "src/models/v1")
            shutil.move(f"src/models/{new}", "src/models/v2")
        else:
            shutil.rmtree("src/models/v1", ignore_errors=True)
            shutil.move(f"src/models/{new}", "src/models/v1")

    @staticmethod
    def load_retrain_state():
        if not os.path.exists(STATE_PATH):
            return {"users_seen": 0, "interactions_seen": 0}
        with open(STATE_PATH) as f:
            return json.load(f)

    @staticmethod
    def save_retrain_state(stats):
        with open(STATE_PATH, "w") as f:
            json.dump({
                "users_seen": stats["users"],
                "interactions_seen": stats["interactions"],
                "last_retrain_ts": pd.Timestamp.utcnow().isoformat()
            }, f, indent=2)

    @staticmethod
    def _get_dvc_md5(dvc_path: str) -> str:
        import yaml
        try:
            with open(dvc_path) as f:
                meta = yaml.safe_load(f)
            return meta["outs"][0]["md5"]
        except Exception:
            return "unknown"

    
    def register_candidate_provenance(self):
        from src.provenance import register_model, register_training_data

        # 1. Training data provenance
        data_id = register_training_data({
            "data_source": "data/raw_data",
            "file_path": "data/training_data.parquet",
            "row_count": None,   # optional
        })

        # 2. DVC hash for candidate model
        dvc_md5 = self._get_dvc_md5("src/models/candidate.dvc")

        # 3. Evaluation metrics
        with open("src/models/candidate/evaluation_results.json") as f:
            eval_results = json.load(f)

        # 4. Register model
        model_version = register_model({
            "artifact_path": f"src/models/candidate (md5={dvc_md5}",
            "training_data_id": data_id,
            "metrics_json": eval_results,
            "model_type": "XGBoost",
        })

        self.logger.info(f"Registered candidate model {model_version}")
        return model_version


# model retraining entrypoint
if __name__ == "__main__":
    # refresh data
    pipeline = DataPipeline()
    manager = DataRefreshManager(data_dir="data")
    report = manager.refresh(pipeline)
    print(json.dumps(report, indent=2))

    # initialise retrain manager
    retrain_manager = RetrainingManager(data_dir="data")

    # check new data quantity
    new_users = report['stats']['users']
    new_interactions = report['stats']['interactions']

    state = retrain_manager.load_retrain_state()

    unseen_users = new_users - state["users_seen"]
    unseen_interactions = new_interactions - state["interactions_seen"]

    print(f"- Total new users from last retrain: {new_users}")
    print(f"- Total new from last interactions: {new_interactions}")

    if new_users < 50 and new_interactions < 100:
        print("Insufficient new data for retraining. Exiting.")
        exit(0)

    # retrain and asave new model
    retrain_manager.run()
    # save retrain state
    retrain_manager.save_retrain_state(current)
