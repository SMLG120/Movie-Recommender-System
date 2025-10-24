import os, json, time, datetime, joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Trainer:
    def __init__(
        self,
        data_file="training_data.csv",
        target="rating",
        test_size=0.2,
        random_state=42,
        importance_out="src/train_results/feature_importance.csv",
        metrics_out="src/train_results/metrics.json",
        reader=None,
        writer=None,
        logger=None,
        model_factory=None
    ):
        """
        Args:
            reader: optional function to read CSVs (mock for tests)
            writer: optional function to write JSON/CSV (mock for tests)
            logger: function for logging (print replacement)
            model_factory: callable returning an estimator (for mocking xgb)
        """
        self.data_file = data_file
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.importance_out = importance_out
        self.metrics_out = metrics_out
        self._read_csv = reader or pd.read_csv
        self._write_file = writer or self._default_writer
        self._log = logger or (lambda m: print(m))
        self._model_factory = model_factory or self._default_xgb_factory

        self.df = None
        self.pipeline = None

    def _default_writer(self, path, data):
        """Dump data as JSON or CSV based on extension."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.endswith(".json"):
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        elif path.endswith(".csv"):
            data.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {path}")

    def _default_xgb_factory(self, **params):
        return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", n_jobs=-1, **params)

    def load_data(self):
        self.df = self._read_csv(self.data_file)
        self._log(f"[INFO] Loaded dataset with shape {self.df.shape}")
        return self.df

    def prepare_features(self):
        categorical = ["age_bin", "occupation", "gender", "original_language"]
        all_cols = self.df.columns.tolist()
        ignore = set(["user_id", "movie_id", self.target] + categorical)
        numeric = [c for c in all_cols if c not in ignore]

        X = self.df[categorical + numeric]
        y = self.df[self.target]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("num", "passthrough", numeric),
            ],
            verbose_feature_names_out=False,
        )
        return X, y, preprocessor, categorical, numeric

    def train(self, tuning_params=None):
        """Train pipeline with injected or default params."""
        if self.df is None:
            self.load_data()

        X, y, preprocessor, _, _ = self.prepare_features()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        params = tuning_params or dict(
            n_estimators=100, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8
        )
        model = self._model_factory(**params)

        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        # train
        start_train = time.time()
        self.pipeline.fit(X_train, y_train)
        train_time = time.time() - start_train

        preds = self.pipeline.predict(X_test)
        infer_time = (time.time() - start_train) / max(len(X_test), 1)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
            "train_time_sec": round(train_time, 3),
            "avg_infer_time_sec": round(infer_time, 6),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
        }

        self._log(f"[RESULTS] RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
        self._write_file(self.metrics_out, results)
        return results

    def save(self, output_dir="src/models"):
        if not self.pipeline:
            self._log("[WARN] No trained pipeline to save.")
            return
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.pipeline.named_steps["preprocessor"], f"{output_dir}/preprocessor.joblib")
        joblib.dump(self.pipeline.named_steps["model"], f"{output_dir}/xgb_model.joblib")
        self._log(f"[INFO] Saved model to {output_dir}")


if __name__ == "__main__":
    train_data = "data/training_data_v2.csv"
    trainer = Trainer(data_file=train_data, target="rating")
    output_dir = "src/train_results/"
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(train_data)
    tuning_df = df.sample(frac=0.4, random_state=42)

    trainer.tune(tuning_file="src/train_results/tuning_results.json", tune_df=tuning_df)
    trainer.train(train_results="src/train_results/training_results.json",
                  tuning_file="src/train_results/tuning_results.json",
                  experiment_name="xgb_recommender_v2")
    trainer.save(model_path="src/models/xgb_recommender.joblib")
