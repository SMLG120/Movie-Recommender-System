import os, json, time, datetime, joblib, gc
import numpy as np
import pandas as pd
import xgboost as xgb
import scipy.stats as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings("ignore")


class Trainer:
    def __init__(
        self,
        data_file="training_data.parquet",
        target="rating",
        test_size=0.1,
        random_state=42,
        metrics_out="src/models/candidate/metrics.json",
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
        self.metrics_out = metrics_out
        self._read_csv = reader or pd.read_parquet
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
        return xgb.XGBRegressor(objective="reg:squarederror", eval_metric="rmse", tree_method="hist", n_jobs=2, **params)

    def optimize_df_memory(self, df):
        for col in df.columns:
            col_type = df[col].dtype
            
            # Optimize integers
            if str(col_type).startswith('int'):
                # Convert to smaller int types (int8, int16, int32)
                df[col] = pd.to_numeric(df[col], downcast='integer')
            
            # Optimize floats
            elif str(col_type).startswith('float'):
                # Convert to float32
                df[col] = pd.to_numeric(df[col], downcast='float')

        return df
    
    def load_data(self):
        self.df = self._read_csv(self.data_file)
        self._log(f"[INFO] Loaded dataset with shape {self.df.shape}")
        self.df = self.optimize_df_memory(self.df)
        return self.df

    def _shuffle(self, df, seed=42, max_rows=None):
        np.random.seed(seed) # for reproducibility
        random_keys = np.random.rand(len(df))
        shuffled_df = df.set_index(random_keys).sort_index().reset_index(drop=True)
        if max_rows is not None:
            shuffled_df = shuffled_df.iloc[:max_rows]
        return shuffled_df

    def prepare_features(self, data=None):
        categorical = ["age_bin", "occupation", "gender", "original_language"]
        self.df = data if data is not None else self.df

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

    def load_best_params(self, tuning_file="tuning_results.json"):
        """Load most recent tuned parameters from file."""
        try:
            with open(tuning_file, "r") as f:
                all_results = json.load(f)
            return all_results[-1]["best_params"]
        except (FileNotFoundError, KeyError, IndexError):
            print("[WARN] No tuned params found, using defaults.")
            return None


    def tune(self, tuning_file="tuning_results.json", tune_df=None, param_distributions=None, n_iter=10, cv=2, lightweight=True):
        """
        Optional hyperparameter tuning step.
        - Set lightweight=True to run a quick test pass (used in CI).
        """
        self.df = tune_df if tune_df is not None else self.load_data()
        print("[INFO] Preparing features for tuning...")
        X, y, preprocessor, _, _ = self.prepare_features()
        X_train, _, y_train, _ = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        for col in X_train.select_dtypes(include=["int", "float"]).columns:
            X_train[col] = X_train[col].astype(np.float32)
        print(f"[INFO] Tuning on data shape: {X_train.shape}")

        param_distributions = param_distributions or {
            "model__n_estimators": sp_randint(50, 200) if lightweight else sp_randint(100, 400),
            "model__max_depth": sp_randint(2, 3) if lightweight else sp_randint(3, 8),
            "model__learning_rate": [0.05, 0.1] if lightweight else [0.03, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }

        model = self._model_factory()
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

        self._log(f"[TUNE] Running RandomizedSearchCV (n_iter={n_iter}, cv={cv})")
        print(f"[INFO] Param distributions: {param_distributions}")
        grid = RandomizedSearchCV(
            pipe,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring="neg_root_mean_squared_error",
            random_state=self.random_state,
            n_jobs=1,
            verbose=2 if not lightweight else 0,
        )
        start = time.time()
        grid.fit(X_train, y_train)
        elapsed = round(time.time() - start, 2)
        print(f"[INFO] Tuning completed in {elapsed} sec.")
        self.best_params_ = grid.best_params_
        best_rmse = -grid.best_score_
        print(f"[INFO] Best RMSE: {best_rmse:.3f} with params: {self.best_params_}")

        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "best_params": self.best_params_,
            "best_cv_rmse": best_rmse,
            "tuning_time_sec": elapsed,
            "cv_folds": cv,
            "param_grid_size": n_iter,
        }

        self._log(f"[TUNE] Best RMSE={best_rmse:.3f}, Params={self.best_params_}")

        # --- Save to tuning log ---
        try:
            with open(tuning_file, "r") as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = []

        all_results.append(results)
        os.makedirs(os.path.dirname(tuning_file) or ".", exist_ok=True)
        with open(tuning_file, "w") as f:
            json.dump(all_results, f, indent=2)

        del X_train, y_train, grid
        gc.collect()

        self._log(f"[INFO] Tuning results appended to {tuning_file}")
        return self.best_params_

    def _evaluate(self, X_test, y_test, model=None):
        if model:
            self.pipeline = model

        start_infer = time.time()
        preds = self.pipeline.predict(X_test)
        infer_time = (time.time() - start_infer) / max(len(X_test), 1)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "metrics": {"rmse": rmse, "mae": mae, "r2": r2},
            "avg_infer_time_sec": round(infer_time, 6),
            "test_samples": int(len(X_test)),
        }
        return results, y_test, preds


    def train(self, tuning_file='src/train_results/tuning_results.json'):
        """Train pipeline with injected or default params."""
        print("[INFO] Loading data for training...")
        self.load_data()
        print("[INFO] Shuffling data...")
        self.df = self._shuffle(self.df, seed=self.random_state, max_rows=100_000)
        print("[INFO] Preparing features for training...")

        # self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

        X, y, preprocessor, _, _ = self.prepare_features()
        print(f"[INFO] Full data shape: {X.shape}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        num_cols = X.select_dtypes(include=["int", "float"]).columns
        X[num_cols] = X[num_cols].astype(np.float32, copy=False)

        print(f"[INFO] Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

        del X, y, self.df
        gc.collect()

        self.best_params = self.load_best_params(tuning_file)
        params = self.best_params or dict(
            n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8
        )

        model = self._model_factory(**params)

        # Add memory-saving parameters
        params['tree_method'] = 'hist'  # Much more memory efficient than default
        params['max_bin'] = 128  # Reduce from default 256
        params['predictor'] = 'cpu_predictor'  # Explicit CPU mode
        params['max_cat_to_onehot'] = 4
        params['enable_categorical'] = False
        params['max_leaves'] = 16        # overrides depth-based growth
        params['grow_policy'] = "lossguide"

        print(f"[INFO] Training with params: {params}")
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])


        # for col in X_train.select_dtypes(include=["int", "float"]).columns:
        #     X_train[col] = X_train[col].astype(np.float32)
        #     X_test[col] = X_test[col].astype(np.float32)

        # train
        print("[INFO] Starting training...")
        start_train = time.time()
        self.pipeline.fit(X_train, y_train)
        train_time = time.time() - start_train
        train_samples = int(len(X_train))
        del X_train, y_train
        gc.collect()
        
        print(f"[INFO] Training completed in {train_time:.2f} sec.")
        results, y_test, preds = self._evaluate(X_test, y_test, model=self.pipeline)
        results["training_time_sec"] = round(train_time, 2)
        results["train_samples"] =  train_samples

        del y_test, preds
        gc.collect()

        self._log(f"[RESULTS] RMSE={results['metrics']['rmse']:.3f}, MAE={results['metrics']['mae']:.3f}, R2={results['metrics']['r2']:.3f}")
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
    tuning_df = df.sample(frac=0.4, random_state=42).reset_index(drop=True)

    # Tune hyperparameters
    best_params = trainer.tune(tuning_file="src/train_results/tuning_results.json", tune_df=tuning_df)
    
    # Extract model params (remove 'model__' prefix)
    model_params = {k.replace('model__', ''): v for k, v in best_params.items()}
    
    trainer.train(tuning_params=model_params)
    trainer.save(output_dir="src/models")
