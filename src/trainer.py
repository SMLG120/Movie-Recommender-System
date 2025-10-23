import json, time, datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


class Trainer:
    def __init__(self, data_file="training_data.csv", target="rating", test_size=0.2,
                random_state=42, importance_out=None, metrics_out=None):

        self.data_file = data_file
        self.target = target
        self.test_size = test_size
        self.random_state = random_state
        self.importance_out = importance_out or "src/train_results/feature_importance.csv"
        self.metrics_out = metrics_out or "src/train_results/metrics.json"

        # Will be filled later
        self.df = None
        self.model = None
        self.pipeline = None

    def load_data(self):
        """Load dataset from CSV"""
        self.df = pd.read_csv(self.data_file)
        print(f"[INFO] Loaded dataset with shape {self.df.shape}")

    def prepare_features(self): # renamed to avoid confusion with preprocessor
        # Categorical -> one-hot
        categorical = ["age_bin", "occupation", "gender", "original_language"]

        # Everything else is numeric (includes multi-hot + CF embeddings)
        all_cols = self.df.columns.tolist()
        ignore = set(["user_id", "movie_id", self.target] + categorical)
        numeric = [c for c in all_cols if c not in ignore]

        # Split
        X = self.df[categorical + numeric]
        y = self.df[self.target]

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("num", "passthrough", numeric),
            ],
            remainder="drop",
            verbose_feature_names_out=False
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

    def tune(self, tuning_file="tuning_results.json"):
        """Hyperparameter tuning with GridSearchCV, logs separately"""
        self.load_data()
        X, y, preprocessor, *_ = self.prepare_features()

        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=self.random_state
        )

        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        param_grid = {
            "model__n_estimators": [200, 400],
            "model__max_depth": [4, 6, 8],
            "model__learning_rate": [0.05, 0.1],
            "model__subsample": [0.7, 0.9],
            "model__colsample_bytree": [0.7, 0.9],
            "model__min_child_weight": [1, 3, 5],
            "model__reg_lambda": [0.1, 1.0]
        }

        start = time.time()
        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=3,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=2
        )
        grid.fit(X_train, y_train)
        elapsed = time.time() - start

        # Prepare tuning results
        results = {
            "timestamp": datetime.datetime.now().isoformat(),
            "best_params": grid.best_params_,
            "best_cv_rmse": -grid.best_score_,
            "tuning_time_sec": round(elapsed, 2),
            "cv_folds": grid.cv,
            "param_grid_size": len(list(ParameterGrid(param_grid)))
        }

        print(f"[TUNING] Best CV RMSE: {-grid.best_score_:.4f}")
        print(f"[TUNING] Best Params: {grid.best_params_}")

        # Append to tuning log
        try:
            with open(tuning_file, "r") as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = []

        all_results.append(results)

        with open(tuning_file, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"[INFO] Tuning results appended to {tuning_file}")

        self.best_model = grid.best_estimator_
        return self.best_model

    def train(self, train_results="training_results.json", tuning_file=None, experiment_name=None):
        """Train XGBoost with pipeline and log metadata"""
        self.load_data()
        X, y, preprocessor, categorical, numeric = self.prepare_features()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Model
        best_params = self.load_best_params(tuning_file) if tuning_file else None
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            random_state=self.random_state,
            **({k.replace("model__", ""): v for k, v in best_params.items()} if best_params else
            dict(n_estimators=300, learning_rate=0.1, max_depth=8, subsample=0.8, colsample_bytree=0.8))
        )

        self.pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb_model)])

        # --- Training ---
        start_train = time.time()
        self.pipeline.fit(X_train, y_train)
        train_time = time.time() - start_train

        # --- Evaluation ---
        start_infer = time.time()
        preds = self.pipeline.predict(X_test)
        infer_time = (time.time() - start_infer) / len(X_test)  # avg per sample

        rmse = mean_squared_error(y_test, preds, squared=False)
        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)

        # Extra ranking-ish sanity checks
        try:
            spearman = pd.Series(y_test).rank().corr(pd.Series(preds).rank(), method="spearman")
            pearson  = pd.Series(y_test).corr(pd.Series(preds), method="pearson")
        except Exception:
            spearman, pearson = None, None

        # --- Metadata ---
        results = {
        "experiment": experiment_name or f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}",
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": {"rmse": rmse, "mae": mae, "r2": r2, "spearman": spearman, "pearson": pearson},
        "train_time_sec": round(train_time, 3),
        "avg_infer_time_sec": round(infer_time, 6),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_features_in": int(X_train.shape[1]),
        "hyperparameters": xgb_model.get_params()
        }

        print(f"[RESULTS] RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        print(f"[INFO] Train time: {train_time:.2f}s, Avg inference: {infer_time*1000:.4f} ms/sample")

        # --- Save Features and Results ---
        try:
            with open(train_results, "r") as f:
                all_results = json.load(f)
        except FileNotFoundError:
            all_results = []

        all_results.append(results)

        with open(train_results, "w") as f:
            json.dump(all_results, f, indent=2)

        print(f"[INFO] Results appended to {train_results}")

        try:
            # Recover model-native importances after fitting
            booster = self.pipeline.named_steps["model"]
            importances = getattr(booster, "feature_importances_", None)
            if importances is not None:
                # Get transformed feature names
                feat_names = self.pipeline.named_steps["preprocessor"].get_feature_names_out()
                fi = (pd.DataFrame({"feature": feat_names, "importance": importances})
                        .sort_values("importance", ascending=False))
                os.makedirs(os.path.dirname(self.importance_out), exist_ok=True)
                fi.to_csv(self.importance_out, index=False)
        except Exception:
            pass

        return rmse, mae, r2


    def save(self, model_path="xgb_recommender.joblib"):
        """Save trained pipeline"""
        if self.pipeline:
            # joblib.dump(self.pipeline, model_path)
            joblib.dump(self.pipeline.named_steps["preprocessor"], "src/models/preprocessor.joblib")
            joblib.dump(self.pipeline.named_steps["model"], "src/models/xgb_model_only.joblib")
            print(f"[INFO] Model saved to {model_path}")
        else:
            print("[WARN] No trained pipeline to save.")


if __name__ == "__main__":
    trainer = Trainer(data_file="data/training_data_v2.csv")
    trainer.tune(tuning_file="src/train_results/tuning_results.json")
    trainer.train(train_results="src/train_results/training_results.json",
                  tuning_file="src/train_results/tuning_results.json",
                  experiment_name="xgb_recommender_v2")
    trainer.save(model_path="src/models/xgb_recommender.joblib")
