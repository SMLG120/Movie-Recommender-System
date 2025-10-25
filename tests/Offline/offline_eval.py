import argparse
import json
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    r2_score,
)
from tqdm import tqdm
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import time

# Add project root to Python path (so imports like src.trainer work)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Import Trainer
from src.trainer import Trainer


# =====================================================
# 🧮 METRICS HELPERS
# =====================================================
def regression_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def classification_metrics(y_true, y_pred):
    """Calculate classification metrics for rating threshold"""
    y_true_bin = (y_true >= 3).astype(int)
    y_pred_bin = (y_pred >= 3).astype(int)

    precision = precision_score(y_true_bin, y_pred_bin)
    recall = recall_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)
    accuracy = accuracy_score(y_true_bin, y_pred_bin)

    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


# =====================================================
# 🧪 MAIN EVALUATION FUNCTION
# =====================================================
def evaluate(
    preproc_path="src/models/preprocessor.joblib",
    model_path="src/models/xgb_model.joblib",
    eval_data="data/training_data_v2.csv",
    results_path="evaluation_results.json",
):
    """Run evaluation of recommenders on test data"""
    # Load model and preprocessor
    print(f"\n🔹 Loading preprocessor from: {preproc_path}")
    preprocessor = joblib.load(preproc_path)

    # --- Debug inspection to detect broken preprocessor ---
    print("Preprocessor type:", type(preprocessor))
    if hasattr(preprocessor, "transformers"):
        print("Transformers in preprocessor:")
        for name, transformer, cols in preprocessor.transformers:
            print(f"  - {name}: {type(transformer)} on columns {cols}")
            if isinstance(transformer, str):
                raise TypeError(
                    f"❌ Invalid preprocessor: transformer '{name}' is a string ('{transformer}'). "
                    f"Please rebuild the preprocessor using actual sklearn transformers like OneHotEncoder or StandardScaler."
                )
    else:
        print("⚠️ Warning: preprocessor has no 'transformers' attribute. Check your joblib file.")

    print(f"\n🔹 Loading XGBoost model from: {model_path}")
    xgb_model = joblib.load(model_path)

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb_model)])

    # Load and prepare dataset
    print(f"\n🔹 Loading evaluation data from: {eval_data}")
    trainer = Trainer()
    trainer.df = pd.read_csv(eval_data)
    X, y, _, _, _ = trainer.prepare_features()

    # Split data 80-20
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get predictions
    print("\n🔹 Running inference...")
    start_time = time.time()
    preds = pipeline.predict(X_test)
    inference_time = (time.time() - start_time) / max(len(X_test), 1)

    # Calculate metrics
    reg_metrics = regression_metrics(y_test, preds)
    class_metrics = classification_metrics(y_test, preds)

    # Combine all results
    results = {
        "regression_metrics": reg_metrics,
        "classification_metrics": class_metrics,
        "inference_time": inference_time,
    }

    # Save results
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print("\n✅ Evaluation completed successfully.")
    print("Results saved to:", results_path)
    print(json.dumps(results, indent=4))

    return results, y_test, preds


# =====================================================
# 🏃‍♂️ DIRECT EXECUTION (for testing)
# =====================================================
if __name__ == "__main__":
    # Get the project root directory (team-6)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # ✅ Correct paths
    preproc_path = os.path.join(project_root, "src", "models", "preprocessor.joblib")
    model_path = os.path.join(project_root, "src", "models", "xgb_model.joblib")
    eval_data = os.path.join(project_root, "data", "training_data_v2.csv")
    results_path = os.path.join(
        project_root, "tests", "Offline", "evaluation_results.json"
    )

    # Run evaluation
    evaluate(
        preproc_path=preproc_path,
        model_path=model_path,
        eval_data=eval_data,
        results_path=results_path,
    )
