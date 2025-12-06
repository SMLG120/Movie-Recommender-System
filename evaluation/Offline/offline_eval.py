import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.trainer import Trainer
from src.feature_builder import FeatureBuilder
from src.provenance import record_evaluation, register_model


def regression_metrics(y_true, y_pred):
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def classification_metrics(y_true, y_pred, threshold=3):
    """Compute binary classification metrics using a threshold."""
    y_true_bin = (y_true >= threshold).astype(int)
    y_pred_bin = (y_pred >= threshold).astype(int)
    precision = precision_score(y_true_bin, y_pred_bin)
    recall = recall_score(y_true_bin, y_pred_bin)
    f1 = f1_score(y_true_bin, y_pred_bin)
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def evaluate(
    preproc_path="src/models/preprocessor.joblib",
    model_path="src/models/xgb_model.joblib",
    results_path="evaluation_results.json",
    eval_data="data/test_data/offline_eval_data.parquet",
):
    """
    Evaluate a trained model on offline data.
    Records evaluation to provenance.
    Returns: (results_dict, y_test, y_pred)
    """
    # Load preprocessor and model
    print(f"🔹 Loading preprocessor from: {preproc_path}")
    preprocessor = joblib.load(preproc_path)

    # --- Debug inspection to detect broken preprocessor ---
    print("Preprocessor type:", type(preprocessor))
    if hasattr(preprocessor, "transformers"):
        print("Transformers in preprocessor:")
        for name, transformer, cols in preprocessor.transformers:
            # print(f"  - {name}: {type(transformer)} on columns {cols}")
            # Only error if transformer is a string but NOT "passthrough" or "drop"
            if isinstance(transformer, str) and transformer not in ("passthrough", "drop"):
                raise TypeError(
                    f"Invalid preprocessor: transformer '{name}' is a string ('{transformer}'). "
                    f"Please rebuild the preprocessor using actual sklearn transformers like OneHotEncoder or StandardScaler."
                )
    else:
        print("Warning: preprocessor has no 'transformers' attribute. Check your joblib file.")


    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Combine preprocessor and model in a pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Load evaluation dataset
    data = pd.read_parquet(eval_data)
    y_test = data["rating"].to_numpy()

    expected_cols = list(preprocessor.feature_names_in_)
    for col in expected_cols:
        if col not in data.columns:
            data[col] = 0

    data = data[expected_cols]

    # Run inference
    start_time = time.time()
    preds = pipeline.predict(data)
    inference_time = (time.time() - start_time) / max(len(data), 1)

    # Compute metrics
    reg_metrics = regression_metrics(y_test, preds)
    class_metrics = classification_metrics(y_test, preds)

    # Combine results
    results = {
        "regression_metrics": reg_metrics,
        "classification_metrics": class_metrics,
        "inference_time": inference_time,
    }

    # Save results to JSON
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    # Record evaluation to provenance
    try:
        model_version = register_model({
            "artifact_path": model_path,
            "training_data_id": "data_training_v2",
            "model_type": "XGBoost",
            "metrics_json": reg_metrics
        })
        
        record_evaluation({
            "model_version": model_version,
            "eval_type": "offline",
            "preproc_path": preproc_path,
            "model_path": model_path,
            "eval_data_path": eval_data,
            "test_set_size": len(data),
            "rmse": reg_metrics["rmse"],
            "mae": reg_metrics["mae"],
            "r2": reg_metrics["r2"],
            "precision": class_metrics["precision"],
            "recall": class_metrics["recall"],
            "f1": class_metrics["f1"],
            "accuracy": class_metrics["accuracy"],
            "inference_time_ms": inference_time * 1000,
            "metrics_json": json.dumps(results)
        })
    except Exception as e:
        print(f"Warning: Could not record evaluation to provenance: {e}")

    print(f"\nOffline evaluation completed. Results saved to: {results_path}")
    # print(json.dumps(results, indent=4))

    return results, y_test, preds


if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    preproc_path = os.path.join("src", "models", "v2", "preprocessor.joblib")
    model_path = os.path.join("src", "models", "v2", "xgb_model.joblib")
    eval_data = os.path.join("data", "test_data", "offline_eval_data.parquet")
    results_path = os.path.join(project_root, "evaluation", "Offline", "evaluation_results.json")

    evaluate(
        preproc_path=preproc_path,
        model_path=model_path,
        eval_data=eval_data,
        results_path=results_path,
    )

