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
from src.trainer import Trainer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ===============================
# Metric helpers
# ===============================
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


# ===============================
# Main offline evaluation function
# ===============================
def evaluate(
    preproc_path="src/models/preprocessor.joblib",
    model_path="src/models/xgb_model.joblib",
    eval_data="data/training_data_v2.csv",
    results_path="evaluation_results.json",
):
    """
    Evaluate a trained model on offline data.
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
            print(f"  - {name}: {type(transformer)} on columns {cols}")
            # Only error if transformer is a string but NOT "passthrough" or "drop"
            if isinstance(transformer, str) and transformer not in ("passthrough", "drop"):
                raise TypeError(
                    f"❌ Invalid preprocessor: transformer '{name}' is a string ('{transformer}'). "
                    f"Please rebuild the preprocessor using actual sklearn transformers like OneHotEncoder or StandardScaler."
                )
    else:
        print("⚠️ Warning: preprocessor has no 'transformers' attribute. Check your joblib file.")


    print(f"🔹 Loading model from: {model_path}")
    model = joblib.load(model_path)

    # Combine preprocessor and model in a pipeline
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

    # Load evaluation dataset
    print(f"🔹 Loading evaluation data from: {eval_data}")
    trainer = Trainer()
    trainer.df = pd.read_csv(eval_data)
    X, y, _, _, _ = trainer.prepare_features()

    # Split into test set (offline evaluation)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Evaluation dataset: {X_test.shape[0]} samples")

    # Run inference
    start_time = time.time()
    preds = pipeline.predict(X_test)
    inference_time = (time.time() - start_time) / max(len(X_test), 1)

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

    print(f"\n✅ Offline evaluation completed. Results saved to: {results_path}")
    print(json.dumps(results, indent=4))

    return results, y_test, preds


# ===============================
# Run directly for testing
# ===============================
if __name__ == "__main__":
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    preproc_path = os.path.join(project_root, "src", "models", "preprocessor.joblib")
    model_path = os.path.join(project_root, "src", "models", "xgb_model.joblib")
    eval_data = os.path.join(project_root, "data", "training_data_v2.csv")
    results_path = os.path.join(project_root, "tests", "Offline", "evaluation_results.json")

    evaluate(
        preproc_path=preproc_path,
        model_path=model_path,
        eval_data=eval_data,
        results_path=results_path,
    )
