import os
import sys
import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

def evaluate(self, preproc_path, model_path, eval_data):
    """Evaluate the trained model on the provided data."""
    # Load preprocessor and model
    preprocessor = joblib.load(preproc_path)
    model = joblib.load(model_path)

    # Load evaluation data
    df = pd.read_csv(eval_data)
    if 'rating' not in df.columns:
        raise ValueError("Evaluation data must contain 'rating' column as target.")

    # Prepare features and target (exclude IDs like in training)
    X = df.drop(columns=['user_id', 'movie_id', 'rating'])
    y = df['rating']

    # Preprocess features
    X_processed = preprocessor.transform(X)

    # Make predictions
    preds = model.predict(X_processed)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    results = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

    print(f"Evaluation Results: RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    return results, y, preds

# Create an empty class to match the self parameter expected by evaluate
class Evaluator: pass

evaluator = Evaluator()
results, y_test, preds = evaluate(
    evaluator,
    preproc_path="src/models/preprocessor.joblib",
    model_path="src/models/xgb_model.joblib",
    eval_data="data/training_data_v2.csv"
)


if __name__ == "__main__":
    preproc_path="src/models/preprocessor.joblib"
    preprocessor = joblib.load(preproc_path)
    print("Preprocessor type:", type(preprocessor))
    exit