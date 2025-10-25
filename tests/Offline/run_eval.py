import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tests.Offline.offline_eval import evaluate

# Create an empty class to match the self parameter expected by evaluate
class Evaluator: pass

evaluator = Evaluator()
results, y_test, preds = evaluate(
    evaluator,
    preproc_path="../../models/preprocessor.joblib",
    model_path="../../models/xgb_model_only.joblib",
    eval_data="../../data/training_data_v2.csv"
)
