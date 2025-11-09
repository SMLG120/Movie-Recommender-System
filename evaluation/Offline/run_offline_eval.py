# run_eval.py
import os
import sys

# Add project root so we can import offline_eval
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from evaluation.Offline import offline_eval  # Import the evaluate function

if __name__ == "__main__":
    # Project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    # Paths to models, preprocessor, and data
    preproc_path = os.path.join(project_root, "src", "models", "preprocessor.joblib")
    model_path = os.path.join(project_root, "src", "models", "xgb_model.joblib")
    eval_data = os.path.join(project_root, "data", "training_data_v2.csv")
    results_path = os.path.join(project_root, "evaluation", "Offline", "evaluation_results.json")

    # Call offline_eval's evaluate function
    results, y_test, preds = offline_eval.evaluate(
        preproc_path=preproc_path,
        model_path=model_path,
        eval_data=eval_data,
        results_path=results_path
    )

    print("\nOffline evaluation finished from run_eval.py")
    print("Results preview:", results)
