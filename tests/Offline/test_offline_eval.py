"""
test_offline_eval.py
--------------------
Offline evaluation test suite for the recommender system.

This version tests the offline_eval.evaluate function with correct paths and function calls.
"""

import sys
import unittest
import os

# Add project root to path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import offline_eval


class TestOfflineEvaluation(unittest.TestCase):
    def setUp(self):
        # Correct paths relative to project root
        self.preproc_path = "src/models/preprocessor.joblib"
        self.model_path = "src/models/xgb_model.joblib"
        self.eval_data = "data/training_data_v2.csv"
        self.results_path = "tests/Offline/evaluation_results.json"

    def test_evaluation(self):
        # Ensure model and data files exist
        self.assertTrue(os.path.exists(self.preproc_path), f"Preprocessor file not found at {self.preproc_path}")
        self.assertTrue(os.path.exists(self.model_path), f"Model file not found at {self.model_path}")
        self.assertTrue(os.path.exists(self.eval_data), f"Evaluation data file not found at {self.eval_data}")

        # Run evaluation using offline_eval.evaluate
        results, y_test, preds = offline_eval.evaluate(
            preproc_path=self.preproc_path,
            model_path=self.model_path,
            eval_data=self.eval_data,
            results_path=self.results_path
        )

        # Check if results contain all expected metrics
        self.assertIn("regression_metrics", results)
        self.assertIn("classification_metrics", results)
        self.assertIn("inference_time", results)

        # Check regression metrics
        reg_metrics = results["regression_metrics"]
        for metric in ["mse", "rmse", "mae", "r2"]:
            self.assertIn(metric, reg_metrics)
            self.assertIsInstance(reg_metrics[metric], (int, float))

        # Check classification metrics
        class_metrics = results["classification_metrics"]
        for metric in ["precision", "recall", "f1", "accuracy"]:
            self.assertIn(metric, class_metrics)
            self.assertIsInstance(class_metrics[metric], (int, float))

        # Check inference time
        self.assertIsInstance(results["inference_time"], (int, float))
        self.assertGreater(results["inference_time"], 0)

        # Check if results were saved
        self.assertTrue(os.path.exists(self.results_path))

        # Verify predictions shape matches test data
        self.assertEqual(len(preds), len(y_test))


if __name__ == '__main__':
    unittest.main()
