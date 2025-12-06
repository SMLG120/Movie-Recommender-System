import sys
import unittest
import os
from unittest import SkipTest


# Add project root to path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import evaluation.Offline.offline_eval as offline_eval


@unittest.skipUnless(
    os.path.exists("src/models/v1/preprocessor.joblib")
    and os.path.exists("src/models/v1/xgb_model.joblib")
    and os.path.exists("data/test_data/offline_eval_data.parquet"),
    "Offline evaluation artifacts not present",
)


class TestOfflineEvaluation(unittest.TestCase):
    def setUp(self):
        # Correct paths relative to project root
        self.preproc_path = "src/models/v1/preprocessor.joblib"
        self.model_path = "src/models/v1/xgb_model.joblib"
        self.eval_data = "data/test_data/offline_eval_data.parquet"
        self.results_path = "evaluation/Offline/evaluation_results.json"

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
            results_path=self.results_path,
        )

        self.assertIn("regression_metrics", results)
        self.assertIn("classification_metrics", results)
        self.assertIn("inference_time", results)
        self.assertEqual(len(preds), len(y_test))

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
