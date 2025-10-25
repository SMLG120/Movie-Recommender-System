"""
test_offline_eval.py
--------------------
Offline evaluation test suite for the recommender system.

This version fixes multiprocessing pickling issues and pandas deprecation warnings.
"""

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import unittest
import os

# Ensure the offline_eval module is importable as a regular module
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import offline_eval  # Safe import (works with multiprocessing)

# ----------------------------------------------------------------------
# Create synthetic test data
# ----------------------------------------------------------------------
def create_test_data():
    """Generate synthetic dataset for testing recommender evaluation."""
    base_time = datetime(2023, 1, 1)
    timestamps = [(base_time + timedelta(days=i)).timestamp() for i in range(20)]
    
    df = pd.DataFrame({
        'user_id': ['u1', 'u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u3', 'u4', 'u4',
                    'u5', 'u5', 'u5', 'u5', 'u6', 'u7', 'u7', 'u8', 'u8', 'u8'],
        'movie_id': ['m1', 'm2', 'm3', 'm1', 'm4', 'm5', 'm2', 'm3', 'm4', 'm5',
                     'm1', 'm2', 'm3', 'm4', 'm5', 'm1', 'm2', 'm3', 'm4', 'm5'],
        'timestamp': timestamps,
        'minutes_watched': [15, 45, 30, 60, 20, 10, 50, 40, 25, 35,
                            55, 15, 25, 35, 45, 20, 30, 40, 50, 60],
        'rating': [4, 5, 3, 5, 2, 1, 4, 3, 2, 3,
                   5, 2, 3, 4, 5, 2, 3, 4, 5, 4]
    })

    # Avoid deprecated .swapaxes warnings if used internally
    pd.DataFrame.transpose = pd.DataFrame.transpose

    return df


SAMPLE_DATA = create_test_data()


# ----------------------------------------------------------------------
# Test definitions
# ----------------------------------------------------------------------

def test_data_splitting():
    """Test different data splitting strategies."""
    train_df, test_df = offline_eval.train_test_split_leave_one(SAMPLE_DATA)
    assert len(test_df) < len(SAMPLE_DATA), "Test set should be smaller than original dataset"
    assert not set(test_df.index).intersection(train_df.index), "Train/test overlap detected"

    train_df, test_df = offline_eval.train_test_split_temporal(SAMPLE_DATA)
    assert max(train_df['timestamp']) <= min(test_df['timestamp']), "Temporal ordering not preserved"


def test_cold_start():
    """Test handling of cold-start scenarios."""
    train_data = SAMPLE_DATA[SAMPLE_DATA['user_id'] != 'u1']
    test_data = SAMPLE_DATA[SAMPLE_DATA['user_id'] == 'u1']

    pop_list = offline_eval.popular_recommender(train_data, top_k=3)
    assert len(pop_list) == 3, "Popular recommender did not return top_k items"

    model = offline_eval.build_item_item_model(train_data)
    recs = offline_eval.score_item_item(model, 'u1', train_data['movie_id'].unique(), top_k=3)
    assert len(recs) == 3, "Item-item recommender failed to return top_k items"


def test_temporal_bias():
    """Test for temporal bias in recommendations."""
    early_data = SAMPLE_DATA[SAMPLE_DATA['timestamp'] <= SAMPLE_DATA['timestamp'].median()]
    late_data = SAMPLE_DATA[SAMPLE_DATA['timestamp'] > SAMPLE_DATA['timestamp'].median()]

    # Force sequential evaluation (disable multiprocessing)
    early_metrics = offline_eval.evaluate(early_data, late_data, top_k=3, neg_samples=2, parallel=False)
    late_metrics = offline_eval.evaluate(late_data, early_data, top_k=3, neg_samples=2, parallel=False)

    assert early_metrics != late_metrics, "Temporal bias not detected — metrics identical across time periods"


def test_popularity_bias():
    """Test for popularity bias in recommendations."""
    train_df, test_df = offline_eval.train_test_split_leave_one(SAMPLE_DATA)

    predictions = {}
    model = offline_eval.build_item_item_model(train_df)
    for uid in test_df['user_id'].unique():
        predictions[uid] = offline_eval.score_item_item(model, uid, train_df['movie_id'].unique(), top_k=3)

    metrics = offline_eval.analyze_predictions(train_df, test_df, predictions)
    assert 'item_coverage' in metrics, "item_coverage metric missing"
    assert 'avg_popularity' in metrics, "avg_popularity metric missing"


def test_user_segments():
    """Test evaluation across different user segments."""
    train_df, test_df = offline_eval.train_test_split_leave_one(SAMPLE_DATA)
    segments = offline_eval.create_user_segments(train_df)

    segment_results = offline_eval.evaluate_user_segments(train_df, test_df, segments, top_k=3, parallel=False)
    assert 'new_users' in segment_results, "new_users segment missing"
    assert 'regular_users' in segment_results, "regular_users segment missing"


def test_metric_properties():
    """Test mathematical properties of metrics."""
    recommended = ['m1', 'm2', 'm3']
    ground_truth = {'m2', 'm3'}
    k = 3

    metrics = {
        'precision': offline_eval.precision_at_k(recommended, ground_truth, k),
        'recall': offline_eval.recall_at_k(recommended, ground_truth, k),
        'ndcg': offline_eval.ndcg_at_k(recommended, ground_truth, k),
        'hit': offline_eval.hit_rate(recommended, ground_truth, k)
    }

    for name, value in metrics.items():
        assert 0 <= value <= 1, f"{name} out of range: {value}"


# ----------------------------------------------------------------------
# Test runner
# ----------------------------------------------------------------------

def run_all_tests():
    tests = [
        test_data_splitting,
        test_cold_start,
        test_temporal_bias,
        test_popularity_bias,
        test_user_segments,
        test_metric_properties
    ]

    results = []
    print("\nRunning offline evaluation tests...")
    print("=" * 55)

    for test in tests:
        try:
            test()
            results.append(f"✓ {test.__name__}")
        except AssertionError as e:
            results.append(f"✗ {test.__name__}: {str(e)}")
        except Exception as e:
            results.append(f"✗ {test.__name__} (error): {str(e)}")

    print("\n".join(results))
    print("=" * 55)
    return results


class TestOfflineEvaluation(unittest.TestCase):
    def setUp(self):
        self.preproc_path = "../../models/preprocessor.joblib"
        self.model_path = "../../models/xgb_model_only.joblib"
        self.eval_data = "../../data/training_data_v2.csv"
        self.results_path = "../../evaluation_results.json"
        
    def test_evaluation(self):
        # Ensure model and data files exist
        self.assertTrue(os.path.exists(self.preproc_path), "Preprocessor file not found")
        self.assertTrue(os.path.exists(self.model_path), "Model file not found")
        self.assertTrue(os.path.exists(self.eval_data), "Evaluation data file not found")
        
        # Run evaluation
        results, y_test, preds = offline_eval.evaluate(
            self,
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
            
        # Check classification metrics
        class_metrics = results["classification_metrics"]
        for metric in ["precision", "recall", "f1", "accuracy"]:
            self.assertIn(metric, class_metrics)
            
        # Check if results were saved
        self.assertTrue(os.path.exists(self.results_path))


if __name__ == '__main__':
    run_all_tests()
    unittest.main()
