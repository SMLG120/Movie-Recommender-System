from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Import offline_eval module
SELF = Path(__file__).parent
MODULE_PATH = SELF / "offline_eval.py"

import importlib.util
spec = importlib.util.spec_from_file_location("offline_eval", str(MODULE_PATH))
offline_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline_eval)

# Create more comprehensive test data
def create_test_data():
    # Generate timestamps for temporal testing
    base_time = datetime(2023, 1, 1)
    timestamps = [(base_time + timedelta(days=i)).timestamp() for i in range(20)]
    
    return pd.DataFrame({
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

SAMPLE_DATA = create_test_data()

def test_data_splitting():
    """Test different data splitting strategies"""
    # Test leave-one-out splitting
    train_df, test_df = offline_eval.train_test_split_leave_one(SAMPLE_DATA)
    assert len(test_df) < len(SAMPLE_DATA)  # Should have fewer test samples
    assert not set(test_df.index).intersection(train_df.index)  # No overlap
    
    # Test temporal splitting
    train_df, test_df = offline_eval.train_test_split_temporal(SAMPLE_DATA)
    assert max(train_df['timestamp']) <= min(test_df['timestamp'])  # Time ordering preserved

def test_cold_start():
    """Test handling of cold-start scenarios"""
    # Remove all u1's data from training
    train_data = SAMPLE_DATA[SAMPLE_DATA['user_id'] != 'u1']
    test_data = SAMPLE_DATA[SAMPLE_DATA['user_id'] == 'u1']
    
    # Test popular recommender with cold start
    pop_list = offline_eval.popular_recommender(train_data, top_k=3)
    assert len(pop_list) == 3  # Should still return k items
    
    # Test item-item with cold start
    model = offline_eval.build_item_item_model(train_data)
    recs = offline_eval.score_item_item(model, 'u1', train_data['movie_id'].unique(), top_k=3)
    assert len(recs) == 3  # Should handle unknown users gracefully

def test_temporal_bias():
    """Test for temporal bias in recommendations"""
    early_data = SAMPLE_DATA[SAMPLE_DATA['timestamp'] <= SAMPLE_DATA['timestamp'].median()]
    late_data = SAMPLE_DATA[SAMPLE_DATA['timestamp'] > SAMPLE_DATA['timestamp'].median()]
    
    # Compare metrics across time periods
    early_metrics = offline_eval.evaluate(early_data, late_data, top_k=3, neg_samples=2)
    late_metrics = offline_eval.evaluate(late_data, early_data, top_k=3, neg_samples=2)
    
    # Metrics should be different when training on different time periods
    assert early_metrics != late_metrics

def test_popularity_bias():
    """Test for popularity bias in recommendations"""
    train_df, test_df = offline_eval.train_test_split_leave_one(SAMPLE_DATA)
    
    # Generate predictions
    predictions = {}
    for uid in test_df['user_id'].unique():
        predictions[uid] = offline_eval.score_item_item(
            offline_eval.build_item_item_model(train_df),
            uid,
            train_df['movie_id'].unique(),
            top_k=3
        )
    
    # Analyze popularity bias
    metrics = offline_eval.analyze_predictions(train_df, test_df, predictions)
    assert 'item_coverage' in metrics
    assert 'avg_popularity' in metrics

def test_user_segments():
    """Test evaluation across different user segments"""
    train_df, test_df = offline_eval.train_test_split_leave_one(SAMPLE_DATA)
    segments = offline_eval.create_user_segments(train_df)
    
    # Test segment-specific evaluation
    segment_results = offline_eval.evaluate_user_segments(train_df, test_df, segments, top_k=3)
    assert 'new_users' in segment_results
    assert 'regular_users' in segment_results

def test_metric_properties():
    """Test mathematical properties of metrics"""
    recommended = ['m1', 'm2', 'm3']
    ground_truth = {'m2', 'm3'}
    k = 3
    
    # Test metric ranges
    metrics = {
        'precision': offline_eval.precision_at_k(recommended, ground_truth, k),
        'recall': offline_eval.recall_at_k(recommended, ground_truth, k),
        'ndcg': offline_eval.ndcg_at_k(recommended, ground_truth, k),
        'hit': offline_eval.hit_rate(recommended, ground_truth, k)
    }
    
    for metric_name, value in metrics.items():
        assert 0 <= value <= 1, f"{metric_name} should be between 0 and 1"

def run_all_tests():
    """Run all offline evaluation tests"""
    tests = [
        test_data_splitting,
        test_cold_start,
        test_temporal_bias,
        test_popularity_bias,
        test_user_segments,
        test_metric_properties
    ]
    
    results = []
    for test in tests:
        try:
            test()
            results.append(f"✓ {test.__name__}")
        except AssertionError as e:
            results.append(f"✗ {test.__name__}: {str(e)}")
        except Exception as e:
            results.append(f"✗ {test.__name__} (error): {str(e)}")
    
    return results

if __name__ == '__main__':
    print("\nRunning offline evaluation tests...")
    print("=" * 50)
    for result in run_all_tests():
        print(result)
    print("=" * 50)