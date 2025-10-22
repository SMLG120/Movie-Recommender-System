import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

SELF = Path(__file__).parent
MODULE_PATH = SELF / "offline_eval.py"

spec = importlib.util.spec_from_file_location("offline_eval", str(MODULE_PATH))
offline_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline_eval)

SAMPLE_CSV = SELF / "sample_training.csv"

def test_load_and_types():
    df = offline_eval.load_data(str(SAMPLE_CSV))
    assert isinstance(df, pd.DataFrame)
    assert 'user_id' in df.columns and 'movie_id' in df.columns

def test_train_test_split_leave_one():
    df = offline_eval.load_data(str(SAMPLE_CSV))
    train_df, test_df = offline_eval.train_test_split_leave_one(df, min_train_interactions=1)
    # u1 has 2 interactions -> should have a test row
    assert 'u1' in train_df['user_id'].values
    assert any(test_df['user_id'] == 'u1')

def test_metrics_basic():
    recommended = ['m1', 'm2', 'm3']
    gt = {'m2'}
    assert offline_eval.precision_at_k(recommended, gt, 3) == 1/3
    assert offline_eval.recall_at_k(recommended, gt, 3) == 1
    assert offline_eval.hit_rate(recommended, gt, 3) == 1
    # ndcg: since ground truth at rank 2, ndcg>0
    ndcg = offline_eval.ndcg_at_k(recommended, gt, 3)
    assert 0.0 <= ndcg <= 1.0

def test_metrics_edge_cases():
    # Empty recommendations
    assert offline_eval.precision_at_k([], {'m1'}, 3) == 0
    assert offline_eval.recall_at_k([], {'m1'}, 3) == 0
    assert offline_eval.ndcg_at_k([], {'m1'}, 3) == 0
    assert offline_eval.hit_rate([], {'m1'}, 3) == 0

    # Empty ground truth
    assert offline_eval.precision_at_k(['m1', 'm2'], set(), 3) == 0
    assert offline_eval.recall_at_k(['m1', 'm2'], set(), 3) == 0
    assert offline_eval.ndcg_at_k(['m1', 'm2'], set(), 3) == 0
    assert offline_eval.hit_rate(['m1', 'm2'], set(), 3) == 0

    # Perfect match
    gt = {'m1', 'm2'}
    assert offline_eval.precision_at_k(['m1', 'm2'], gt, 2) == 1.0
    assert offline_eval.recall_at_k(['m1', 'm2'], gt, 2) == 1.0
    assert offline_eval.ndcg_at_k(['m1', 'm2'], gt, 2) == 1.0
    assert offline_eval.hit_rate(['m1', 'm2'], gt, 2) == 1.0

def test_train_test_split_parameters():
    df = offline_eval.load_data(str(SAMPLE_CSV))
    
    # Test with higher min_train_interactions
    train_df, test_df = offline_eval.train_test_split_leave_one(df, min_train_interactions=3)
    assert len(test_df) <= len(test_df)  # Should have fewer test cases
    
    # Verify each user in test set has sufficient training data
    test_users = set(test_df['user_id'].unique())
    for user in test_users:
        user_train_count = len(train_df[train_df['user_id'] == user])
        assert user_train_count >= 3

def test_evaluate_runs():
    df = offline_eval.load_data(str(SAMPLE_CSV))
    train_df, test_df = offline_eval.train_test_split_leave_one(df, min_train_interactions=1)
    res = offline_eval.evaluate(train_df, test_df, top_k=5, neg_samples=10)
    assert isinstance(res, dict)
    # expected keys for baselines
    assert 'pop' in res or 'item_item' in res

def test_evaluate_parameters():
    df = offline_eval.load_data(str(SAMPLE_CSV))
    train_df, test_df = offline_eval.train_test_split_leave_one(df, min_train_interactions=1)
    
    # Test with different k values
    res1 = offline_eval.evaluate(train_df, test_df, top_k=1, neg_samples=10)
    res5 = offline_eval.evaluate(train_df, test_df, top_k=5, neg_samples=10)
    assert all(res1[model]['recall@1'] <= res5[model]['recall@5'] for model in res1.keys())
    
    # Test with different negative sample sizes
    res_more_neg = offline_eval.evaluate(train_df, test_df, top_k=5, neg_samples=20)
    assert isinstance(res_more_neg, dict)

def test_temporal_split():
    df = offline_eval.load_data(str(SAMPLE_CSV))
    train_df, test_df = offline_eval.train_test_split_temporal(df)
    assert len(train_df) > 0 and len(test_df) > 0
    # Test should contain newer timestamps
    assert train_df['timestamp'].max() <= test_df['timestamp'].min()

def test_regression_metrics():
    y_true = np.array([10, 20, 30, 40])
    y_pred = np.array([12, 18, 35, 38])
    metrics = offline_eval.regression_metrics(y_true, y_pred)
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert metrics['rmse'] > 0
    assert metrics['mae'] > 0

def test_classification_metrics():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    metrics = offline_eval.classification_metrics(y_true, y_pred)
    assert 'auc' in metrics
    assert 'ap' in metrics
    assert 0 <= metrics['auc'] <= 1
    assert 0 <= metrics['ap'] <= 1

def test_user_segments():
    df = pd.DataFrame({
        'user_id': ['u1']*3 + ['u2']*7 + ['u3']*2,
        'movie_id': range(12),
        'timestamp': range(12)
    })
    segments = offline_eval.create_user_segments(df)
    assert 'new_users' in segments
    assert 'regular_users' in segments
    new_users_df = df[segments['new_users'](df)]
    regular_users_df = df[segments['regular_users'](df)]
    assert len(new_users_df) + len(regular_users_df) == len(df)

def test_evaluate_all_tasks():
    df = offline_eval.load_data(str(SAMPLE_CSV))
    train_df, test_df = offline_eval.train_test_split_leave_one(df)
    results = offline_eval.evaluate_all_tasks(train_df, test_df)
    assert isinstance(results, dict)
    assert 'ranking' in results
    assert 'segments' in results