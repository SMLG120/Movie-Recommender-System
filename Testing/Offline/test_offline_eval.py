from pathlib import Path
import pandas as pd
import numpy as np

# Import offline_eval module
SELF = Path(__file__).parent
MODULE_PATH = SELF / "offline_eval.py"

import importlib.util
spec = importlib.util.spec_from_file_location("offline_eval", str(MODULE_PATH))
offline_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(offline_eval)

# Test data setup
SAMPLE_DATA = pd.DataFrame({
    'user_id': ['u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u3'],
    'movie_id': ['m1', 'm2', 'm1', 'm3', 'm4', 'm2', 'm3'],
    'timestamp': [1, 2, 3, 4, 5, 6, 7],
    'minutes_watched': [15, 45, 30, 60, 20, 10, 50]
})

def test_complete_evaluation():
    # Use functions from offline_eval instead of reimplementing them
    train_df, test_df = offline_eval.train_test_split_leave_one(SAMPLE_DATA)
    
    # Test popular recommender
    pop_list = offline_eval.popular_recommender(train_df, top_k=2)
    assert len(pop_list) == 2
    non_empty = [m for m in pop_list if m != '']
    assert set(non_empty) == {'m1', 'm3'} or set(non_empty) == {'m3', 'm1'}
    
    # Test user segments
    segments = offline_eval.create_user_segments(train_df)
    new_users_df = train_df[segments['new_users'](train_df)]
    regular_users_df = train_df[segments['regular_users'](train_df)]
    assert set(new_users_df['user_id'].unique()) == {'u1', 'u3'}
    assert set(regular_users_df['user_id'].unique()) == {'u2'}
    
    # Test regression metrics
    y_true = np.array([30, 45, 20])
    y_pred = np.array([35, 40, 25])
    metrics = offline_eval.regression_metrics(y_true, y_pred)
    assert 'rmse' in metrics
    assert 'mae' in metrics
    assert metrics['rmse'] > 0
    assert metrics['mae'] > 0
    
    # Test ranking metrics
    recommended = ['m1', 'm2', 'm3']
    ground_truth = {'m2'}
    k = 3
    
    assert offline_eval.precision_at_k(recommended, ground_truth, k) == 1/3
    assert offline_eval.recall_at_k(recommended, ground_truth, k) == 1.0
    assert offline_eval.hit_rate(recommended, ground_truth, k) == 1
    
    ndcg = offline_eval.ndcg_at_k(recommended, ground_truth, k)
    assert 0 <= ndcg <= 1

    # Test complete evaluation pipeline
    eval_results = offline_eval.evaluate(
        train_df,
        test_df,
        top_k=3,
        neg_samples=2,
        n_jobs=1,
        batch_size=2
    )
    
    assert isinstance(eval_results, dict)
    assert 'pop' in eval_results
    assert 'item_item' in eval_results
    for model in ['pop', 'item_item']:
        assert all(metric in eval_results[model] 
                  for metric in ['hit', 'precision', 'recall', 'ndcg'])

def test_edge_cases():
    # Test empty inputs
    empty_df = pd.DataFrame(columns=SAMPLE_DATA.columns)
    train_df, test_df = offline_eval.train_test_split_leave_one(empty_df)
    assert len(train_df) == 0
    assert len(test_df) == 0
    
    # Test single user
    single_user = SAMPLE_DATA[SAMPLE_DATA['user_id'] == 'u1']
    train_df, test_df = offline_eval.train_test_split_leave_one(single_user)
    assert len(test_df) == 1
    assert len(train_df) == 1

if __name__ == '__main__':
    try:
        test_complete_evaluation()
        test_edge_cases()
        print("✓ All offline evaluation tests passed!")
    except AssertionError as e:
        print(f"✗ Test failed: {str(e)}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")