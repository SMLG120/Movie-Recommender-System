import importlib.util
from pathlib import Path
import pandas as pd

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

def test_evaluate_runs():
    df = offline_eval.load_data(str(SAMPLE_CSV))
    train_df, test_df = offline_eval.train_test_split_leave_one(df, min_train_interactions=1)
    res = offline_eval.evaluate(train_df, test_df, top_k=5, neg_samples=10)
    assert isinstance(res, dict)
    # expected keys for baselines
    assert 'pop' in res or 'item_item' in res