import os
import sys
import json
import joblib
import pandas as pd
# Add project root to path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from src.trainer import Trainer


def _make_small_df():
    """Helper: small fake dataset for testing Trainer."""
    return pd.DataFrame({
        "user_id": ["u1", "u2", "u3", "u4"],
        "movie_id": ["m1", "m2", "m3", "m4"],
        "age_bin": ["26-35", "19-25", "36-50", "50+"],
        "occupation": ["student", "engineer", "doctor", "artist"],
        "gender": ["M", "F", "M", "F"],
        "original_language": ["en", "fr", "en", "fr"],
        "feat1": [1.1, 2.2, 3.3, 4.4],
        "feat2": [4.4, 3.3, 2.2, 1.1],
        "rating": [5, 4, 3, 2],
    })


def test_load_and_prepare_features(tmp_path):
    df = _make_small_df()
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)

    trainer = Trainer(data_file=str(csv))
    trainer.load_data()
    X, y, preprocessor, cat, num = trainer.prepare_features()

    assert not X.empty
    assert len(y) == len(df)
    assert set(cat) == {"age_bin", "occupation", "gender", "original_language"}
    assert all(isinstance(c, str) for c in num)


def test_tune_lightweight(tmp_path):
    df = _make_small_df()
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)
    tune_file = tmp_path / "tune.json"

    trainer = Trainer(data_file=str(csv))
    best = trainer.tune(
        tuning_file=str(tune_file),
        tune_df=df,
        lightweight=True,
        n_iter=2,
        cv=2,
    )
    assert isinstance(best, dict)
    assert "model__n_estimators" in best

    log = json.load(open(tune_file))
    assert isinstance(log, list)
    assert "best_params" in log[-1]


def test_train_and_metrics(tmp_path):
    df = _make_small_df()
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)
    metrics_path = tmp_path / "metrics.json"

    trainer = Trainer(data_file=str(csv), metrics_out=str(metrics_path))
    results = trainer.train()
    assert "metrics" in results
    assert metrics_path.exists()

    metrics = json.load(open(metrics_path))
    assert "rmse" in metrics["metrics"]
    assert "mae" in metrics["metrics"]
    assert "r2" in metrics["metrics"]


def test_save_pipeline(tmp_path):
    df = _make_small_df()
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)
    model_dir = tmp_path / "models"

    trainer = Trainer(data_file=str(csv))
    trainer.train()
    trainer.save(output_dir=str(model_dir))

    assert (model_dir / "preprocessor.joblib").exists()
    assert (model_dir / "xgb_model.joblib").exists()

    pre = joblib.load(model_dir / "preprocessor.joblib")
    assert pre is not None
