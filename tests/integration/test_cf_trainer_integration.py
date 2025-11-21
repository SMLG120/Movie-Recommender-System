import os
import sys
import json
import pandas as pd
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock
# Add project root to path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from src.cf_trainer import CFTrainer 


@pytest.fixture
def tmp_cfg(tmp_path):
    return {
        "ratings_csv": tmp_path / "ratings.csv",
        "watch_csv": tmp_path / "watch.csv",
        "movies_csv": tmp_path / "movies.csv",
        "out_dir": tmp_path / "embeddings"
    }


def test_train_explicit_saves_expected_outputs(tmp_cfg):
    # setup fake data
    df = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "movie_id": ["m1", "m2"],
        "rating": [4.0, 5.0]
    })
    df.to_csv(tmp_cfg["ratings_csv"], index=False)

    # mock SVD
    fake_svd = MagicMock()
    fake_svd.pu = np.array([[0.1, 0.2], [0.3, 0.4]])
    fake_svd.qi = np.array([[0.5, 0.6], [0.7, 0.8]])
    fake_svd.fit.return_value = None

    # patch init call to return our fake model
    fake_svd_cls = MagicMock(return_value=fake_svd)

    trainer = CFTrainer(tmp_cfg, svd_cls=fake_svd_cls)
    trainer.train_explicit()

    assert (tmp_cfg["out_dir"] / "user_factors_explicit.csv").exists()
    assert (tmp_cfg["out_dir"] / "movie_factors_explicit.csv").exists()
    assert (tmp_cfg["out_dir"] / "maps/explicit_maps.json").exists()

    maps = json.load(open(tmp_cfg["out_dir"] / "maps/explicit_maps.json"))
    assert "user_map" in maps and "item_map" in maps


def test_train_implicit_saves_expected_outputs(tmp_cfg):
    # setup fake inputs
    pd.DataFrame({
        "user_id": ["u1", "u2"],
        "movie_id": ["m1", "m2"],
        "interaction_count": [5, 3],
        "max_minute_reached": [10, 7]
    }).to_csv(tmp_cfg["watch_csv"], index=False)

    pd.DataFrame({
        "id": ["m1", "m2"],
        "runtime": [100, 90]
    }).to_csv(tmp_cfg["movies_csv"], index=False)

    # fake ALS model
    fake_als = MagicMock()
    fake_als.factors = 2
    fake_als.user_factors = np.array([[0.1, 0.2], [0.3, 0.4]])
    fake_als.item_factors = np.array([[0.5, 0.6], [0.7, 0.8]])
    fake_als.fit.return_value = None
    fake_als_cls = MagicMock(return_value=fake_als)

    trainer = CFTrainer(tmp_cfg, als_cls=fake_als_cls)
    trainer.train_implicit()

    assert (tmp_cfg["out_dir"] / "user_factors_implicit.csv").exists()
    assert (tmp_cfg["out_dir"] / "movie_factors_implicit.csv").exists()
    assert (tmp_cfg["out_dir"] / "maps/implicit_maps.json").exists()


def test_run_combined_invokes_both(monkeypatch, tmp_cfg):
    trainer = CFTrainer(tmp_cfg)

    called = {"exp": False, "imp": False}

    def mock_explicit(): called["exp"] = True
    def mock_implicit(): called["imp"] = True

    trainer.train_explicit = mock_explicit
    trainer.train_implicit = mock_implicit

    trainer.run(run_explicit=True, run_implicit=True)
    assert called["exp"] and called["imp"]

def test_run_produces_mean_embeddings(tmp_path, tmp_cfg):
    trainer = CFTrainer(tmp_cfg)
    trainer.run()
    mean_path = Path("src/models/mean_embeddings.joblib")
    assert mean_path.exists()
