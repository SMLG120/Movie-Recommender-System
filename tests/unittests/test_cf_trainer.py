import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.cf_trainer import CFTrainer


# Fixtures
@pytest.fixture
def dummy_config(tmp_path):
    """Simple config for testing CFTrainer."""
    return {
        "ratings_csv": str(tmp_path / "ratings.csv"),
        "watch_csv": str(tmp_path / "watch.csv"),
        "movies_csv": str(tmp_path / "movies.csv"),
        "out_dir": str(tmp_path / "out"),
        "svd_factors": 2,
        "als_factors": 2,
    }


@pytest.fixture
def fake_reader(tmp_path, dummy_config):
    """Creates fake CSVs and returns pd.read_csv wrapper."""
    # Ratings for explicit CF
    ratings = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "movie_id": ["m1", "m2"],
        "rating": [4.0, 5.0]
    })
    ratings.to_csv(dummy_config["ratings_csv"], index=False)

    # Watch data for implicit CF
    watch = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "movie_id": ["m1", "m2"],
        "interaction_count": [10, 5],
        "max_minute_reached": [90, 50]
    })
    watch.to_csv(dummy_config["watch_csv"], index=False)

    # Movies data
    movies = pd.DataFrame({"id": ["m1", "m2"], "runtime": [100, 120]})
    movies.to_csv(dummy_config["movies_csv"], index=False)

    return pd.read_csv


@pytest.fixture
def mock_writer():
    """Mock writer for CSV/JSON saves."""
    writer = MagicMock()
    writer.side_effect = lambda df, path=None: df  # no disk writes
    return writer


@pytest.fixture
def mock_json_writer():
    """Mock JSON saver."""
    writer = MagicMock()
    writer.side_effect = lambda obj, path=None: obj
    return writer


@pytest.fixture
def mock_svd_cls():
    """Fake SVD model that mimics Surprise SVD API."""
    class MockSVD:
        def __init__(self, *a, **kw):
            self.pu = np.random.rand(2, 2)
            self.qi = np.random.rand(2, 2)
        def fit(self, trainset): return self
    return MockSVD


@pytest.fixture
def mock_als_cls():
    """Fake ALS model that mimics implicit ALS API."""
    class MockALS:
        def __init__(self, *a, **kw):
            self.factors = 2
            self.user_factors = np.random.rand(2, 2)
            self.item_factors = np.random.rand(2, 2)
        def fit(self, mat): return self
    return MockALS


def test_build_confidence_returns_expected_cols(dummy_config):
    df = pd.DataFrame({
        "user_id": ["u1"], "movie_id": ["m1"],
        "interaction_count": [10],
        "max_minute_reached": [90],
        "movie_duration": [100]
    })
    trainer = CFTrainer(dummy_config)
    conf = trainer._build_confidence(df)
    assert set(conf.columns) == {"user_id", "movie_id", "confidence"}
    assert conf["confidence"].iloc[0] > 1


def test_train_explicit_with_mock_model(fake_reader, mock_writer, mock_json_writer, mock_svd_cls, dummy_config):
    trainer = CFTrainer(
        dummy_config,
        reader=fake_reader,
        writer=mock_writer,
        json_writer=mock_json_writer,
        svd_cls=mock_svd_cls
    )
    trainer.train_explicit()
    # Check CSV and JSON writes were called
    assert mock_writer.call_count >= 2
    assert mock_json_writer.call_count == 1


def test_train_implicit_with_mock_model(fake_reader, mock_writer, mock_json_writer, mock_als_cls, dummy_config):
    trainer = CFTrainer(
        dummy_config,
        reader=fake_reader,
        writer=mock_writer,
        json_writer=mock_json_writer,
        als_cls=mock_als_cls
    )
    trainer.train_implicit()
    assert mock_writer.call_count >= 2
    assert mock_json_writer.call_count == 1


def test_run_calls_both_methods(monkeypatch, dummy_config):
    """Ensure run() executes both training phases."""
    trainer = CFTrainer(dummy_config)
    called = {"explicit": False, "implicit": False}

    trainer.train_explicit = lambda: called.__setitem__("explicit", True)
    trainer.train_implicit = lambda: called.__setitem__("implicit", True)
    trainer.run()
    assert all(called.values())
