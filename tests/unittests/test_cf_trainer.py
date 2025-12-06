import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.cf_trainer import CFTrainer

import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


# Fixtures
@pytest.fixture
def dummy_config(tmp_path):
    """Simple config for testing CFTrainer."""
    return {
        "ratings_csv": str(tmp_path / "ratings.csv"),
        "watch_csv": str(tmp_path / "watch.csv"),
        "movies_csv": str(tmp_path / "movies.csv"),
        "out_dir": str(tmp_path / "out"),
        "mean_embed_out": str(tmp_path / "mean_embeddings.joblib"),  # ✅ REQUIRED
        "svd_factors": 2,
        "als_factors": 2,
    }


@pytest.fixture
def fake_reader(tmp_path, dummy_config):
    """Creates fake CSVs and returns pd.read_csv wrapper."""
    ratings = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "movie_id": ["m1", "m2"],
        "rating": [4.0, 5.0]
    })
    ratings.to_csv(dummy_config["ratings_csv"], index=False)

    watch = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "movie_id": ["m1", "m2"],
        "interaction_count": [10, 5],
        "max_minute_reached": [90, 50]
    })
    watch.to_csv(dummy_config["watch_csv"], index=False)

    movies = pd.DataFrame({"id": ["m1", "m2"], "runtime": [100, 120]})
    movies.to_csv(dummy_config["movies_csv"], index=False)

    return pd.read_csv


@pytest.fixture
def mock_writer():
    writer = MagicMock()
    writer.side_effect = lambda df, path=None: df
    return writer


@pytest.fixture
def mock_json_writer():
    writer = MagicMock()
    writer.side_effect = lambda obj, path=None: obj
    return writer


@pytest.fixture
def mock_svd_cls():
    class MockSVD:
        def __init__(self, *a, **kw):
            self.pu = np.random.rand(2, 2)
            self.qi = np.random.rand(2, 2)
        def fit(self, trainset): return self
    return MockSVD


@pytest.fixture
def mock_als_cls():
    class MockALS:
        def __init__(self, *a, **kw):
            self.factors = 2
            self.user_factors = np.random.rand(2, 2)
            self.item_factors = np.random.rand(2, 2)
        def fit(self, mat): return self
    return MockALS


def test_build_confidence_returns_expected_cols(dummy_config):
    df = pd.DataFrame({
        "user_id": ["u1"],
        "movie_id": ["m1"],
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
        svd_cls=mock_svd_cls,
    )
    trainer.train_explicit()
    assert mock_writer.call_count >= 2
    assert mock_json_writer.call_count == 1


def test_train_implicit_with_mock_model(fake_reader, mock_writer, mock_json_writer, mock_als_cls, dummy_config):
    trainer = CFTrainer(
        dummy_config,
        reader=fake_reader,
        writer=mock_writer,
        json_writer=mock_json_writer,
        als_cls=mock_als_cls,
    )
    trainer.train_implicit()
    assert mock_writer.call_count >= 2
    assert mock_json_writer.call_count == 1


def test_run_calls_both_methods(monkeypatch, dummy_config):
    trainer = CFTrainer(dummy_config)
    called = {"explicit": False, "implicit": False}

    trainer.train_explicit = lambda: called.__setitem__("explicit", True)
    trainer.train_implicit = lambda: called.__setitem__("implicit", True)

    # ✅ bypass filesystem dependency
    monkeypatch.setattr(trainer, "_compute_mean_embeddings", lambda *a, **k: None)

    trainer.run()
    assert all(called.values())


def test_compute_mean_embeddings_creates_file(tmp_path, dummy_config):
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    pd.DataFrame({"user_id": ["u1", "u2"], "exp_f1": [0.1, 0.2]}).to_csv(out_dir / "user_factors_explicit.csv", index=False)
    pd.DataFrame({"movie_id": ["m1", "m2"], "exp_f1": [0.3, 0.4]}).to_csv(out_dir / "movie_factors_explicit.csv", index=False)
    pd.DataFrame({"user_id": ["u1", "u2"], "imp_f1": [0.5, 0.6]}).to_csv(out_dir / "user_factors_implicit.csv", index=False)
    pd.DataFrame({"movie_id": ["m1", "m2"], "imp_f1": [0.7, 0.8]}).to_csv(out_dir / "movie_factors_implicit.csv", index=False)

    trainer = CFTrainer({**dummy_config, "out_dir": str(out_dir)})
    output_path = tmp_path / "mean_embeddings.joblib"

    mean_embeds = trainer._compute_mean_embeddings(output_path)

    assert output_path.exists()
    assert isinstance(mean_embeds, dict)
    assert all(k in mean_embeds for k in ["exp_user", "imp_user", "exp_movie", "imp_movie"])


def test_compute_mean_embeddings_handles_missing_files(tmp_path, dummy_config, caplog):
    trainer = CFTrainer({**dummy_config, "out_dir": str(tmp_path)})
    result = trainer._compute_mean_embeddings(output_path=tmp_path / "fake.joblib")
    assert result is None
    assert "Failed to compute mean embeddings" in caplog.text


def test_build_confidence_handles_invalid_values(dummy_config):
    trainer = CFTrainer(dummy_config)
    df = pd.DataFrame({
        "user_id": ["u1"],
        "movie_id": ["m1"],
        "interaction_count": ["bad"],
        "max_minute_reached": [np.nan],
        "movie_duration": [0],
    })
    conf = trainer._build_confidence(df)
    assert "confidence" in conf.columns
    assert pd.api.types.is_numeric_dtype(conf["confidence"])
    assert len(conf) == 1
