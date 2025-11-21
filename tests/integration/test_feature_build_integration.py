import os
import sys
import pandas as pd
import pytest
import joblib
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
# Add project root to path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from src.feature_builder import FeatureBuilder


@pytest.fixture
def tmp_data(tmp_path):
    """Prepare minimal test CSVs for train mode."""
    data_dir = tmp_path
    # users
    pd.DataFrame({
        "user_id": ["u1"],
        "age": [30],
        "occupation": ["engineer"],
        "gender": ["M"]
    }).to_csv(data_dir / "users.csv", index=False)

    # movies
    pd.DataFrame({
        "id": ["m1"],
        "title": ["Movie X"],
        "genres": ["Action,Drama"],
        "original_language": ["en"],
        "release_date": ["2020-01-01"],
        "runtime": [120],
        "popularity": [1.2],
        "vote_average": [8.0],
        "vote_count": [50],
        "production_countries": ["US"],
        "spoken_languages": ["en"]
    }).to_csv(data_dir / "movies.csv", index=False)

    # ratings
    pd.DataFrame({
        "user_id": ["u1"],
        "movie_id": ["m1"],
        "rating": [5.0]
    }).to_csv(data_dir / "ratings.csv", index=False)

    # embeddings
    pd.DataFrame({"user_id": ["u1"], "exp_f1": [0.1]}).to_csv(data_dir / "user_factors_explicit.csv", index=False)
    pd.DataFrame({"movie_id": ["m1"], "exp_f1": [0.2]}).to_csv(data_dir / "movie_factors_explicit.csv", index=False)
    pd.DataFrame({"user_id": ["u1"], "imp_f1": [0.3]}).to_csv(data_dir / "user_factors_implicit.csv", index=False)
    pd.DataFrame({"movie_id": ["m1"], "imp_f1": [0.4]}).to_csv(data_dir / "movie_factors_implicit.csv", index=False)

    return data_dir


def test_train_mode_build_creates_expected_features(tmp_data):
    fb = FeatureBuilder(
        movies_file=tmp_data / "movies.csv",
        ratings_file=tmp_data / "ratings.csv",
        users_file=tmp_data / "users.csv",
        user_explicit_factors=tmp_data / "user_factors_explicit.csv",
        movie_explicit_factors=tmp_data / "movie_factors_explicit.csv",
        user_implicit_factors=tmp_data / "user_factors_implicit.csv",
        movie_implicit_factors=tmp_data / "movie_factors_implicit.csv",
        mode="train"
    )

    df = fb.build()

    assert "genre_Action" in df.columns
    assert "country_US" in df.columns
    assert "lang_en" in df.columns
    assert "exp_user_exp_f1" in df.columns
    assert "imp_movie_imp_f1" in df.columns
    assert not df.isnull().any().any()


@patch("src.feature_builder.joblib.load")
def test_inference_mode_uses_mean_embeddings(mock_joblib_load, tmp_path):
    mean_embeddings = {
        "exp_user": {"exp_f1": 0.1},
        "imp_user": {"imp_f1": 0.2},
        "exp_movie": {"exp_f1": 0.3},
        "imp_movie": {"imp_f1": 0.4},
    }
    mock_joblib_load.return_value = mean_embeddings

    # Minimal inference data
    df_override = pd.DataFrame({
        "user_id": ["uX"],
        "movie_id": ["mX"],
        "age": [25],
        "occupation": ["student"],
        "gender": ["F"],
        "genres": ["Action,Drama"],
        "original_language": ["en"],
        "runtime": [100],
        "popularity": [2.0],
        "vote_average": [7.5],
        "vote_count": [30],
        "release_date": ["2022-05-05"],
        "production_countries": ["US"],
        "spoken_languages": ["en"]
    })

    fb = FeatureBuilder(mode="inference")
    df = fb.build(df_override=df_override)

    assert any(c.startswith("exp_user_") for c in df.columns)
    assert any(c.startswith("imp_movie_") for c in df.columns)
    assert not df.empty


def test_missing_file_logs_warning(tmp_path):
    logs = []
    fb = FeatureBuilder(
        movies_file=tmp_path / "missing.csv",
        mode="train",
        logger=logs.append
    )
    assert any("[WARN]" in log for log in logs)
