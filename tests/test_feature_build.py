import pytest
import pandas as pd
import numpy as np
from src.feature_builder import FeatureBuilder


# Fixtures (synthetic data)
@pytest.fixture
def mock_train_data(tmp_path):
    """Creates minimal CSVs for movies, users, and ratings."""
    movies = pd.DataFrame({
        "id": ["m1", "m2"],
        "title": ["Movie 1", "Movie 2"],
        "release_date": ["2020-01-01", "2018-05-10"],
        "runtime": [120, 90],
        "popularity": [10, 20],
        "vote_average": [8.0, 7.0],
        "vote_count": [100, 50],
        "genres": ["Action,Comedy", "Drama"],
        "production_countries": ["US,CA", "FR"],
        "spoken_languages": ["English,French", "French"],
        "original_language": ["en", "fr"],
    })
    users = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "age": [25, 35],
        "occupation": ["student", "engineer"],
        "gender": ["M", "F"]
    })
    ratings = pd.DataFrame({
        "user_id": ["u1", "u2"],
        "movie_id": ["m1", "m2"],
        "rating": [4.5, 3.0]
    })
    # embeddings
    user_exp = pd.DataFrame({"user_id": ["u1", "u2"], "f1": [0.1, 0.2], "f2": [0.3, 0.4]})
    movie_exp = pd.DataFrame({"movie_id": ["m1", "m2"], "f1": [0.9, 0.8], "f2": [0.7, 0.6]})

    paths = {}
    for name, df in {
        "movies": movies,
        "users": users,
        "ratings": ratings,
        "user_exp": user_exp,
        "movie_exp": movie_exp,
    }.items():
        p = tmp_path / f"{name}.csv"
        df.to_csv(p, index=False)
        paths[name] = str(p)

    return paths


def test_build_train_mode(mock_train_data):
    """Test that train mode merges correctly and outputs features."""
    fb = FeatureBuilder(
        movies_file=mock_train_data["movies"],
        ratings_file=mock_train_data["ratings"],
        users_file=mock_train_data["users"],
        user_explicit_factors=mock_train_data["user_exp"],
        movie_explicit_factors=mock_train_data["movie_exp"],
        mode="train"
    )

    df = fb.build()
    # core structure checks
    assert not df.empty
    assert "rating" in df.columns
    assert "genre_Action" in df.columns
    assert "exp_user_f1" in df.columns
    assert "exp_movie_f1" in df.columns
    assert set(df["user_id"]) == {"u1", "u2"}

def test_coercion_and_fill_missing():
    """Test numeric coercion and missing handling."""
    raw = pd.DataFrame({
        "user_id": ["u1"],
        "movie_id": ["m1"],
        "age": [0],
        "runtime": [np.nan],
        "popularity": [np.nan],
        "vote_average": [np.nan],
        "vote_count": [np.nan],
        "occupation": [None],
        "gender": [None],
        "original_language": [None],
    })
    fb = FeatureBuilder(mode="inference")
    df = fb._fill_missing(raw.copy())
    assert not df["age"].isna().any()
    assert (df["runtime"] == 0).any()
    assert (df["occupation"] == "unknown").any()
    assert set(df["gender"].unique()) <= {"unknown", "U"}

# def test_inference_mode_accepts_override():
#     """Ensure inference mode requires df_override and processes it."""
#     fb = FeatureBuilder(mode="inference")
#     data = pd.DataFrame({"user_id": ["u1"], "movie_id": ["m1"], "age": [30]})
#     df = fb.build(df_override=data)
#     assert "user_id" in df.columns
#     assert "movie_id" in df.columns
#     assert not df.empty

# def test_inference_mode_requires_override():
#     fb = FeatureBuilder(mode="inference")
#     with pytest.raises(ValueError, match="df_override"):
#         fb.build()

def test_merge_embeddings_handles_missing(mock_train_data):
    """Should not crash if some embeddings missing."""
    fb = FeatureBuilder(
        movies_file=mock_train_data["movies"],
        ratings_file=mock_train_data["ratings"],
        users_file=mock_train_data["users"],
        mode="train"
    )
    df = fb.build()
    assert "user_id" in df.columns
    assert not df.empty
