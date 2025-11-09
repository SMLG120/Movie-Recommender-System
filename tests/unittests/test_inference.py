import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from inference import RecommenderEngine


# FIXTURES
@pytest.fixture
def my_user():
    return {"user_id": 1, "age": 40, "occupation": "clerical/admin", "gender": "F"}

@pytest.fixture
def my_movies_dataframe():
    return pd.DataFrame({
        "id": ["Movie+A+2020", "Movie+B+2004", "Movie+C+2002"],
        "title": ["Movie A", "Movie B", "Movie C"],
        "original_language": ["ja", "en", "en"],
        "release_date": ["2020-10-01", "2004-05-07", "2002-09-24"],
        "runtime": [90, 120, 150],
        "popularity": [3.6, 1.5, 5.6],
        "vote_average": [5.2, 9.4, 7.8],
        "vote_count": [65, 23, 39],
        "genres": [["Action", "Thriller"], ["Comedy", "Drama"], ["Drama", "Romance"]],
    })

@pytest.fixture
def mock_pipeline():
    pipe = MagicMock()
    pipe.predict.return_value = np.array([0.7, 0.3, 0.9])
    return pipe


# TESTS
@patch("inference.joblib.load")
@patch("inference.pd.read_csv")
def test_init_loads_model_and_movies(mock_read_csv, mock_joblib_load, my_movies_dataframe, mock_pipeline):
    """Tests initialization of RecommenderEngine and resource loading"""
    mock_joblib_load.side_effect = [MagicMock(), MagicMock()]  # preprocessor, model
    mock_read_csv.return_value = my_movies_dataframe

    engine = RecommenderEngine(model_dir="placeholder", movies_file="placeholder")

    assert isinstance(engine.movies, pd.DataFrame)
    assert "title" in engine.movies.columns
    assert hasattr(engine, "model")
    assert engine.base_user.startswith("http")


@patch("inference.requests.get")
@patch("inference.joblib.load")
@patch("inference.pd.read_csv")
def test_get_user_info_success(mock_read_csv, mock_joblib_load, mock_requests_get, my_user, my_movies_dataframe):
    mock_joblib_load.side_effect = [MagicMock(), MagicMock()]
    mock_read_csv.return_value = my_movies_dataframe

    mock_response = MagicMock(status_code=200, json=lambda: my_user)
    mock_requests_get.return_value = mock_response

    engine = RecommenderEngine(model_dir="placeholder", movies_file="placeholder")
    result = engine.get_user_info(1)

    assert result["user_id"] == 1
    assert result["gender"] == "F"


@patch("inference.requests.get")
@patch("inference.joblib.load")
@patch("inference.pd.read_csv")
def test_get_user_info_failure_returns_default(mock_read_csv, mock_joblib_load, mock_requests_get, my_movies_dataframe):
    mock_joblib_load.side_effect = [MagicMock(), MagicMock()]
    mock_read_csv.return_value = my_movies_dataframe

    mock_response = MagicMock(status_code=404)
    mock_requests_get.return_value = mock_response

    engine = RecommenderEngine(model_dir="placeholder", movies_file="placeholder")
    result = engine.get_user_info(22)

    assert result["age"] == -1
    assert result["gender"] == "U"


@patch("inference.pd.read_csv")
@patch("inference.joblib.load")
def test_build_inference_df_creates_user_movie_pairs(mock_joblib_load, mock_read_csv, my_movies_dataframe, my_user):
    mock_joblib_load.side_effect = [MagicMock(), MagicMock()]
    mock_read_csv.return_value = my_movies_dataframe

    engine = RecommenderEngine(model_dir="placeholder", movies_file="placeholder")
    df = engine.build_inference_df(my_user)

    assert "user_id" in df.columns
    assert "rating" in df.columns
    assert len(df) == len(my_movies_dataframe)


@patch("inference.FeatureBuilder")
@patch("inference.pd.read_csv")
@patch("inference.joblib.load")
@patch("inference.requests.get")
def test_recommend_returns_top_movies(mock_requests_get, mock_joblib_load, mock_read_csv, mock_fb, my_user, my_movies_dataframe):
    """End-to-end test for recommend() with mocks"""
    mock_joblib_load.side_effect = [MagicMock(), MagicMock()]
    mock_read_csv.return_value = my_movies_dataframe

    mock_response = MagicMock(status_code=200, json=lambda: my_user)
    mock_requests_get.return_value = mock_response

    # mock FeatureBuilder output
    mock_fb_instance = MagicMock()
    mock_fb_instance.build.return_value = pd.DataFrame(np.random.rand(3, 5))
    mock_fb.return_value = mock_fb_instance

    engine = RecommenderEngine(model_dir="placeholder", movies_file="placeholder")
    engine.model.predict = MagicMock(return_value=np.array([0.2, 0.9, 0.5]))

    result = engine.recommend(user_id=1, top_n=2)

    assert isinstance(result, str)
    assert len(result.split(", ")) == 2


# ADDITIONAL EDGE TESTS

@patch("inference.FeatureBuilder")
@patch("inference.joblib.load")
@patch("inference.pd.read_csv")
def test_recommend_handles_prediction_error(mock_read_csv, mock_joblib_load, mock_fb, my_movies_dataframe):
    """Should not crash if model prediction fails"""
    mock_joblib_load.side_effect = [MagicMock(), MagicMock()]
    mock_read_csv.return_value = my_movies_dataframe

    mock_fb_instance = MagicMock()
    mock_fb_instance.build.side_effect = Exception("Failed to build features")
    mock_fb.return_value = mock_fb_instance

    engine = RecommenderEngine(model_dir="placeholder", movies_file="placeholder")

    result = engine.recommend(user_id=999, top_n=3)
    assert isinstance(result, str)
    assert "error" in result.lower() or result == ""

@patch("inference.joblib.load")
@patch("inference.pd.read_csv")
def test_inference_mean_embedding_fallback(mock_read_csv, mock_joblib_load):
    """Covers cold start user handling with mean embeddings loaded"""
    mean_embeds = {
        "exp_user": {"f1": 0.1},
        "imp_user": {"f1": 0.2},
        "exp_movie": {"f1": 0.3},
        "imp_movie": {"f1": 0.4},
    }
    mock_joblib_load.return_value = mean_embeds
    mock_read_csv.return_value = pd.DataFrame({
        "id": ["m1"], "title": ["Movie A"], "original_language": ["en"],
        "release_date": ["2020-01-01"], "runtime": [100], "popularity": [1.0],
        "vote_average": [5.0], "vote_count": [10], "genres": [["Drama"]],
    })

    engine = RecommenderEngine(model_dir="placeholder", movies_file="placeholder")
    df = engine.build_inference_df({"user_id": 0, "age": -1, "gender": "U"})
    assert isinstance(df, pd.DataFrame)


