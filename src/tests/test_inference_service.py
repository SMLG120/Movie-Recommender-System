import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from inference import RecommenderEngine


#fixtures----------------------------------------------------------------------

@pytest.fixture
def my_user():
    return {
        "user_id": 1,
        "age": 40,
        "occupation": "clerical/admin",
        "gender": "F"
    }

@pytest.fixture
def my_movies_dataframe():
    return pd.DataFrame({
        "id": ["Movie+A+2020", "Movie+B+2004", "Move+C+2002"],
        "title": ["Movie A", "Movie B", "Movie C"],
        "original_language": ["ja", "en", "en"],
        "release_date": ["2020-10-01", "2004-05-07", "2002-09-24"],
        "runtime": [90, 120, 150],
        "popularity": [3.638912, 1.540592, 5.614419],
        "vote_average": [5.2, 9.4, 7.8],
        "vote_count": [65, 23, 39],
        "genres": [["Action", "Thriller"], ["Comedy", "Drama"], ["Drama", "Romance"]]
    })

@pytest.fixture
def my_model():
    model = MagicMock()
    model.predict.return_value = np.array([0.7, 0.3, 0.9])
    return model


#testing--------------------------------------------------------------------
@patch("inference.joblib.load")
@patch("inference.pd.read_csv")
def test_init(mock_read_csv, mock_joblib_load, my_movies_dataframe, my_model):
    # Set mock values
    mock_joblib_load.return_value = my_model
    mock_read_csv.return_value = my_movies_dataframe

    # Initialize engine with placeholder path and filenames
    engine = RecommenderEngine(model_path="placeholder", movies_file="placeholder", mode='dev')

    # Assert that a model containing the parsed movies dataframe exists
    assert engine.model == my_model
    assert isinstance(engine.movies, pd.DataFrame)
    assert "title" in engine.movies.columns

@patch("inference.requests.get")
@patch("inference.pd.read_csv")
@patch("inference.joblib.load")
def test_get_user_info_works(mock_joblib_load, mock_read_csv, mock_requests_get, my_user, my_model, my_movies_dataframe):
    # Set mock values
    mock_joblib_load.return_value = my_model
    mock_read_csv.return_value = my_movies_dataframe  

    # Mimic successful retrieval of user info: 200 = success
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = my_user
    mock_requests_get.return_value = mock_response

    # Initialize engine with placeholder path and filenames
    engine = RecommenderEngine(model_path="placeholder", movies_file="placeholder", mode='dev')
    result = engine.get_user_info(1)

    # Assert the correct user info was retrived
    assert result["user_id"] == 1
    assert result["gender"] == "F"

@patch("inference.requests.get")
@patch("inference.pd.read_csv")
@patch("inference.joblib.load")
def test_get_user_info_fails(mock_joblib_load, mock_read_csv, mock_requests_get, my_model, my_movies_dataframe):
    # Set mock values
    mock_joblib_load.return_value = my_model
    mock_read_csv.return_value = my_movies_dataframe 

    # Mimic failure to retrive user info: 404 = not found
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_requests_get.return_value = mock_response

    # Initialize engine with placeholder path and filenames
    engine = RecommenderEngine(model_path="placeholder", movies_file="placeholder", mode='dev')
    result = engine.get_user_info(22)

    # Assert the failure to retrive an existing user
    assert result["age"] == -1
    assert result["gender"] == "U"



