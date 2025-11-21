import pandas as pd
import joblib
import json
import numpy as np
import tempfile
import os
import sys
from unittest.mock import MagicMock, patch
# Add project root to path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from src.inference import RecommenderEngine


# def _create_fake_models(tmp_path):
#     """Create minimal model_dir with mock preprocessor + model."""
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.pipeline import Pipeline
#     from sklearn.dummy import DummyRegressor

#     preprocessor = StandardScaler()
#     model = DummyRegressor(strategy="mean")
#     model.fit([[0], [1], [2]], [3, 4, 5])  # fake fit to allow predict()

#     joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
#     joblib.dump(model, tmp_path / "xgb_model.joblib")

def _create_fake_models(tmp_path, n_features=5):
    from sklearn.preprocessing import StandardScaler
    from sklearn.dummy import DummyRegressor
    import numpy as np

    X_fake = np.random.rand(10, n_features)
    y_fake = np.random.rand(10)

    preprocessor = StandardScaler().fit(X_fake)
    model = DummyRegressor(strategy="mean").fit(X_fake, y_fake)

    joblib.dump(preprocessor, tmp_path / "preprocessor.joblib")
    joblib.dump(model, tmp_path / "xgb_model.joblib")


def _create_movies_csv(tmp_path):
    movies = pd.DataFrame({
        "id": ["m1", "m2", "m3"],
        "title": ["A", "B", "C"],
        "original_language": ["en", "fr", "en"],
        "release_date": ["2020-01-01", "2019-05-01", "2018-07-07"],
        "runtime": [120, 90, 100],
        "popularity": [10, 20, 15],
        "vote_average": [7.5, 6.8, 8.0],
        "vote_count": [100, 50, 75],
        "genres": ["Action,Drama", "Comedy", "Drama"],
        "spoken_languages": ["en", "fr", "en"],
        "production_countries": ["US", "FR", "US"]
    })
    path = tmp_path / "movies.csv"
    movies.to_csv(path, index=False)
    return path


@patch("src.inference.requests.get")
def test_recommender_pipeline(mock_get, tmp_path):
    """Integration test for RecommenderEngine.recommend() with mocks."""

    # mock API response
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "user_id": "u1",
        "age": 30,
        "occupation": "student",
        "gender": "M"
    }

    # fake model and data
    _create_fake_models(tmp_path, n_features=5)
    movies_csv = _create_movies_csv(tmp_path)

    # patch FeatureBuilder.build to skip heavy feature ops 
    with patch("src.inference.FeatureBuilder.build") as mock_build:
        mock_build.return_value = pd.DataFrame(np.random.rand(3, 5))  # fake features

        engine = RecommenderEngine(
            model_dir=str(tmp_path),
            movies_file=str(movies_csv),
        )

        recs = engine.recommend("u1", top_n=2)
        predicted_ids = [m.strip() for m in recs.split(",")]

        # validations
        assert isinstance(recs, str)
        assert len(predicted_ids) <= 3
        assert all(mid in ["m1", "m2", "m3"] for mid in predicted_ids)
        mock_get.assert_called_once()
        mock_build.assert_called_once()
