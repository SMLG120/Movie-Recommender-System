import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from unittest.mock import patch

from cf_trainer import CFTrainer
from feature_builder import FeatureBuilder
from trainer import Trainer
from inference import RecommenderEngine


def _create_fake_data(tmp_path):
    """Creates minimal fake raw data for the full pipeline."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    # users
    pd.DataFrame({
        "user_id": ["u1", "u2"],
        "age": [25, 35],
        "occupation": ["student", "engineer"],
        "gender": ["M", "F"]
    }).to_csv(raw_dir / "users.csv", index=False)

    # movies
    pd.DataFrame({
        "id": ["m1", "m2"],
        "title": ["Movie A", "Movie B"],
        "original_language": ["en", "fr"],
        "release_date": ["2020-01-01", "2019-01-01"],
        "runtime": [100, 120],
        "popularity": [10, 20],
        "vote_average": [7.5, 8.0],
        "vote_count": [50, 70],
        "genres": ["Action,Comedy", "Drama"],
        "spoken_languages": ["en", "fr"],
        "production_countries": ["US", "FR"]
    }).to_csv(raw_dir / "movies.csv", index=False)

    # ratings
    pd.DataFrame({
        "user_id": ["u1", "u2"],
        "movie_id": ["m1", "m2"],
        "rating": [4.0, 5.0],
    }).to_csv(raw_dir / "ratings.csv", index=False)

    # watch interactions (for implicit CF)
    pd.DataFrame({
        "user_id": ["u1", "u2"],
        "movie_id": ["m1", "m2"],
        "interaction_count": [8, 5],
        "max_minute_reached": [80, 50],
    }).to_csv(raw_dir / "watch.csv", index=False)

    return raw_dir


def test_full_pipeline_cf_to_inference(tmp_path):
    """End-to-end test: CFTrainer → FeatureBuilder → Trainer → RecommenderEngine."""
    raw_dir = _create_fake_data(tmp_path)
    emb_dir = tmp_path / "embeddings"
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # ---------- 1️⃣ CF TRAINER ----------
    cfg = {
        "ratings_csv": str(raw_dir / "ratings.csv"),
        "watch_csv": str(raw_dir / "watch.csv"),
        "movies_csv": str(raw_dir / "movies.csv"),
        "out_dir": str(emb_dir),
        "svd_factors": 5,
        "als_factors": 5,
    }
    cf = CFTrainer(cfg)
    cf.run()
    cf._compute_mean_embeddings(emb_dir / "mean_embeddings.joblib")

    # Check embeddings exist
    assert (emb_dir / "user_factors_explicit.csv").exists()
    assert (emb_dir / "movie_factors_implicit.csv").exists()
    assert (emb_dir / "mean_embeddings.joblib").exists()

    # ---------- 2️⃣ FEATURE BUILDER ----------
    fb = FeatureBuilder(
        movies_file=raw_dir / "movies.csv",
        ratings_file=raw_dir / "ratings.csv",
        users_file=raw_dir / "users.csv",
        user_explicit_factors=emb_dir / "user_factors_explicit.csv",
        movie_explicit_factors=emb_dir / "movie_factors_explicit.csv",
        user_implicit_factors=emb_dir / "user_factors_implicit.csv",
        movie_implicit_factors=emb_dir / "movie_factors_implicit.csv",
        mode="train",
    )
    features = fb.build()
    train_path = tmp_path / "train_features.csv"
    features.to_csv(train_path, index=False)
    assert not features.empty

    # ---------- 3️⃣ TRAINER ----------
    trainer = Trainer(data_file=str(train_path))
    trainer.train()
    trainer.save(output_dir=str(model_dir))
    assert (model_dir / "xgb_model.joblib").exists()
    assert (model_dir / "preprocessor.joblib").exists()

    # ---------- 4️⃣ RECOMMENDER ENGINE ----------
    with patch("src.inference.requests.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json.return_value = {
            "user_id": "u1",
            "age": 25,
            "occupation": "student",
            "gender": "M",
        }

        engine = RecommenderEngine(
            model_dir=str(model_dir),
            movies_file=str(raw_dir / "movies.csv"),
        )
        recs = engine.recommend("u1", top_n=2)

    # Assertions
    assert isinstance(recs, str)
    assert any(mid in recs for mid in ["m1", "m2"])
    assert "Movie" not in recs 