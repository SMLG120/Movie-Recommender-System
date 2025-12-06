import pytest
import pandas as pd
from unittest.mock import MagicMock
from src.trainer import Trainer
from sklearn.base import BaseEstimator, RegressorMixin

import warnings
warnings.filterwarnings("ignore")


class DummyModel(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.mean_ = float(y.mean()) if len(y) else 0.0
        return self

    def predict(self, X, **kwargs):
        n = len(X) if hasattr(X, "__len__") else 1
        return [self.mean_] * n

@pytest.fixture
def dummy_model_factory():
    return lambda **_: DummyModel()


# Fixtures
@pytest.fixture
def mock_train_df():
    """Synthetic training data with both categorical + numeric features."""
    return pd.DataFrame({
        "user_id": [1, 2, 3, 4],
        "movie_id": [10, 11, 12, 13],
        "age_bin": ["19-25", "26-35", "19-25", "36-50"],
        "occupation": ["student", "engineer", "student", "doctor"],
        "gender": ["M", "F", "M", "F"],
        "original_language": ["en", "fr", "en", "fr"],
        "runtime": [120, 100, 90, 110],
        "popularity": [5, 8, 6, 9],
        "vote_average": [7.5, 6.8, 8.0, 7.0],
        "vote_count": [100, 200, 150, 80],
        "rating": [4.0, 3.0, 5.0, 2.0],
    })


@pytest.fixture
def fake_reader(mock_train_df):
    """Return a lambda that ignores filename and returns mock DataFrame."""
    return lambda _: mock_train_df.copy()


@pytest.fixture
def fake_writer(tmp_path):
    """Intercept file writes and store them in memory."""
    written = {}

    def _writer(path, data):
        written[path] = data

    return written, _writer


# @pytest.fixture
# def dummy_model_factory():
#     """Return a fake regressor with fit/predict stubs."""
#     class DummyModel:
#         def fit(self, X, y):  # pretend to train
#             self.fitted = True
#         def predict(self, X):
#             return [y.mean() if len(y := X.index) else 0 for _ in range(len(X))]  # constant preds
#     return lambda **_: DummyModel()

# @pytest.fixture
# def dummy_model_factory():
#     """Return a scikit-learn-compliant dummy regressor."""
#     class DummyModel(BaseEstimator, RegressorMixin):
#         def fit(self, X, y):
#             self.mean_ = float(y.mean()) if len(y) else 0.0
#             return self

#         def predict(self, X, **kwargs):
#             n = len(X) if hasattr(X, "__len__") else 1
#             return [self.mean_] * n

#     return lambda **_: DummyModel()


def test_load_data_uses_injected_reader(fake_reader, mock_train_df):
    trainer = Trainer(reader=fake_reader, logger=lambda _: None)
    df = trainer.load_data()
    assert isinstance(df, pd.DataFrame)
    assert "rating" in df.columns
    pd.testing.assert_frame_equal(
    trainer.df.sort_index(axis=1),
    mock_train_df.sort_index(axis=1),
    check_dtype=False)


def test_prepare_features_returns_expected_shapes(fake_reader):
    trainer = Trainer(reader=fake_reader, logger=lambda _: None)
    trainer.load_data()
    X, y, preprocessor, cat, num = trainer.prepare_features()
    assert len(X) == len(y)
    assert set(cat) == {"age_bin", "occupation", "gender", "original_language"}
    assert "runtime" in num and "vote_average" in num

def test_train_with_dummy_model(fake_reader, fake_writer, dummy_model_factory):
    written, writer_fn = fake_writer
    logs = []
    logger = lambda m: logs.append(m)

    trainer = Trainer(
        reader=fake_reader,
        writer=writer_fn,
        logger=logger,
        model_factory=dummy_model_factory,
        metrics_out="results.json"
    )

    results = trainer.train()
    # Check returned results
    assert "metrics" in results
    assert "rmse" in results["metrics"]
    # Check it wrote results to fake writer
    assert "results.json" in written
    # Check logs include metrics
    assert any("RMSE" in m for m in logs)

def test_save_saves_model_files(tmp_path, fake_reader, dummy_model_factory):
    """Ensure save() dumps artifacts correctly."""
    trainer = Trainer(reader=fake_reader, model_factory=dummy_model_factory)
    trainer.train()
    output_dir = tmp_path / "models"
    trainer.save(output_dir=str(output_dir))

    files = list(output_dir.glob("*.joblib"))
    assert any("preprocessor" in str(f) for f in files)
    assert any("xgb_model" in str(f) for f in files)
