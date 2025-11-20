import os
import sys
import pytest
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from provenance import (
    register_model,
    record_prediction,
    trace_prediction,
    get_model_by_version,
    get_predictions_by_model,
    get_predictions_by_user,
    _hash_input,
    _compute_input_stats,
    _now
)


class TestHelperFunctions:
    """Test helper functions."""

    def test_now_returns_iso_format(self):
        """Test _now() returns ISO-8601 format."""
        now_str = _now()
        assert now_str.endswith("Z")
        assert "T" in now_str
        # Verify it can be parsed back
        datetime.fromisoformat(now_str.replace("Z", "+00:00"))

    def test_hash_input_consistent(self):
        """Test _hash_input produces consistent hash."""
        data = {"user_id": 123, "age": 25}
        hash1 = _hash_input(data)
        hash2 = _hash_input(data)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

    def test_hash_input_different_for_different_data(self):
        """Test _hash_input produces different hash for different data."""
        data1 = {"user_id": 123}
        data2 = {"user_id": 456}
        assert _hash_input(data1) != _hash_input(data2)

    def test_compute_input_stats_dict_list(self):
        """Test _compute_input_stats with list of dicts."""
        data = [
            {"age": 25, "income": None},
            {"age": 30, "income": 50000},
            {"age": None, "income": 60000}
        ]
        stats = _compute_input_stats(data)
        assert stats["input_row_count"] == 3
        assert stats["input_missing_count"] == 2

    def test_compute_input_stats_empty(self):
        """Test _compute_input_stats with empty data."""
        stats = _compute_input_stats({})
        assert stats["input_row_count"] is None
        assert stats["input_missing_count"] is None


class TestRegisterModel:
    """Test register_model function."""

    def test_register_model_requires_git_commit(self):
        """Test register_model raises error without git_commit."""
        with pytest.raises(ValueError, match="git_commit"):
            register_model({"artifact_path": "/models/model.pkl"})

    def test_register_model_requires_artifact_path(self):
        """Test register_model raises error without artifact_path."""
        with pytest.raises(ValueError, match="artifact_path"):
            register_model({"git_commit": "abc123"})

    def test_register_model_generates_version(self):
        """Test register_model generates model_version if not provided."""
        model_version = register_model({
            "git_commit": "abc123",
            "artifact_path": "/models/model.pkl"
        })
        assert model_version.startswith("v")
        assert len(model_version) > 10

    def test_register_model_uses_provided_version(self):
        """Test register_model uses provided model_version."""
        provided_version = "v1.0.0"
        model_version = register_model({
            "git_commit": "abc123",
            "artifact_path": "/models/model.pkl",
            "model_version": provided_version
        })
        assert model_version == provided_version

    def test_register_model_converts_metrics_to_json(self):
        """Test register_model converts dict metrics to JSON string."""
        model_version = register_model({
            "git_commit": "abc123",
            "artifact_path": "/models/model.pkl",
            "metrics_json": {"accuracy": 0.91, "precision": 0.85}
        })
        # Verify it was stored (no error)
        assert model_version is not None


class TestRecordPrediction:
    """Test record_prediction function."""

    @pytest.fixture
    def model_version(self):
        """Create a model for testing."""
        return register_model({
            "git_commit": "abc123",
            "artifact_path": "/models/model.pkl"
        })

    def test_record_prediction_requires_userid(self, model_version):
        """Test record_prediction requires userid."""
        with pytest.raises(ValueError, match="userid"):
            record_prediction({
                "model_version": model_version,
                "prediction": 1,
                "input_data": {}
            })

    def test_record_prediction_requires_model_version(self):
        """Test record_prediction requires model_version."""
        with pytest.raises(ValueError, match="model_version"):
            record_prediction({
                "userid": "user1",
                "prediction": 1,
                "input_data": {}
            })

    def test_record_prediction_requires_prediction(self, model_version):
        """Test record_prediction requires prediction."""
        with pytest.raises(ValueError, match="prediction"):
            record_prediction({
                "userid": "user1",
                "model_version": model_version,
                "input_data": {}
            })

    def test_record_prediction_requires_input_data(self, model_version):
        """Test record_prediction requires input_data."""
        with pytest.raises(ValueError, match="input_data"):
            record_prediction({
                "userid": "user1",
                "model_version": model_version,
                "prediction": 1
            })

    def test_record_prediction_generates_request_id(self, model_version):
        """Test record_prediction generates request_id if not provided."""
        request_id = record_prediction({
            "userid": "user1",
            "model_version": model_version,
            "prediction": 1,
            "input_data": {"age": 25}
        })
        assert request_id is not None
        assert len(request_id) > 10

    def test_record_prediction_uses_provided_request_id(self, model_version):
        """Test record_prediction uses provided request_id."""
        provided_id = "custom-request-123"
        request_id = record_prediction({
            "userid": "user1",
            "model_version": model_version,
            "prediction": 1,
            "input_data": {"age": 25},
            "request_id": provided_id
        })
        assert request_id == provided_id


class TestTracePrediction:
    """Test trace_prediction function."""

    @pytest.fixture
    def model_version(self):
        """Create a model for testing."""
        return register_model({
            "git_commit": "abc123",
            "artifact_path": "/models/model.pkl"
        })

    def test_trace_prediction_not_found(self):
        """Test trace_prediction returns None for unknown request_id."""
        result = trace_prediction("non-existent-request-id")
        assert result is None

    def test_trace_prediction_returns_event_and_provenance(self, model_version):
        """Test trace_prediction returns both event and provenance."""
        request_id = record_prediction({
            "userid": "user1",
            "model_version": model_version,
            "prediction": 1,
            "input_data": {"age": 25}
        })
        
        result = trace_prediction(request_id)
        assert result is not None
        assert "event" in result
        assert "provenance" in result
        assert result["event"]["request_id"] == request_id
        assert result["provenance"]["model_version"] == model_version


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
