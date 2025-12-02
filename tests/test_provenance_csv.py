import os
import sys
import pytest
from datetime import datetime
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from provenance_csv import (
    register_model,
    record_prediction,
    trace_prediction,
    get_model_by_version,
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

    def test_hash_input_consistent(self):
        """Test _hash_input produces consistent hash."""
        data = {"user_id": 123, "age": 25}
        hash1 = _hash_input(data)
        hash2 = _hash_input(data)
        assert hash1 == hash2
        assert len(hash1) == 64

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


class TestRegisterModel:
    """Test register_model function."""

    def test_register_model_requires_artifact_path(self):
        """Test register_model raises error without artifact_path."""
        with pytest.raises(ValueError, match="artifact_path"):
            register_model({"training_data_id": "data_123"})

    def test_register_model_generates_version(self):
        """Test register_model generates model_version if not provided."""
        model_version = register_model({
            "artifact_path": "/models/model.pkl",
            "training_data_id": "data_123"
        })
        assert model_version.startswith("v")
        assert len(model_version) > 10


class TestRecordPrediction:
    """Test record_prediction function."""

    @pytest.fixture
    def model_version(self):
        """Create a model for testing."""
        return register_model({
            "artifact_path": "/models/model.pkl",
            "training_data_id": "data_123"
        })

    def test_record_prediction_requires_userid(self, model_version):
        """Test record_prediction requires userid."""
        with pytest.raises(ValueError, match="userid"):
            record_prediction({
                "model_version": model_version,
                "prediction": 1,
                "input_data": {}
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


class TestTracePrediction:
    """Test trace_prediction function."""

    @pytest.fixture
    def model_version(self):
        """Create a model for testing."""
        return register_model({
            "artifact_path": "/models/model.pkl",
            "training_data_id": "data_123"
        })

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
        assert "model_provenance" in result
        assert result["event"]["request_id"] == request_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
