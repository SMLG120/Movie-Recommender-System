import os
import pytest
from datetime import datetime
from provenance import (
    register_model,
    record_prediction,
    trace_prediction,
    provenance_collection,
    predictions_collection
)

@pytest.fixture(autouse=True)
def clear_db():
    """Clear MongoDB before each test."""
    provenance_collection.delete_many({})
    predictions_collection.delete_many({})

def test_register_model():
    metadata = {
        "git_commit": "abc123",
        "artifact_path": "/models/model.pkl",
        "metrics_json": {"acc": 0.95},
        "training_missing_values_json": {"colA": 2}
    }

    model_version = register_model(metadata)
    assert isinstance(model_version, str)

    stored = provenance_collection.find_one({"model_version": model_version})
    assert stored is not None
    assert stored["git_commit"] == "abc123"

def test_record_prediction():
    model_version = register_model({
        "git_commit": "abc123",
        "artifact_path": "/m.pkl"
    })

    event = {
        "userid": "test-user",
        "model_version": model_version,
        "prediction": 0.87,
        "input_data": [
            {"x": 1, "y": None},
            {"x": 3, "y": 5}
        ]
    }

    record_prediction(event)
    stored = predictions_collection.find_one({"userid": "test-user"})
    assert stored is not None
    assert "input_hash" in stored
    assert stored["input_row_count"] == 2
    assert stored["input_missing_count"] == 1

def test_trace_prediction():
    model_version = register_model({
        "git_commit": "ff12",
        "artifact_path": "/model.pth"
    })

    event = {
        "userid": "sam",
        "model_version": model_version,
        "prediction": 1.0,
        "input_data": [{"x": 1}]
    }

    record_prediction(event)
    saved_event = predictions_collection.find_one({"userid": "sam"})
    result = trace_prediction(saved_event["request_id"])

    assert result["event"]["userid"] == "sam"
    assert result["provenance"]["git_commit"] == "ff12"
