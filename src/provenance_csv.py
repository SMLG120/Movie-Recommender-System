import os
import csv
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Optional, Any


# -------------------------------------------------------------------
# Paths Setup
# -------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

_PROV_PATH = os.path.join(_MODELS_DIR, "provenance.csv")
_PRED_PATH = os.path.join(_MODELS_DIR, "prediction_events.csv")

# CSV Headers
_PROV_FIELDS = [
    "model_version",
    "model_tag",
    "build_time",
    "git_commit",
    "pipeline_version",
    "python_env_hash",
    "training_data_id",
    "training_data_range_start",
    "training_data_range_end",
    "training_row_count",
    "metrics_json",
    "artifact_path"
]

_PRED_FIELDS = [
    "request_id",
    "timestamp",
    "userid",
    "model_version",
    "prediction",
    "input_hash",
    "input_row_count",
    "input_missing_count",
    "extra_json"
]


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def _now() -> str:
    """Return current time in ISO-8601 format."""
    return datetime.utcnow().isoformat() + "Z"


def _ensure_models_dir():
    """Create models directory if it doesn't exist."""
    if not os.path.exists(_MODELS_DIR):
        os.makedirs(_MODELS_DIR, exist_ok=True)
        print(f"✓ Created models directory: {_MODELS_DIR}")


def _ensure_csv(path: str, headers: list):
    """Create CSV file with headers if it doesn't exist."""
    _ensure_models_dir()
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        print(f"✓ Created CSV file: {path}")


def _hash_input(data: Any) -> str:
    """Compute SHA-256 hash of input data."""
    try:
        if hasattr(data, "to_dict"):
            data = data.to_dict()
    except Exception:
        pass

    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def _compute_input_stats(data: Any) -> Dict[str, Any]:
    """Compute row count and missing value count from input data."""
    stats = {
        "input_row_count": None,
        "input_missing_count": None
    }

    # pandas DataFrame
    if hasattr(data, "shape"):
        stats["input_row_count"] = int(data.shape[0])
        if hasattr(data, "isnull"):
            stats["input_missing_count"] = int(data.isnull().sum().sum())
        return stats

    # list of dictionaries
    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        stats["input_row_count"] = len(data)
        missing = sum(1 for row in data for v in row.values() if v is None)
        stats["input_missing_count"] = missing
        return stats

    return stats


# -------------------------------------------------------------------
# Register Model
# -------------------------------------------------------------------

def register_model(metadata: Dict) -> str:
    """
    Store model metadata in provenance.csv.
    Required fields: git_commit, artifact_path
    Returns model_version.
    """
    metadata = metadata.copy()

    # Validate required fields
    required = ["git_commit", "artifact_path"]
    missing = [r for r in required if r not in metadata]
    if missing:
        raise ValueError(f"Missing required model metadata fields: {missing}")

    # Generate model version if not provided
    model_version = metadata.get("model_version") or (
        datetime.utcnow().strftime("v%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    )
    metadata["model_version"] = model_version
    metadata.setdefault("model_tag", model_version)
    metadata.setdefault("build_time", _now())

    # Ensure CSV exists
    _ensure_csv(_PROV_PATH, _PROV_FIELDS)

    # Convert nested dicts to JSON strings
    if "metrics_json" in metadata and not isinstance(metadata["metrics_json"], str):
        metadata["metrics_json"] = json.dumps(metadata["metrics_json"])

    # Build row with only expected fields
    row = {field: metadata.get(field, "") for field in _PROV_FIELDS}

    # Append to CSV
    with open(_PROV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PROV_FIELDS)
        writer.writerow(row)

    print(f"✓ Registered model {model_version} to {_PROV_PATH}")
    return model_version


# -------------------------------------------------------------------
# Record Prediction
# -------------------------------------------------------------------

def record_prediction(event: Dict) -> str:
    """
    Record a prediction event to prediction_events.csv.
    Required fields: userid, model_version, prediction, input_data
    Returns request_id.
    """
    event = event.copy()

    # Validate required fields
    required = ["userid", "model_version", "prediction", "input_data"]
    missing = [k for k in required if k not in event]
    if missing:
        raise ValueError(f"Missing required prediction fields: {missing}")

    # Generate IDs and timestamps
    request_id = event.get("request_id") or str(uuid.uuid4())
    event["request_id"] = request_id
    event.setdefault("timestamp", _now())

    # Compute input hash and statistics
    event["input_hash"] = _hash_input(event["input_data"])
    input_stats = _compute_input_stats(event["input_data"])
    event.update(input_stats)

    # Remove raw input data to save space
    del event["input_data"]

    # Convert extra JSON to string if needed
    if "extra_json" in event and not isinstance(event["extra_json"], str):
        event["extra_json"] = json.dumps(event["extra_json"])

    # Ensure CSV exists
    _ensure_csv(_PRED_PATH, _PRED_FIELDS)

    # Build row with only expected fields
    row = {field: event.get(field, "") for field in _PRED_FIELDS}

    # Append to CSV
    with open(_PRED_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PRED_FIELDS)
        writer.writerow(row)

    print(f"✓ Recorded prediction {request_id} for user {event['userid']}")
    return request_id


# -------------------------------------------------------------------
# Trace Prediction
# -------------------------------------------------------------------

def trace_prediction(request_id: str) -> Optional[Dict]:
    """
    Retrieve a prediction event and its associated model provenance.
    Returns dict with 'event' and 'provenance' keys, or None if not found.
    """
    if not os.path.exists(_PRED_PATH):
        print(f"No prediction events file found at {_PRED_PATH}")
        return None

    # Find prediction event
    event_row = None
    with open(_PRED_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("request_id") == request_id:
                event_row = row
                break

    if event_row is None:
        print(f"Request id {request_id} not found in {_PRED_PATH}")
        return None

    # Find corresponding model provenance
    model_version = event_row.get("model_version")
    provenance_row = None

    if model_version and os.path.exists(_PROV_PATH):
        with open(_PROV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model_version") == model_version:
                    provenance_row = row
                    break

    return {
        "event": event_row,
        "provenance": provenance_row
    }


# -------------------------------------------------------------------
# Query Functions
# -------------------------------------------------------------------

def get_model_by_version(model_version: str) -> Optional[Dict]:
    """Retrieve model provenance by version."""
    if not os.path.exists(_PROV_PATH):
        return None

    with open(_PROV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model_version") == model_version:
                return row
    return None


def get_predictions_by_model(model_version: str, limit: int = 100) -> list:
    """Retrieve recent predictions for a given model version."""
    if not os.path.exists(_PRED_PATH):
        return []

    predictions = []
    with open(_PRED_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model_version") == model_version:
                predictions.append(row)

    # Return most recent first
    return predictions[-limit:] if predictions else []


def get_predictions_by_user(userid: str, limit: int = 50) -> list:
    """Retrieve recent predictions for a given user."""
    if not os.path.exists(_PRED_PATH):
        return []

    predictions = []
    with open(_PRED_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("userid") == userid:
                predictions.append(row)

    # Return most recent first
    return predictions[-limit:] if predictions else []


def get_all_models() -> list:
    """Retrieve all registered models."""
    if not os.path.exists(_PROV_PATH):
        return []

    models = []
    with open(_PROV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append(row)
    return models


def get_all_predictions(limit: int = 1000) -> list:
    """Retrieve all prediction events."""
    if not os.path.exists(_PRED_PATH):
        return []

    predictions = []
    with open(_PRED_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(row)

    # Return most recent first
    return predictions[-limit:] if predictions else []
