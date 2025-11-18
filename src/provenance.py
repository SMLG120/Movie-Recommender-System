# Below is sampe code of provenance.py
import os
import csv
import json
import uuid
from datetime import datetime
from typing import Dict, Optional

# ----------------------------
# PATHS
# ----------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")
_PROV_PATH = os.path.join(_MODELS_DIR, "provenance.csv")
_PRED_EVENTS_PATH = os.path.join(_MODELS_DIR, "provenance_predictions.csv")

# ----------------------------
# CSV SCHEMAS
# ----------------------------

# Updated provenance fields
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

    # NEW: dataset statistics
    "training_row_count",
    "training_missing_values_json",

    "metrics_json",
    "artifact_path"
]

# Updated per-prediction fields
_PRED_FIELDS = [
    "request_id",
    "timestamp",
    "userid",
    "model_version",
    "mode_id",                # NEW
    "artifact_path",
    "input_hash",
    "prediction",

    # NEW optional input statistics
    "input_row_count",
    "input_missing_count",

    "extra_json"
]


# ----------------------------
# UTILITIES
# ----------------------------

def _ensure_models_dir():
    if not os.path.exists(_MODELS_DIR):
        os.makedirs(_MODELS_DIR, exist_ok=True)


def _ensure_csv(path: str, headers: list):
    """Create CSV if it doesn't exist."""
    _ensure_models_dir()
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()


# ----------------------------
# REGISTER MODEL
# ----------------------------

def register_model(metadata: Dict, prov_path: str = _PROV_PATH) -> str:
    """
    Register model build provenance.
    Required minimal fields: git_commit, artifact_path, training_data info.

    NEW:
    - training_row_count
    - training_missing_values_json
    """
    _ensure_csv(prov_path, _PROV_FIELDS)

    metadata = metadata.copy()

    # Generate model version if missing
    model_version = metadata.get("model_version") or \
        datetime.utcnow().strftime("v%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    metadata["model_version"] = model_version

    metadata.setdefault("model_tag", model_version)
    metadata.setdefault("build_time", datetime.utcnow().isoformat() + "Z")

    # Convert to JSON strings
    if "metrics_json" in metadata and not isinstance(metadata["metrics_json"], str):
        metadata["metrics_json"] = json.dumps(metadata["metrics_json"])

    if "training_missing_values_json" in metadata and \
       not isinstance(metadata.get("training_missing_values_json"), str):
        metadata["training_missing_values_json"] = json.dumps(
            metadata["training_missing_values_json"]
        )

    # Ensure all fields exist (missing ones become "")
    row = {k: metadata.get(k, "") for k in _PROV_FIELDS}

    with open(prov_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PROV_FIELDS)
        writer.writerow(row)

    return model_version


# ----------------------------
# RECORD PREDICTION
# ----------------------------

def record_prediction(event: Dict, events_path: str = _PRED_EVENTS_PATH) -> None:
    """
    Record a prediction event.
    Minimum required keys: userid, model_version, prediction.

    NEW:
    - mode_id
    - input stats (optional)
    """
    _ensure_csv(events_path, _PRED_FIELDS)

    event = event.copy()

    # Auto-fill required fields
    event.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
    event.setdefault("request_id", str(uuid.uuid4()))
    event.setdefault("input_hash", "")
    event.setdefault("mode_id", "")

    # Required validation
    required = ["userid", "model_version", "prediction"]
    missing = [k for k in required if k not in event]
    if missing:
        raise ValueError(f"Missing required fields in prediction event: {missing}")

    # Convert extra_json to string
    if "extra_json" in event and not isinstance(event["extra_json"], str):
        event["extra_json"] = json.dumps(event["extra_json"])

    # Ensure all expected columns
    row = {k: event.get(k, "") for k in _PRED_FIELDS}

    with open(events_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PRED_FIELDS)
        writer.writerow(row)


# ----------------------------
# TRACE PREDICTION
# ----------------------------

def trace_prediction(request_id: str,
                     events_path: str = _PRED_EVENTS_PATH,
                     prov_path: str = _PROV_PATH) -> Optional[Dict]:
    """
    Return {"event": {…}, "provenance": {…}} for a given request_id.
    """
    if not os.path.exists(events_path):
        return None

    # Find the prediction event
    with open(events_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        event_row = next((r for r in reader if r["request_id"] == request_id), None)

    if event_row is None:
        return None

    # Now find the model provenance entry
    model_version = event_row["model_version"]
    prov_row = None

    if os.path.exists(prov_path):
        with open(prov_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            prov_row = next((r for r in reader if r["model_version"] == model_version), None)

    return {"event": event_row, "provenance": prov_row}
