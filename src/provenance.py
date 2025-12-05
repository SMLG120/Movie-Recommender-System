import os
import csv
import json
import uuid
import hashlib
import subprocess
from datetime import datetime
from typing import Dict, Optional, Any, List


# -------------------------------------------------------------------
# Paths Setup
# -------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

_PROV_PATH = os.path.join(_MODELS_DIR, "model_provenance.csv")
_DATA_PROV_PATH = os.path.join(_MODELS_DIR, "data_provenance.csv")
_PRED_PATH = os.path.join(_MODELS_DIR, "prediction_events.csv")
_EVAL_PATH = os.path.join(_MODELS_DIR, "evaluation_events.csv")

# CSV Headers for Model Provenance
_MODEL_PROV_FIELDS = [
    "model_version",
    "model_tag",
    "build_time",
    "git_commit",
    "git_branch",
    "pipeline_version",
    "training_data_id",
    "training_row_count",
    "model_type",
    "metrics_json",
    "artifact_path",
    "framework_versions"
]

# CSV Headers for Data Provenance
_DATA_PROV_FIELDS = [
    "data_id",
    "timestamp",
    "data_source",
    "file_path",
    "file_hash",
    "row_count",
    "column_count",
    "missing_values_json",
    "date_range_start",
    "date_range_end"
]

# CSV Headers for Prediction Events (Online)
_PRED_FIELDS = [
    "request_id",
    "timestamp",
    "userid",
    "model_version",
    "prediction",
    "input_hash",
    "input_row_count",
    "input_missing_count",
    "inference_latency_ms",
    "extra_json"
]

# CSV Headers for Evaluation Events (Offline)
_EVAL_FIELDS = [
    "eval_id",
    "timestamp",
    "eval_type",
    "model_version",
    "preproc_path",
    "model_path",
    "eval_data_path",
    "test_set_size",
    "rmse",
    "mae",
    "r2",
    "precision",
    "recall",
    "f1",
    "accuracy",
    "inference_time_ms",
    "metrics_json"
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


def _ensure_csv(path: str, headers: list):
    """Create CSV file with headers if it doesn't exist."""
    _ensure_models_dir()
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    try:
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return "unknown"


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

    if hasattr(data, "shape"):
        stats["input_row_count"] = int(data.shape[0])
        if hasattr(data, "isnull"):
            stats["input_missing_count"] = int(data.isnull().sum().sum())
        return stats

    if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
        stats["input_row_count"] = len(data)
        missing = sum(1 for row in data for v in row.values() if v is None)
        stats["input_missing_count"] = missing
        return stats

    return stats


# -------------------------------------------------------------------
# Register Training Data Provenance
# -------------------------------------------------------------------

def register_training_data(data_metadata: Dict) -> str:
    """
    Register training data provenance.
    Required fields: data_source, file_path
    Returns data_id.
    """
    data_metadata = data_metadata.copy()

    required = ["data_source", "file_path"]
    missing = [r for r in required if r not in data_metadata]
    if missing:
        raise ValueError(f"Missing required data metadata fields: {missing}")

    data_id = data_metadata.get("data_id") or (
        "data_" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    )
    data_metadata["data_id"] = data_id
    data_metadata.setdefault("timestamp", _now())

    if os.path.exists(data_metadata["file_path"]):
        data_metadata.setdefault("file_hash", _compute_file_hash(data_metadata["file_path"]))

    _ensure_csv(_DATA_PROV_PATH, _DATA_PROV_FIELDS)

    row = {field: data_metadata.get(field, "") for field in _DATA_PROV_FIELDS}
    with open(_DATA_PROV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_DATA_PROV_FIELDS)
        writer.writerow(row)

    return data_id


# -------------------------------------------------------------------
# Register Model Provenance
# -------------------------------------------------------------------

def register_model(metadata: Dict) -> str:
    """
    Store model metadata in model_provenance.csv.
    Required fields: artifact_path, training_data_id
    Returns model_version.
    """
    metadata = metadata.copy()

    required = ["artifact_path", "training_data_id"]
    missing = [r for r in required if r not in metadata]
    if missing:
        raise ValueError(f"Missing required model metadata fields: {missing}")

    metadata.setdefault("git_commit", _get_git_commit())
    metadata.setdefault("git_branch", _get_git_branch())

    model_version = metadata.get("model_version") or (
        datetime.utcnow().strftime("v%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    )
    metadata["model_version"] = model_version
    metadata.setdefault("model_tag", model_version)
    metadata.setdefault("build_time", _now())
    metadata.setdefault("pipeline_version", "pipeline_v1.0")
    metadata.setdefault("model_type", "XGBoost")

    for field in ["metrics_json", "framework_versions"]:
        if field in metadata and not isinstance(metadata[field], str):
            metadata[field] = json.dumps(metadata[field])

    _ensure_csv(_PROV_PATH, _MODEL_PROV_FIELDS)

    row = {field: metadata.get(field, "") for field in _MODEL_PROV_FIELDS}
    with open(_PROV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_MODEL_PROV_FIELDS)
        writer.writerow(row)

    return model_version


# -------------------------------------------------------------------
# Record Online Prediction
# -------------------------------------------------------------------

def record_prediction(event: Dict) -> str:
    """
    Record a prediction event to prediction_events.csv (online).
    Required fields: userid, model_version, prediction, input_data
    Returns request_id.
    """
    event = event.copy()

    required = ["userid", "model_version", "prediction", "input_data"]
    missing = [k for k in required if k not in event]
    if missing:
        raise ValueError(f"Missing required prediction fields: {missing}")

    request_id = event.get("request_id") or str(uuid.uuid4())
    event["request_id"] = request_id
    event.setdefault("timestamp", _now())

    event["input_hash"] = _hash_input(event["input_data"])
    input_stats = _compute_input_stats(event["input_data"])
    event.update(input_stats)

    del event["input_data"]

    if "extra_json" in event and not isinstance(event["extra_json"], str):
        event["extra_json"] = json.dumps(event["extra_json"])

    _ensure_csv(_PRED_PATH, _PRED_FIELDS)

    row = {field: event.get(field, "") for field in _PRED_FIELDS}
    with open(_PRED_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PRED_FIELDS)
        writer.writerow(row)

    return request_id


# -------------------------------------------------------------------
# Record Offline Evaluation
# -------------------------------------------------------------------

def record_evaluation(eval_metadata: Dict) -> str:
    """
    Record offline evaluation results to evaluation_events.csv.
    Required fields: model_version, eval_type, preproc_path, model_path, eval_data_path
    Optional fields: rmse, mae, r2, precision, recall, f1, accuracy, inference_time_ms, test_set_size
    Returns eval_id.
    """
    eval_metadata = eval_metadata.copy()

    required = ["model_version", "eval_type", "preproc_path", "model_path", "eval_data_path"]
    missing = [r for r in required if r not in eval_metadata]
    if missing:
        raise ValueError(f"Missing required evaluation fields: {missing}")

    eval_id = eval_metadata.get("eval_id") or (
        "eval_" + datetime.utcnow().strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    )
    eval_metadata["eval_id"] = eval_id
    eval_metadata.setdefault("timestamp", _now())

    # Convert metrics dict to JSON string if needed
    if "metrics_json" in eval_metadata and not isinstance(eval_metadata["metrics_json"], str):
        eval_metadata["metrics_json"] = json.dumps(eval_metadata["metrics_json"])

    _ensure_csv(_EVAL_PATH, _EVAL_FIELDS)

    row = {field: eval_metadata.get(field, "") for field in _EVAL_FIELDS}
    with open(_EVAL_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_EVAL_FIELDS)
        writer.writerow(row)

    return eval_id


# -------------------------------------------------------------------
# Trace Prediction
# -------------------------------------------------------------------

def trace_prediction(request_id: str) -> Optional[Dict]:
    """
    Retrieve a prediction event and its associated model/data provenance.
    Returns dict with 'event', 'model_provenance', and 'data_provenance'.
    """
    if not os.path.exists(_PRED_PATH):
        return None

    event_row = None
    with open(_PRED_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("request_id") == request_id:
                event_row = row
                break

    if event_row is None:
        return None

    model_version = event_row.get("model_version")
    model_prov_row = None
    if model_version and os.path.exists(_PROV_PATH):
        with open(_PROV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model_version") == model_version:
                    model_prov_row = row
                    break

    data_prov_row = None
    if model_prov_row and os.path.exists(_DATA_PROV_PATH):
        training_data_id = model_prov_row.get("training_data_id")
        with open(_DATA_PROV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("data_id") == training_data_id:
                    data_prov_row = row
                    break

    return {
        "event": event_row,
        "model_provenance": model_prov_row,
        "data_provenance": data_prov_row
    }


# -------------------------------------------------------------------
# Trace Evaluation
# -------------------------------------------------------------------

def trace_evaluation(eval_id: str) -> Optional[Dict]:
    """
    Retrieve evaluation event and its associated model provenance.
    Returns dict with 'eval_event' and 'model_provenance'.
    """
    if not os.path.exists(_EVAL_PATH):
        return None

    eval_row = None
    with open(_EVAL_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("eval_id") == eval_id:
                eval_row = row
                break

    if eval_row is None:
        return None

    model_version = eval_row.get("model_version")
    model_prov_row = None
    if model_version and os.path.exists(_PROV_PATH):
        with open(_PROV_PATH, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("model_version") == model_version:
                    model_prov_row = row
                    break

    return {
        "eval_event": eval_row,
        "model_provenance": model_prov_row
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


def get_all_models() -> List[Dict]:
    """Retrieve all registered models."""
    if not os.path.exists(_PROV_PATH):
        return []

    models = []
    with open(_PROV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append(row)
    return models


def get_predictions_by_model(model_version: str, limit: int = 100) -> List[Dict]:
    """Retrieve recent predictions for a given model version."""
    if not os.path.exists(_PRED_PATH):
        return []

    predictions = []
    with open(_PRED_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model_version") == model_version:
                predictions.append(row)

    return predictions[-limit:] if predictions else []


def get_predictions_by_user(userid: str, limit: int = 50) -> List[Dict]:
    """Retrieve recent predictions for a given user."""
    if not os.path.exists(_PRED_PATH):
        return []

    predictions = []
    with open(_PRED_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("userid") == userid:
                predictions.append(row)

    return predictions[-limit:] if predictions else []


def get_evaluations_by_model(model_version: str, limit: int = 50) -> List[Dict]:
    """Retrieve evaluation events for a given model version."""
    if not os.path.exists(_EVAL_PATH):
        return []

    evaluations = []
    with open(_EVAL_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("model_version") == model_version:
                evaluations.append(row)

    return evaluations[-limit:] if evaluations else []


def get_all_predictions(limit: int = 1000) -> List[Dict]:
    """Retrieve all online prediction events."""
    if not os.path.exists(_PRED_PATH):
        return []

    predictions = []
    with open(_PRED_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            predictions.append(row)

    return predictions[-limit:] if predictions else []


def get_all_evaluations(limit: int = 100) -> List[Dict]:
    """Retrieve all offline evaluation events."""
    if not os.path.exists(_EVAL_PATH):
        return []

    evaluations = []
    with open(_EVAL_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            evaluations.append(row)

    return evaluations[-limit:] if evaluations else []