import os
import csv
import json
import uuid
from datetime import datetime

from provenance import (
    register_model as register_model_mongo,
    record_prediction as record_prediction_mongo,
)

# Paths
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models")

_PROV_PATH = os.path.join(_MODELS_DIR, "provenance.csv")
_PRED_PATH = os.path.join(_MODELS_DIR, "prediction_events.csv")

_PROV_FIELDS = [
    "model_version",
    "timestamp",
    "git_commit",
    "artifact_path",
    "metrics_json",
    "training_row_count",
    "training_missing_values_json"
]

_PRED_FIELDS = [
    "request_id",
    "timestamp",
    "userid",
    "model_version",
    "model_id",
    "prediction",
    "input_hash",
    "input_row_count",
    "input_missing_count"
]

def _ensure_csv(path, headers):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

# -------------------------------------------------------
# REGISTER MODEL (MongoDB + CSV)
# -------------------------------------------------------

def register_model(metadata: dict) -> str:
    model_version = register_model_mongo(metadata)   # save in MongoDB

    _ensure_csv(_PROV_PATH, _PROV_FIELDS)

    row = {
        "model_version": model_version,
        "timestamp": metadata.get("build_time", datetime.utcnow().isoformat() + "Z"),
        "git_commit": metadata["git_commit"],
        "artifact_path": metadata["artifact_path"],
        "metrics_json": json.dumps(metadata.get("metrics_json", {})),
        "training_row_count": metadata.get("training_row_count", ""),
        "training_missing_values_json": json.dumps(metadata.get("training_missing_values_json", {})),
    }

    with open(_PROV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PROV_FIELDS)
        writer.writerow(row)

    return model_version

# -------------------------------------------------------
# RECORD PREDICTION (MongoDB + CSV)
# -------------------------------------------------------

def record_prediction(event: dict):
    request_id = record_prediction_mongo(event)  # save in MongoDB

    _ensure_csv(_PRED_PATH, _PRED_FIELDS)

    row = {
        "request_id": event.get("request_id"),
        "timestamp": event.get("timestamp"),
        "userid": event["userid"],
        "model_version": event["model_version"],
        "model_id": event.get("model_id", ""),
        "prediction": event["prediction"],
        "input_hash": event.get("input_hash", ""),
        "input_row_count": event.get("input_row_count", ""),
        "input_missing_count": event.get("input_missing_count", "")
    }

    with open(_PRED_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_PRED_FIELDS)
        writer.writerow(row)

    return request_id
