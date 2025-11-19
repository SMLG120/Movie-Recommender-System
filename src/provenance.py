import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Optional, Any

from dotenv import load_dotenv
from pymongo import MongoClient

# -------------------------------------------------------------------
# Load environment variables
# -------------------------------------------------------------------
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("ERROR: MONGO_URI is not set in your .env file")

client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
db = client["provenance_db"]

provenance_collection = db["model_provenance"]
predictions_collection = db["prediction_events"]

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _now() -> str:
    """Return current time in ISO-8601 format."""
    return datetime.utcnow().isoformat() + "Z"


def _hash_input(data: Any) -> str:
    """Compute SHA-256 hash of an input object (list, dict, DF)."""
    try:
        # pandas / numpy support
        if hasattr(data, "to_dict"):
            data = data.to_dict()
    except Exception:
        pass

    json_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()


def _compute_input_stats(data: Any) -> Dict[str, Any]:
    """
    Compute:
      - row count
      - total missing values
    Supports:
      - pandas DataFrames
      - list of dicts
    """
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
        missing = 0
        for row in data:
            for v in row.values():
                if v is None:
                    missing += 1
        stats["input_missing_count"] = missing
        return stats

    # Unknown input type
    return stats


# -------------------------------------------------------------------
# REGISTER MODEL
# -------------------------------------------------------------------

def register_model(metadata: Dict) -> str:
    """
    Store model metadata in MongoDB with full provenance.
    Required: git_commit, artifact_path
    """

    metadata = metadata.copy()

    # Validate essential fields
    required = ["git_commit", "artifact_path"]
    missing = [r for r in required if r not in metadata]
    if missing:
        raise ValueError(f"Missing required model metadata fields: {missing}")

    model_version = metadata.get("model_version") or (
        datetime.utcnow().strftime("v%Y%m%dT%H%M%SZ-") +
        uuid.uuid4().hex[:8]
    )
    metadata["model_version"] = model_version
    metadata.setdefault("model_tag", model_version)
    metadata.setdefault("build_time", _now())

    # Convert nested fields to proper JSON strings for consistent storage
    for field in ["metrics_json", "training_missing_values_json"]:
        if field in metadata and not isinstance(metadata[field], str):
            metadata[field] = json.dumps(metadata[field])

    provenance_collection.insert_one(metadata)

    return model_version


# -------------------------------------------------------------------
# RECORD PREDICTION
# -------------------------------------------------------------------

def record_prediction(event: Dict) -> None:
    """
    Record a prediction event with:
      - request_id
      - model_version
      - model_id (optional)
      - input_hash
      - automatic input statistics
    """

    event = event.copy()

    # Check required fields
    required = ["userid", "model_version", "prediction", "input_data"]
    missing = [k for k in required if k not in event]
    if missing:
        raise ValueError(f"Missing required prediction fields: {missing}")

    # Generate IDs and timestamps
    event.setdefault("request_id", str(uuid.uuid4()))
    event.setdefault("timestamp", _now())

    # Fix typo: model_id instead of mode_id
    event.setdefault("model_id", None)

    # Compute input hash
    event["input_hash"] = _hash_input(event["input_data"])

    # Compute statistics
    input_stats = _compute_input_stats(event["input_data"])
    event.update(input_stats)

    # You may want to remove the raw data from MongoDB for storage size reasons
    del event["input_data"]

    # Convert any nested dicts to strings for consistency
    if "extra_json" in event and not isinstance(event["extra_json"], str):
        event["extra_json"] = json.dumps(event["extra_json"])

    predictions_collection.insert_one(event)


# -------------------------------------------------------------------
# TRACE PREDICTION
# -------------------------------------------------------------------

def trace_prediction(request_id: str) -> Optional[Dict]:
    """
    Return:
      - prediction event
      - associated model provenance
    """

    event = predictions_collection.find_one({"request_id": request_id})
    if not event:
        return None

    model_version = event.get("model_version")
    prov = None
    if model_version:
        prov = provenance_collection.find_one({"model_version": model_version})

    return {
        "event": event,
        "provenance": prov
    }

# -------------------------------------------------------
# Create MongoDB Indexes for high performance
# -------------------------------------------------------

provenance_collection.create_index("model_version", unique=True)
predictions_collection.create_index("request_id", unique=True)
predictions_collection.create_index("model_version")
predictions_collection.create_index("userid")
