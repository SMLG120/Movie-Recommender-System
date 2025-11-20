import os
import json
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Optional, Any

from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING

# -------------------------------------------------------------------
# Load Environment Variables
# -------------------------------------------------------------------

# Load .env from project root (parent of src/)
dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path)

MONGO_USER = os.getenv("MONGO_USER")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_HOST = os.getenv("MONGO_HOST", "mongodb")
MONGO_PORT = int(os.getenv("MONGO_PORT", 27017))

# Validate required environment variables
if not MONGO_USER or not MONGO_PASSWORD:
    raise ValueError(
        "Missing required environment variables: MONGO_USER and MONGO_PASSWORD. "
        "Please check your .env file."
    )

MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASSWORD}@{MONGO_HOST}:{MONGO_PORT}/{MONGO_DB}?authSource=admin"

# -------------------------------------------------------------------
# MongoDB Connection
# -------------------------------------------------------------------

try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command('ping')
    print(f"✓ Connected to MongoDB at {MONGO_HOST}:{MONGO_PORT}")
except Exception as e:
    print(f"✗ Failed to connect to MongoDB: {e}")
    raise

db = client[MONGO_DB]
provenance_collection = db["model_provenance"]
predictions_collection = db["prediction_events"]

# Create indexes for query performance (explained below)
provenance_collection.create_index([("model_version", ASCENDING)], unique=True)
predictions_collection.create_index([("request_id", ASCENDING)], unique=True)
predictions_collection.create_index([("model_version", ASCENDING)])
predictions_collection.create_index([("userid", ASCENDING)])
predictions_collection.create_index([("timestamp", ASCENDING)])


# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------

def _now() -> str:
    """Return current time in ISO-8601 format."""
    return datetime.utcnow().isoformat() + "Z"


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
# Register Model
# -------------------------------------------------------------------

def register_model(metadata: Dict) -> str:
    """
    Store model metadata in MongoDB.
    Required fields: git_commit, artifact_path
    """
    metadata = metadata.copy()

    required = ["git_commit", "artifact_path"]
    missing = [r for r in required if r not in metadata]
    if missing:
        raise ValueError(f"Missing required model metadata fields: {missing}")

    model_version = metadata.get("model_version") or (
        datetime.utcnow().strftime("v%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    )
    metadata["model_version"] = model_version
    metadata.setdefault("model_tag", model_version)
    metadata.setdefault("build_time", _now())

    for field in ["metrics_json", "training_missing_values_json"]:
        if field in metadata and not isinstance(metadata[field], str):
            metadata[field] = json.dumps(metadata[field])

    provenance_collection.insert_one(metadata)
    return model_version


# -------------------------------------------------------------------
# Record Prediction
# -------------------------------------------------------------------

def record_prediction(event: Dict) -> str:
    """
    Record a prediction event with input statistics and hashing.
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

    predictions_collection.insert_one(event)
    return request_id


# -------------------------------------------------------------------
# Trace Prediction
# -------------------------------------------------------------------

def trace_prediction(request_id: str) -> Optional[Dict]:
    """
    Retrieve a prediction event and its associated model provenance.
    Returns dict with 'event' and 'provenance' keys, or None if not found.
    """
    event = predictions_collection.find_one({"request_id": request_id})
    if not event:
        return None

    model_version = event.get("model_version")
    provenance = None
    if model_version:
        provenance = provenance_collection.find_one({"model_version": model_version})

    return {
        "event": event,
        "provenance": provenance
    }


# -------------------------------------------------------------------
# Query Functions
# -------------------------------------------------------------------

def get_model_by_version(model_version: str) -> Optional[Dict]:
    """Retrieve model provenance by version."""
    return provenance_collection.find_one({"model_version": model_version})


def get_predictions_by_model(model_version: str, limit: int = 100) -> list:
    """Retrieve recent predictions for a given model version."""
    return list(
        predictions_collection.find({"model_version": model_version})
        .sort("timestamp", -1)
        .limit(limit)
    )


def get_predictions_by_user(userid: str, limit: int = 50) -> list:
    """Retrieve recent predictions for a given user."""
    return list(
        predictions_collection.find({"userid": userid})
        .sort("timestamp", -1)
        .limit(limit)
    )
