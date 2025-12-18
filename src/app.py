#!/usr/bin/env python3

import os
import sys
import logging
import random
import time
from datetime import datetime
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
from collections import OrderedDict

from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram
from prometheus_flask_exporter import PrometheusMetrics

# Ensure repo root on sys.path so src imports resolve when running from src/
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.inference import RecommenderEngine  

# Extend default buckets up to 5 minutes
Histogram.DEFAULT_BUCKETS = (
    0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 90, 120, 180, 300
)

# Config
PORT = int(os.getenv("PORT", "8080"))
MODEL_V1_PATH = os.getenv("MODEL_PATH", "src/models/v1")
MODEL_V2_PATH = os.getenv("MODEL_CANARY_PATH", "src/models/v2")
MOVIES_FILE = os.getenv("MOVIES_FILE", "data/raw_data/movies.csv")

# Canonical repo-root and canary config used by evaluator and app
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_V1_PATH = os.getenv("MODEL_PATH", str(REPO_ROOT / "src" / "models" / "v1"))
MODEL_V2_PATH = os.getenv("MODEL_CANARY_PATH", str(REPO_ROOT / "src" / "models" / "v2"))
MOVIES_FILE = os.getenv("MOVIES_FILE", str(REPO_ROOT / "data" / "raw_data" / "movies.csv"))

# Canonical canary config and telemetry log
CANARY_CONFIG = REPO_ROOT / "tests" / "deployment" / "config" / "canary_config.json"
#DEFAULT_CANARY_PERCENTAGE = float(os.getenv("CANARY_PERCENTAGE", "10"))
ONLINE_LOG = REPO_ROOT / "evaluation" / "Online" / "logs" / "online_metrics.json"


# App + Logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reco-api")

metrics = PrometheusMetrics(app, path="/metrics", group_by="endpoint")


# Canary helpers (reads the same config the evaluator writes)
def get_current_canary_percentage() -> float:
    if not CANARY_CONFIG.exists():
        return 0.0 #DEFAULT_CANARY_PERCENTAGE
    try:
        with CANARY_CONFIG.open() as f:
            cfg = json.load(f)
        return max(0.0, min(100.0, float(cfg.get("canary_percentage", 0.0 )))) #(cfg.get("canary_percentage",DEFAULT_CANARY_PERCENTAGE)
    except Exception:
        return 0.0 #DEFAULT_CANARY_PERCENTAGE


def append_online_recommendation(user_id: int, items: List[str], model_version: str) -> None:
    ONLINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if ONLINE_LOG.exists():
        try:
            with ONLINE_LOG.open() as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    data.setdefault("recommendations", [])
    data.setdefault("user_interactions", [])
    data.setdefault("recommendation_quality", [])

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": str(user_id),
        "items": items,
        "model_version": model_version,
    }
    data["recommendations"].append(entry)

    with ONLINE_LOG.open("w") as f:
        json.dump(data, f, indent=2)



# Model-quality metrics 

RECO_SERVED  = Counter("model_reco_served_total", "Recommendations served to users", registry=metrics.registry)
CTR_HITS     = Counter("model_ctr_at_k_total", "Clicks/engagements that match served K-list", registry=metrics.registry)
HITRATE_HITS = Counter("model_hits_at_k_total", "Hits@K that match served K-list", registry=metrics.registry)
MAE_SUM      = Counter("model_mae_sum", "Sum of absolute errors |y - y_hat|", registry=metrics.registry)
RMSE_SSE     = Counter("model_rmse_sse", "Sum of squared errors (y - y_hat)^2", registry=metrics.registry)
ERR_COUNT    = Counter("model_err_count", "Count of labeled events contributing to errors", registry=metrics.registry)

for c in (RECO_SERVED, CTR_HITS, HITRATE_HITS, MAE_SUM, RMSE_SSE, ERR_COUNT):
    c.inc(0)

# Load model
# try:
#     recommender_engine = RecommenderEngine(
#         model_dir= MODEL_V1_PATH,
#         movies_file=MOVIES_FILE,
#     )
#     # Canary model
#     recommender_engine_canary = RecommenderEngine(
#         model_dir= MODEL_V2_PATH,
#         movies_file=MOVIES_FILE,
#     )
#     logger.info("Both RecommenderEngines loaded successfully.")
# except Exception as e:
#     logger.exception("Failed to load both RecommenderEngines: %s", e)
#     recommender_engine, recommender_engine_canary = None, None

# Helper to check if new model (v2) for canary exists
# NOTE: Later, we can edit this to check for specific model files
def model_dir_has_artifact(model_dir: str | Path) -> bool:
    p = Path(model_dir)
    if not p.exists() or not p.is_dir():
        return False
    return any(p.iterdir())

# Load stable model (v1) — failure is critical (service degraded)
try:
    recommender_engine = RecommenderEngine(
        model_dir=MODEL_V1_PATH,
        movies_file=MOVIES_FILE,
    )
    logger.info("Stable model (v1) loaded from %s", MODEL_V1_PATH)
except Exception:
    logger.exception("Failed to load stable model from %s", MODEL_V1_PATH)
    recommender_engine = None

# Load canary model (v2) only if artifacts exist
recommender_engine_canary = None
try:
    if model_dir_has_artifact(MODEL_V2_PATH):
        try:
            recommender_engine_canary = RecommenderEngine(
                model_dir=MODEL_V2_PATH,
                movies_file=MOVIES_FILE,
            )
            logger.info("Canary model (v2) loaded from %s", MODEL_V2_PATH)
        except Exception:
            logger.exception("Failed to load canary model from %s; disabling canary", MODEL_V2_PATH)
            recommender_engine_canary = None
            # ensure canary percentage is set to 0 when canary load fails
            try:
                CANARY_CONFIG.parent.mkdir(parents=True, exist_ok=True)
                with CANARY_CONFIG.open("w") as f:
                    json.dump({"canary_percentage": 0.0}, f, indent=2)
                logger.info("Wrote canary config with 0%% due to failed canary load")
            except Exception:
                logger.exception("Failed to write canary config to disable canary")
    else:
        logger.info("No canary artifacts found at %s; canary disabled.", MODEL_V2_PATH)
        # explicitly record canary as 0% so evaluator and other tooling see it
        try:
            CANARY_CONFIG.parent.mkdir(parents=True, exist_ok=True)
            with CANARY_CONFIG.open("w") as f:
                json.dump({"canary_percentage": 0.0}, f, indent=2)
            logger.info("Wrote canary config with 0%% because v2 dir is empty or missing")
        except Exception:
            logger.exception("Failed to write canary config to disable canary")
except Exception:
    logger.exception("Unexpected error during canary load detection; canary disabled.")
    recommender_engine_canary = None

# Bounded in-memory LRU caches for recent served lists and model provenance
# Keys are strings (user_id as string) to match telemetry JSON keys
LAST_SERVED_MAX = int(os.getenv("LAST_SERVED_MAX", "10000"))
LAST_SERVED: "OrderedDict[str, List[str]]" = OrderedDict()
LAST_MODEL: "OrderedDict[str, str]" = OrderedDict()


def _lru_put(d: OrderedDict, key: str, value, maxsize: int) -> None:
    if key in d:
        try:
            del d[key]
        except Exception:
            pass
    d[key] = value
    # evict oldest while over capacity
    while len(d) > maxsize:
        try:
            d.popitem(last=False)
        except Exception:
            break


def set_last_served(user_key: str, recs: List[str]) -> None:
    _lru_put(LAST_SERVED, user_key, recs, LAST_SERVED_MAX)


def get_last_served(user_key: str) -> List[str]:
    return LAST_SERVED.get(user_key, [])


def set_last_model(user_key: str, model_version: str) -> None:
    _lru_put(LAST_MODEL, user_key, model_version, LAST_SERVED_MAX)


def get_last_model(user_key: str) -> str:
    return LAST_MODEL.get(user_key, "unknown")

#random selection between models
# two options: ticky or random
# TODO: come up with a better way to select models
def select_engine():
    if recommender_engine_canary is None:
        return recommender_engine, "v1"
    pct = get_current_canary_percentage()
    if random.random() * 100 < pct:
        return recommender_engine_canary, "v2"
    return recommender_engine, "v1"

# Routes
@app.route("/", methods=["GET"])
def root():
    return Response("Movie Recommender API", mimetype="text/plain")

@app.route("/_live", methods=["GET"])
def live():
    """Liveness probe."""
    return Response("OK", mimetype="text/plain"), 200

@app.route("/_ready", methods=["GET"])
def ready():
    """Readiness probe: model must be loaded."""
    ok = recommender_engine is not None
    return (Response("READY", mimetype="text/plain"), 200) if ok \
        else (Response("NOT_READY", mimetype="text/plain"), 503)

@app.route("/health", methods=["GET"])
def health():
    status = "OK" if recommender_engine is not None else "Service Degraded"
    code = 200 if recommender_engine is not None else 503
    canary_loaded = recommender_engine_canary is not None
    canary_pct = get_current_canary_percentage()
    return jsonify({
        "status": status,
        "canary_loaded": canary_loaded,
        "canary_percentage": float(canary_pct),
    }), code

@app.route("/recommend", methods=["GET"])
def recommend():
    """
    Serve recommendations.
    """
    if recommender_engine is None:
        return jsonify({"error": "model not loaded"}), 503
    try:
        user_id = int(request.args.get("user_id", ""))
    except Exception:
        return jsonify({"error": "user_id (int) is required"}), 400

    top_n = int(request.args.get("top_n", 10))
    try:
        engine, model_version = select_engine()
        if engine is None:
            return jsonify({"error": "no model loaded"}), 503

        recs_csv = engine.recommend(user_id, top_n=top_n)
        recs = [x.strip() for x in recs_csv.split(",") if x.strip()]
        # track served list for CTR/HitRate and provenance (model_version)
        key = str(user_id)
        set_last_served(key, recs)
        set_last_model(key, model_version)
        RECO_SERVED.inc()

        # Append to online metrics with model_version for evaluator
        try:
            append_online_recommendation(user_id, recs, model_version)
        except Exception:
            logger.exception("Failed to append online recommendation")

        return jsonify({"user_id": user_id, "recommendations": recs, "model_version": model_version}), 200
    except Exception as e:
        logger.exception("recommend failed: %s", e)
        return jsonify({"error": str(e)}), 500

@app.route("/event/click", methods=["POST"])
def event_click():
    """
    Record a click/engagement event.
    """
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    try:
        user_id = int(data.get("user_id", ""))
        item_id = str(data.get("item_id", ""))
    except Exception:
        return jsonify({"error": "user_id(int) and item_id(str) required"}), 400

    key = str(user_id)
    served = get_last_served(key)
    model_version = get_last_model(key)
    if item_id in served:
        CTR_HITS.inc()
        HITRATE_HITS.inc()
        try:
            append_online_interaction(user_id, item_id, model_version, True)
        except Exception:
            logger.exception("Failed to append online interaction")
        return jsonify({"recorded": True, "matched_served": True}), 200
    try:
        append_online_interaction(user_id, item_id, model_version, False)
    except Exception:
        logger.exception("Failed to append online interaction")
    return jsonify({"recorded": True, "matched_served": False}), 200

@app.route("/event/rating", methods=["POST"])
def event_rating():
    """
    Record a rating with the predicted score to compute online errors.
    """
    data: Dict[str, Any] = request.get_json(silent=True) or {}
    try:
        user_id = int(data.get("user_id", ""))
    except Exception:
        return jsonify({"error": "user_id (int) is required"}), 400

    try:
        rating = float(data["rating"])
    except Exception:
        return jsonify({"error": "rating (float) is required"}), 400

    pred_raw: Optional[float] = data.get("predicted")
    if pred_raw is None:
        return jsonify({"error": "predicted (float) is required for online error"}), 400

    try:
        yhat = float(pred_raw)
    except Exception:
        return jsonify({"error": "predicted must be float"}), 400

    err = rating - yhat
    MAE_SUM.inc(abs(err))
    RMSE_SSE.inc(err * err)
    ERR_COUNT.inc()

    model_version = get_last_model(str(user_id))
    try:
        append_recommendation_quality(user_id, rating, yhat, model_version)
    except Exception:
        logger.exception("Failed to append recommendation quality")

    return jsonify({"recorded": True}), 200


def append_online_interaction(user_id: int, item_id: str, model_version: str, matched_served: bool) -> None:
    ONLINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if ONLINE_LOG.exists():
        try:
            with ONLINE_LOG.open() as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    data.setdefault("recommendations", [])
    data.setdefault("user_interactions", [])
    data.setdefault("recommendation_quality", [])

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": str(user_id),
        "item_id": str(item_id),
        "model_version": model_version,
        "matched_served": bool(matched_served),
    }
    data["user_interactions"].append(entry)

    with ONLINE_LOG.open("w") as f:
        json.dump(data, f, indent=2)


def append_recommendation_quality(user_id: int, rating: float, predicted: float, model_version: str) -> None:
    ONLINE_LOG.parent.mkdir(parents=True, exist_ok=True)
    if ONLINE_LOG.exists():
        try:
            with ONLINE_LOG.open() as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    data.setdefault("recommendations", [])
    data.setdefault("user_interactions", [])
    data.setdefault("recommendation_quality", [])

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": str(user_id),
        "rating": float(rating),
        "predicted": float(predicted),
        "model_version": model_version,
    }
    data["recommendation_quality"].append(entry)

    with ONLINE_LOG.open("w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    logger.info(f"Starting Movie Recommender API on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
