#!/usr/bin/env python3
"""
Movie Recommender API – Production Server with Monitoring
- Availability metrics via prometheus_flask_exporter
- Model-quality metrics (CTR@K, HitRate@K, Online MAE/RMSE aggregates)
- Health endpoints
"""

import os
import logging
from typing import List, Dict, Any, Optional
import json
import pprint

from flask import Flask, request, jsonify, Response
from prometheus_client import Counter
from prometheus_flask_exporter import PrometheusMetrics

# Local imports (paths set for Docker and local runs)
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.inference import RecommenderEngine  
from provenance import (
    register_model,
    record_prediction,
    trace_prediction,
    get_model_by_version,
    get_predictions_by_model
)

from prometheus_client import Histogram
# Extend default buckets up to 5 minutes
Histogram.DEFAULT_BUCKETS = (
    0.1, 0.25, 0.5, 1, 2.5, 5, 10, 20, 30, 60, 90, 120, 180, 300
)

# Config
PORT = int(os.getenv("PORT", "8080"))
MODEL_PATH = os.getenv("MODEL_PATH", "src/models")
MOVIES_FILE = os.getenv("MOVIES_FILE", "data/raw_data/movies.csv")

# App + Logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reco-api")

metrics = PrometheusMetrics(app, path="/metrics", group_by="endpoint")


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
try:
    recommender_engine = RecommenderEngine(
        model_dir= MODEL_PATH,
        movies_file=MOVIES_FILE,
    )
    logger.info("RecommenderEngine loaded successfully.")
except Exception as e:
    logger.exception("Failed to load RecommenderEngine: %s", e)
    recommender_engine = None


LAST_SERVED: Dict[int, List[str]] = {}


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
    return jsonify({"status": status}), code

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
        recs_csv = recommender_engine.recommend(user_id, top_n=top_n)
        recs = [x.strip() for x in recs_csv.split(",") if x.strip()]
        # track served list for CTR/HitRate
        LAST_SERVED[user_id] = recs
        RECO_SERVED.inc()
        return jsonify({"user_id": user_id, "recommendations": recs}), 200
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

    served = LAST_SERVED.get(user_id, [])
    if item_id in served:
        CTR_HITS.inc()
        HITRATE_HITS.inc()
        return jsonify({"recorded": True, "matched_served": True}), 200
    return jsonify({"recorded": True, "matched_served": False}), 200

@app.route("/event/rating", methods=["POST"])
def event_rating():
    """
    Record a rating with the predicted score to compute online errors.
    """
    data: Dict[str, Any] = request.get_json(silent=True) or {}
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

    return jsonify({"recorded": True}), 200


@app.route("/provenance/test", methods=["POST"])
def provenance_test():
    """
    Test the provenance system: register model, record predictions, trace prediction, query predictions.
    """
    print("\n" + "="*60)
    print("Provenance System - Test Run")
    print("="*60 + "\n")

    # Step 1: Register a model
    print("Step 1: Registering model...")
    model_version = register_model({
        "git_commit": "a7b9c3def456",
        "artifact_path": "/models/xgb_model_v1.pkl",
        "pipeline_version": "pipeline_v1.2",
        "python_env_hash": "py3.10-reqs-abc123",
        "training_data_id": "data_20251115",
        "training_data_range_start": "2025-11-10",
        "training_data_range_end": "2025-11-15",
        "training_row_count": 5000,
        "training_missing_values_json": {"age": 15, "income": 30},
        "metrics_json": {"ndcg@20": 0.142, "recall@20": 0.18}
    })
    print(f"✓ Model registered: {model_version}\n")

    # Step 2: Record predictions
    print("Step 2: Recording predictions...")
    request_ids = []
    for i in range(3):
        request_id = record_prediction({
            "userid": f"user_{i+1}",
            "model_version": model_version,
            "prediction": [f"movie_{j}" for j in range(5)],
            "input_data": {
                "user_history": [1, 2, 3],
                "user_features": {"age": 25, "country": "US"}
            },
            "extra_json": {
                "inference_latency_ms": 42 + i*10,
                "model_confidence": 0.85 + i*0.02
            }
        })
        request_ids.append(request_id)
        print(f"  ✓ Prediction {i+1} recorded: {request_id}")
    print()

    # Step 3: Trace a prediction
    print("Step 3: Tracing first prediction...")
    trace_result = trace_prediction(request_ids[0])
    if trace_result:
        print("Prediction Event:")
        pprint.pprint(trace_result["event"], width=60)
        print("\nModel Provenance:")
        pprint.pprint(trace_result["provenance"], width=60)
    else:
        print("✗ Prediction not found")
    print()

    # Step 4: Query predictions by model
    print("Step 4: Querying predictions by model...")
    predictions = get_predictions_by_model(model_version, limit=10)
    print(f"✓ Found {len(predictions)} predictions for model {model_version}\n")

    print("="*60)
    print("✓ Provenance system test completed successfully!")
    print("="*60 + "\n")

    return jsonify({"status": "provenance test completed"}), 200


if __name__ == "__main__":
    logger.info(f"Starting Movie Recommender API on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
