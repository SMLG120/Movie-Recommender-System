#!/usr/bin/env python3
"""
Movie Recommender API - Production Server Deployment
Flask API for serving movie recommendations on McGill CS servers
"""

from flask import Flask, Response
import sys
import os
import logging
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

# Set up import paths for Docker container
sys.path.append('/app/src')
sys.path.append('/app')

from inference import RecommenderEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define Metrics for Prometheus
REQ_LAT = Histogram(
    "api_request_seconds",
    "Request latency by endpoint (seconds)",
    ["endpoint"],
    buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
)
INF_LAT = Histogram(
    "model_inference_seconds",
    "Model inference latency (seconds)"
)
API_ERRORS = Counter(
    "api_errors_total",
    "5xx error count by endpoint",
    ["endpoint"]
)
INPROG = Gauge(
    "api_inprogress_requests",
    "In-progress requests by endpoint",
    ["endpoint"]
)
MODEL_READY = Gauge(
    "model_ready",
    "1 if model/engine is initialized, else 0"
)

app = Flask(__name__)
recommender_engine = None

def initialize_recommender():
    """Initialize the RecommenderEngine for server deployment"""
    global recommender_engine
    
    # Server deployment paths
    model_path = "/app/src/models/xgb_recommender.joblib"
    movies_path = "/app/data/raw_data/movies.csv"
    
    try:
        recommender_engine = RecommenderEngine(
            model_path=model_path,
            movies_file=movies_path,
            mode='dev'
        )
        logger.info("RecommenderEngine initialized successfully")
        MODEL_READY.set(1)
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RecommenderEngine: {e}")
        return False

# Initialize recommender on startup
initialize_recommender()

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id: int):
    """Main recommendation endpoint - returns comma-separated movie IDs"""
    ep = "/recommend"
    INPROG.labels(ep).inc()
    t0 = time.perf_counter()
    try:
        if recommender_engine is None:
            logger.warning("Engine not ready; returning fallback for user %s", user_id)
            fallback = ",".join(str(i) for i in range(1, 21))
            return Response(fallback, mimetype='text/plain'), 200

        # measure pure inference latency
        with INF_LAT.time():
            recommendations = recommender_engine.recommend(user_id, top_n=20)

        return Response(recommendations, mimetype='text/plain'), 200

    except Exception as e:
        logger.exception("Error for user %s: %s", user_id, e)
        API_ERRORS.labels(ep).inc()
        # still return a fallback to stay available
        fallback = ",".join(str(i) for i in range(1, 21))
        return Response(fallback, mimetype='text/plain'), 200

    finally:
        REQ_LAT.labels(ep).observe(time.perf_counter() - t0)
        INPROG.labels(ep).dec()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = "OK" if recommender_engine is not None else "Service Degraded"
    code = 200 if recommender_engine is not None else 503
    return Response(status, mimetype='text/plain'), code

@app.route('/metrics', methods=['GET'])
def metrics():
    """Prometheus scrape endpoint"""
    return generate_latest(REGISTRY), 200, {"Content-Type": "text/plain; version=0.0.4; charset=utf-8"}

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return Response("Movie Recommender API", mimetype='text/plain')

if __name__ == '__main__':
    logger.info("Starting Movie Recommender API on port 8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
