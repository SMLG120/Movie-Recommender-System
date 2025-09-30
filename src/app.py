#!/usr/bin/env python3
"""
Movie Recommender API - Production Server Deployment
Flask API for serving movie recommendations on McGill CS servers
"""

from flask import Flask, Response
import sys
import os
import logging

# Set up import paths for Docker container
sys.path.append('/app/src')
sys.path.append('/app')

from inference import RecommenderEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RecommenderEngine: {e}")
        return False

# Initialize recommender on startup
initialize_recommender()

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    """Main recommendation endpoint - returns comma-separated movie IDs"""
    try:
        if recommender_engine is None:
            # Fallback recommendations
            fallback = ",".join([str(i) for i in range(1, 21)])
            return Response(fallback, mimetype='text/plain')
        
        recommendations = recommender_engine.recommend(user_id, top_n=20)
        return Response(recommendations, mimetype='text/plain')
    
    except Exception as e:
        logger.error(f"Error for user {user_id}: {e}")
        fallback = ",".join([str(i) for i in range(1, 21)])
        return Response(fallback, mimetype='text/plain')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = "OK" if recommender_engine is not None else "Service Degraded"
    code = 200 if recommender_engine is not None else 503
    return Response(status, mimetype='text/plain'), code

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return Response("Movie Recommender API", mimetype='text/plain')

if __name__ == '__main__':
    logger.info("🚀 Starting Movie Recommender API on port 8080")
    app.run(host='0.0.0.0', port=8080, debug=False)
