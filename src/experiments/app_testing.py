from flask import Flask, jsonify
import requests
import pandas as pd
import numpy as np
from kafka import KafkaConsumer
import json
import joblib
import pickle
import os

app = Flask(__name__)

# Cache for storing movie and user data
movie_cache = {}
user_cache = {}

def fetch_movie_data(movie_id):
    """Fetch movie data from the API"""
    if movie_id not in movie_cache:
        response = requests.get(f'http://fall2025-comp585.cs.mcgill.ca:8080/movie/{movie_id}')
        if response.status_code == 200:
            movie_cache[movie_id] = response.json()
    return movie_cache.get(movie_id)

def fetch_user_data(user_id):
    """Fetch user data from the API"""
    if user_id not in user_cache:
        response = requests.get(f'http://fall2025-comp585.cs.mcgill.ca:8080/user/{user_id}')
        if response.status_code == 200:
            user_cache[user_id] = response.json()
    return user_cache.get(user_id)

# Load XGBoost model and mappings
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data', 'model_watch_time_xgb.pkl')
MAPPINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'Data', 'model_watch_time_mappings.pkl')

try:
    xgb_model = joblib.load(MODEL_PATH)
    with open(MAPPINGS_PATH, 'rb') as f:
        mappings = pickle.load(f)
    user_map = mappings['user_map']
    movie_map = mappings['movie_map']
    all_movie_ids = mappings['movie_ids']
    scaler = mappings['scaler']
    print('Loaded watch_time XGBoost model and mappings')
except Exception as e:
    print(f'Error loading model: {e}')
    xgb_model = None
    user_map = {}
    movie_map = {}
    all_movie_ids = []
    scaler = None

def setup_kafka_consumer():
    """Setup Kafka consumer for movie ratings"""
    consumer = KafkaConsumer(
        'movielog6',  # Assuming team 6
        bootstrap_servers=['fall2025-comp585.cs.mcgill.ca:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='team6-recommender'
    )
    return consumer

# Add user ratings storage
user_ratings = {}  # Format: {user_id: {movie_id: rating}}

def process_kafka_messages():
    """Process Kafka messages to collect ratings"""
    consumer = setup_kafka_consumer()
    for message in consumer:
        try:
            log_entry = message.value.decode('utf-8')
            if 'GET /rate/' in log_entry:
                # Parse rating entry
                parts = log_entry.split(',')
                user_id = parts[1]
                rating_part = parts[2]
                movie_id = rating_part.split('=')[0].split('/')[-1]
                rating = int(rating_part.split('=')[1])
                
                # Store rating
                if user_id not in user_ratings:
                    user_ratings[user_id] = {}
                user_ratings[user_id][movie_id] = rating
        except Exception as e:
            print(f"Error processing message: {e}")

class XGBoostRecommender:
    def __init__(self, model, user_map, movie_map, all_movie_ids, scaler):
        self.model = model
        self.user_map = user_map
        self.movie_map = movie_map
        self.all_movie_ids = all_movie_ids
        self.scaler = scaler
        
    def recommend(self, user_id, n_recommendations=20):
        """Get recommendations for user using XGBoost predictions"""
        if self.model is None or user_id not in self.user_map:
            # Fallback to popular movies
            return ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"] * 2

        user_idx = self.user_map[user_id]
        rated_movies = user_ratings.get(user_id, {})

        # Predict watch_time for all movies
        predictions = []
        for movie_id in self.all_movie_ids:
            if movie_id not in rated_movies:
                movie_idx = self.movie_map.get(movie_id)
                if movie_idx is not None:
                    pred_watch_time_scaled = self.model.predict(np.array([[user_idx, movie_idx]]))[0]
                    # Inverse scale to minutes for interpretability (optional for ranking)
                    if self.scaler:
                        pred_watch_time = self.scaler.inverse_transform([[pred_watch_time_scaled]])[0][0]
                    else:
                        pred_watch_time = pred_watch_time_scaled
                    predictions.append((movie_id, pred_watch_time))

        # Sort by predicted watch_time descending
        predictions.sort(key=lambda x: x[1], reverse=True)
        recommended_ids = [pred[0] for pred in predictions[:n_recommendations]]

        return recommended_ids

# Initialize recommender
recommender = XGBoostRecommender(xgb_model, user_map, movie_map, all_movie_ids, scaler)

@app.route('/recommend/<user_id>')
def recommend(user_id):
    # Get recommendations
    recommendations = recommender.recommend(user_id)
    
    return ','.join(recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
