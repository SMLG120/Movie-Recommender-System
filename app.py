from flask import Flask, jsonify
import requests
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np
from kafka import KafkaConsumer
import json

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

class MovieRecommender:
    def __init__(self):
        self.user_matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.model = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        
    def update_model(self, ratings_dict):
        """Update model with current ratings"""
        # Convert ratings dict to matrix
        unique_users = sorted(ratings_dict.keys())
        all_movies = set()
        for user_ratings in ratings_dict.values():
            all_movies.update(user_ratings.keys())
        unique_movies = sorted(all_movies)
        
        # Create matrix
        matrix = np.zeros((len(unique_users), len(unique_movies)))
        for i, user_id in enumerate(unique_users):
            for j, movie_id in enumerate(unique_movies):
                matrix[i, j] = ratings_dict[user_id].get(movie_id, 0)
        
        self.user_matrix = matrix
        self.user_ids = unique_users
        self.movie_ids = unique_movies
        self.model.fit(matrix)
        
    def recommend(self, user_id, n_recommendations=20):
        """Get recommendations for user"""
        if user_id not in self.user_ids:
            return []
            
        user_idx = self.user_ids.index(user_id)
        _, indices = self.model.kneighbors([self.user_matrix[user_idx]])
        
        # Get recommendations from similar users
        similar_users = indices[0]
        recommendations = set()
        for sim_user_idx in similar_users:
            user_movies = np.where(self.user_matrix[sim_user_idx] > 0)[0]
            recommendations.update([self.movie_ids[idx] for idx in user_movies])
        
        return list(recommendations)[:n_recommendations]

# Initialize recommender
recommender = MovieRecommender()

@app.route('/recommend/<user_id>')
def recommend(user_id):
    # Update model with current ratings
    recommender.update_model(user_ratings)
    
    # Get recommendations
    recommendations = recommender.recommend(user_id)
    if not recommendations:
        # Fallback to popular movies
        recommendations = ["1", "2", "3", "4", "5"]
    
    return ','.join(recommendations[:20])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)