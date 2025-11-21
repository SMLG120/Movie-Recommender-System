import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Concatenate, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import sys
import os

# usage: python train_model.py Data/watch_time.csv model.pkl
ratings_csv = sys.argv[1]
out_model = sys.argv[2]

# Adjust path relative to script location (assuming ratings_csv starts with 'Data/')
ratings_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..', ratings_csv))
df = pd.read_csv(ratings_path)  # columns: user_id, movie_id, minutes_watched
df = df.rename(columns={'minutes_watched': 'watch_time'})

# Encode user_id and movie_id to integers
user_ids = df['user_id'].unique()
movie_ids = df['movie_id'].unique()

user_map = {uid: i for i, uid in enumerate(user_ids)}
movie_map = {mid: i for i, mid in enumerate(movie_ids)}

df['user_idx'] = df['user_id'].map(user_map)
df['movie_idx'] = df['movie_id'].map(movie_map)

num_users = len(user_ids)
num_movies = len(movie_ids)
emb_dim = 50  # embedding dimension

# Build MLP model
user_input = Input(shape=(1,))
movie_input = Input(shape=(1,))

user_emb = Embedding(num_users, emb_dim)(user_input)
movie_emb = Embedding(num_movies, emb_dim)(movie_input)

concat = Concatenate()([user_emb, movie_emb])
flatten = Flatten()(concat)

dense1 = Dense(128, activation='relu')(flatten)
dense2 = Dense(64, activation='relu')(dense1)
output = Dense(1)(dense2)  # predict watch_time

model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

# Prepare data
X_user = df['user_idx'].values
X_movie = df['movie_idx'].values
y = df['watch_time'].values

# Split train/test
X_user_train, X_user_test, X_movie_train, X_movie_test, y_train, y_test = train_test_split(
    X_user, X_movie, y, test_size=0.2, random_state=42)

# Scale watch_time to [0,1]
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

# Train
model.fit([X_user_train, X_movie_train], y_train_scaled, epochs=10, batch_size=32, validation_data=([X_user_test, X_movie_test], y_test_scaled))

# Save model and mappings
model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..', 'Data', out_model + '_watch_time_mlp.h5'))
mappings_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..', 'Data', out_model + '_watch_time_mappings.pkl'))

model.save(model_path)

mappings = {
    'user_map': user_map,
    'movie_map': movie_map,
    'user_ids': user_ids,
    'movie_ids': movie_ids,
    'scaler': scaler
}

with open(mappings_path, 'wb') as f:
    pickle.dump(mappings, f)

print('Saved MLP model to', model_path)
print('Saved mappings to', mappings_path)
