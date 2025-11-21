import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import joblib
import sys
import os

# usage: python train_model_logistic_regression.py Data/watch_time.csv model
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

# Prepare data
X_user = df['user_idx'].values
X_movie = df['movie_idx'].values
X = np.column_stack([X_user, X_movie])
y = df['watch_time'].values

# Create binary target: 1 if watch_time > 30, else 0
threshold = 30
y_binary = (y > threshold).astype(int)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Save model and mappings
model_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..', 'Data', out_model + '_watch_time_logistic.pkl'))
mappings_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..', 'Data', out_model + '_watch_time_mappings.pkl'))

joblib.dump(model, model_path)

mappings = {
    'user_map': user_map,
    'movie_map': movie_map,
    'user_ids': user_ids,
    'movie_ids': movie_ids,
    'threshold': threshold
}

with open(mappings_path, 'wb') as f:
    pickle.dump(mappings, f)

print('Saved Logistic Regression model to', model_path)
print('Saved mappings to', mappings_path)
