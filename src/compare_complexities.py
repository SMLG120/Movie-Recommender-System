import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Concatenate, Flatten, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle
import joblib
import sys
import os
import time
import psutil
import tempfile

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # MB

def measure_training(model_func, *args, **kwargs):
    start_time = time.time()
    start_mem = get_memory_usage()
    model = model_func(*args, **kwargs)
    end_time = time.time()
    end_mem = get_memory_usage()
    training_time = end_time - start_time
    peak_mem = max(start_mem, end_mem)  # Approximate peak
    return model, training_time, peak_mem

def measure_inference(model, X_test, is_mlp=False):
    times = []
    for _ in range(10):
        start = time.time()
        if is_mlp:
            model.predict(X_test, verbose=0)
        else:
            model.predict(X_test)
        end = time.time()
        times.append(end - start)
    avg_time = np.mean(times)
    return avg_time

def get_model_size(model, model_type):
    import uuid
    temp_dir = tempfile.gettempdir()
    if model_type == 'xgb':
        temp_file = os.path.join(temp_dir, f"temp_xgb_{uuid.uuid4().hex}.pkl")
        joblib.dump(model, temp_file)
    elif model_type == 'mlp':
        temp_file = os.path.join(temp_dir, f"temp_mlp_{uuid.uuid4().hex}.h5")
        model.save(temp_file)
    size = os.path.getsize(temp_file) / (1024 * 1024)  # MB
    os.remove(temp_file)
    return size

def get_param_count(model, model_type):
    if model_type == 'xgb':
        return model.n_estimators
    elif model_type == 'mlp':
        return model.count_params()

# Usage: python compare_complexities.py Data/watch_time.csv
if len(sys.argv) != 2:
    print("Usage: python src/compare_complexities.py Data/watch_time.csv")
    sys.exit(1)

ratings_csv = sys.argv[1]
ratings_path = os.path.join(os.path.dirname(__file__), '..', ratings_csv)
df = pd.read_csv(ratings_path)
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
emb_dim = 50

# Prepare data
X_user = df['user_idx'].values
X_movie = df['movie_idx'].values
X_xgb = np.column_stack([X_user, X_movie])
y = df['watch_time'].values

# Split train/test
X_user_train, X_user_test, X_movie_train, X_movie_test, y_train, y_test = train_test_split(
    X_user, X_movie, y, test_size=0.2, random_state=42)
X_xgb_train, X_xgb_test, _, _ = train_test_split(X_xgb, y, test_size=0.2, random_state=42)

# Scale y
scaler = MinMaxScaler()
y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

# Train XGBoost
def train_xgb(X_train, y_train):
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

xgb_model, xgb_train_time, xgb_peak_mem = measure_training(train_xgb, X_xgb_train, y_train_scaled)

# Train MLP
def train_mlp(X_user_train, X_movie_train, y_train):
    user_input = Input(shape=(1,))
    movie_input = Input(shape=(1,))
    user_emb = Embedding(num_users, emb_dim)(user_input)
    movie_emb = Embedding(num_movies, emb_dim)(movie_input)
    concat = Concatenate()([user_emb, movie_emb])
    flatten = Flatten()(concat)
    dense1 = Dense(128, activation='relu')(flatten)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(1)(dense2)
    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(optimizer='adam', loss='mse')
    model.fit([X_user_train, X_movie_train], y_train, epochs=1, batch_size=32, verbose=0)
    return model

mlp_model, mlp_train_time, mlp_peak_mem = measure_training(train_mlp, X_user_train, X_movie_train, y_train_scaled)

# Measure inference
xgb_inf_time = measure_inference(xgb_model, X_xgb_test, is_mlp=False)
mlp_inf_time = measure_inference(mlp_model, [X_user_test, X_movie_test], is_mlp=True)

# Model sizes and params
xgb_size = get_model_size(xgb_model, 'xgb')
mlp_size = get_model_size(mlp_model, 'mlp')
xgb_params = get_param_count(xgb_model, 'xgb')
mlp_params = get_param_count(mlp_model, 'mlp')

# Print results
print("Complexity Comparison: XGBoost vs MLP")
print("=" * 50)
print(f"{'Metric':<20} {'XGBoost':<15} {'MLP':<15}")
print("-" * 50)
print(f"{'Training Time (s)':<20} {xgb_train_time:<15.2f} {mlp_train_time:<15.2f}")
print(f"{'Inference Time (s)':<20} {xgb_inf_time:<15.2f} {mlp_inf_time:<15.2f}")
print(f"{'Peak Memory (MB)':<20} {xgb_peak_mem:<15.2f} {mlp_peak_mem:<15.2f}")
print(f"{'Model Size (MB)':<20} {xgb_size:<15.2f} {mlp_size:<15.2f}")
print(f"{'Param Count':<20} {xgb_params:<15} {mlp_params:<15}")
