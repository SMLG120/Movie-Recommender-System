import argparse
import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import xgboost as xgb
# Import trainer.py and run for offline evaluation
from inference import recommender
# create a testset 
# run the cf trainer to get test embeddings
# run it through the feature builder, pass the test embeddins to the feature builder also 
# load eaxisting model
# run prediction and evaluation  

# train
# start_train = time.time()
# self.pipeline.fit(X_train, y_train)
# train_time = time.time() - start_train

# preds = self.pipeline.predict(X_test)
# infer_time = (time.time() - start_train) / max(len(X_test), 1)

# mse = mean_squared_error(y_test, preds)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, preds)
# r2 = r2_score(y_test, preds)


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------
# Data loading and preprocessing
# ---------------------------
def load_data(path="data/training_data.csv"):
    """Load and validate data"""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    required_cols = ["user_id", "movie_id", "rating"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    df = df.dropna(subset=required_cols)
    df["user_id"] = df["user_id"].astype(str)
    df["movie_id"] = df["movie_id"].astype(str)
    return df


def train_test_split_stratified(df, test_size=0.2):
    """Stratified split ensuring all users appear in training"""
    # Sort by timestamp if available
    df = df.sort_values("timestamp" if "timestamp" in df.columns else "rating")
    
    # Ensure each user has at least one interaction in training
    train_df = df.groupby('user_id').head(1)
    remaining_df = df[~df.index.isin(train_df.index)]
    
    # Split remaining data
    n_test = int(len(df) * test_size)
    test_df = remaining_df.tail(n_test)
    train_df = pd.concat([train_df, remaining_df.head(-n_test)])
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# ---------------------------
# XGBoost
# ---------------------------
def evaluate_regression_metrics(y_true, y_pred):
    """Calculate regression metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # R² calculation
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def encode_features(train_df, test_df):
    """Encode features ensuring all labels are in training set"""
    # Get all unique values
    all_users = sorted(set(train_df['user_id'].unique()) | set(test_df['user_id'].unique()))
    all_movies = sorted(set(train_df['movie_id'].unique()) | set(test_df['movie_id'].unique()))
    
    # Create label encoders
    user_encoder = LabelEncoder().fit(all_users)
    movie_encoder = LabelEncoder().fit(all_movies)
    
    # Transform data
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df['user_idx'] = user_encoder.transform(train_df['user_id'])
    train_df['movie_idx'] = movie_encoder.transform(train_df['movie_id'])
    test_df['user_idx'] = user_encoder.transform(test_df['user_id'])
    test_df['movie_idx'] = movie_encoder.transform(test_df['movie_id'])
    
    return train_df, test_df


def train_and_evaluate_xgboost(train_df, test_df):
    """Train and evaluate XGBoost model"""
    # Encode features
    train_df, test_df = encode_features(train_df, test_df)
    
    # Train XGBoost
    print("Training XGBoost model...")
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=RANDOM_SEED
    )
    
    X_train = train_df[['user_idx', 'movie_idx']].values
    y_train = train_df['rating'].values
    model.fit(X_train, y_train)
    
    # Evaluate
    X_test = test_df[['user_idx', 'movie_idx']].values
    y_test = test_df['rating'].values
    y_pred = model.predict(X_test)
    
    metrics = evaluate_regression_metrics(y_test, y_pred)
    return metrics


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_data.csv")
    parser.add_argument("--sample", type=int, default=0, help="Number of users to sample (0=all)")
    args = parser.parse_args()
    
    try:
        # Load data
        df = load_data(args.input)
        if args.sample > 0:
            users = np.random.choice(df['user_id'].unique(), size=args.sample, replace=False)
            df = df[df['user_id'].isin(users)]
        
        # Split data
        train_df, test_df = train_test_split_stratified(df)
        print(f"Training set: {len(train_df)} rows, {train_df['user_id'].nunique()} users")
        print(f"Test set: {len(test_df)} rows, {test_df['user_id'].nunique()} users")
        
        # Train and evaluate
        metrics = train_and_evaluate_xgboost(train_df, test_df)
        
        print("\nXGBoost Regression Metrics:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
