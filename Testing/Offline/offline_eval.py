import argparse
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import random
import joblib
import pickle
from multiprocessing import Pool
from functools import partial

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------- metrics ----------
def precision_at_k(recommended, ground_truth, k):
    return len(set(recommended[:k]).intersection(ground_truth)) / k

def recall_at_k(recommended, ground_truth, k):
    return len(set(recommended[:k]).intersection(ground_truth)) / max(1, len(ground_truth))

def hit_rate(recommended, ground_truth, k):
    return 1 if len(set(recommended[:k]).intersection(ground_truth)) > 0 else 0

def ndcg_at_k(recommended, ground_truth, k):
    """Fixed NDCG calculation with proper padding"""
    # Ensure recommended list is exactly length k by padding with empty strings
    recommended_k = (recommended[:k] + [''] * k)[:k]
    gains = [1 if rec in ground_truth and rec != '' else 0 for rec in recommended_k]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts)
    
    # Calculate IDCG
    n_relevant = min(len(ground_truth), k)
    ideal_gains = [1] * n_relevant + [0] * (k - n_relevant)
    idcg = np.sum(ideal_gains / discounts) if n_relevant > 0 else 0.0
    
    return dcg / idcg if idcg > 0 else 0.0

# Add new metrics for regression and classification
def regression_metrics(y_true, y_pred):
    """Compute regression metrics for watch time prediction"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    return {'rmse': rmse, 'mae': mae}

def classification_metrics(y_true, y_pred_proba):
    """Compute classification metrics for watch/no-watch prediction"""
    auc = roc_auc_score(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)
    return {'auc': auc, 'ap': ap}

# ---------- data load & checks ----------
def load_data(path):
    """Load and validate data with progress information"""
    print(f"Reading CSV file from {path}...")
    try:
        df = pd.read_csv(path)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Data validation
        required_cols = ['user_id', 'movie_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Handle timestamp
        if 'timestamp' not in df.columns:
            print("No timestamp column found, creating sequential timestamps...")
            df['timestamp'] = np.arange(len(df))
        
        # Clean and convert IDs
        print("Cleaning data...")
        df = df.dropna(subset=['user_id', 'movie_id'])
        df['user_id'] = df['user_id'].astype(str)
        df['movie_id'] = df['movie_id'].astype(str)
        
        print(f"Final dataset: {len(df)} interactions, "
              f"{df['user_id'].nunique()} users, "
              f"{df['movie_id'].nunique()} movies")
        return df
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def train_test_split_leave_one(df, min_train_interactions=1):
    """Split data with progress information"""
    print("Performing leave-one-out split...")
    print(f"Minimum training interactions per user: {min_train_interactions}")
    
    df = df.sort_values('timestamp')
    train_parts = []
    test_parts = []
    
    # Use tqdm for progress bar
    for uid, g in tqdm(df.groupby('user_id'), desc="Splitting users"):
        if len(g) <= min_train_interactions:
            train_parts.append(g)
            continue
        test_parts.append(g.tail(1))
        train_parts.append(g.iloc[:-1])
    
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=df.columns)
    
    # Ensure unique index spaces
    train_df.index = range(len(train_df))
    test_df.index = range(len(train_df), len(train_df) + len(test_df))
    
    print(f"Split complete: {len(train_df)} training interactions, {len(test_df)} test interactions")
    return train_df, test_df

def train_test_split_temporal(df, test_ratio=0.2):
    """Split based on timestamp to avoid future data leakage"""
    df = df.sort_values('timestamp')
    split_idx = int(len(df) * (1 - test_ratio))
    return df.iloc[:split_idx], df.iloc[split_idx:]

# ---------- simple recommenders ----------
def popular_recommender(train_df, top_k=1000):
    """Ensure we always return exactly top_k items"""
    pop = train_df.groupby('movie_id').size().sort_values(ascending=False)
    result = list(pop.index)[:top_k]
    # Pad with empty strings if needed
    return (result + [''] * top_k)[:top_k]

def build_item_item_model(train_df, max_items=5000):
    """Build item-item model with memory constraints"""
    # Get most popular items to limit matrix size
    pop_items = train_df.groupby('movie_id').size().nlargest(max_items).index
    train_df = train_df[train_df['movie_id'].isin(pop_items)]
    
    users = train_df['user_id'].unique()
    movies = pop_items
    uenc = LabelEncoder().fit(users)
    menc = LabelEncoder().fit(movies)
    
    # Use sparse matrix for efficiency
    from scipy.sparse import csr_matrix
    rows = uenc.transform(train_df['user_id'])
    cols = menc.transform(train_df['movie_id'])
    vals = train_df['rating'].fillna(1).astype(float).values
    mat = csr_matrix((vals, (rows, cols)), shape=(len(users), len(movies)))
    
    # Keep matrix in sparse format for memory efficiency
    return {
        'users': list(uenc.classes_),
        'movies': list(menc.classes_),
        'uenc': uenc,
        'menc': menc,
        'sim': cosine_similarity(mat.T),  # This still needs to be dense for cosine similarity
        'mat': mat  # Keep as sparse matrix
    }

def score_item_item(model, user_id, candidate_movies, top_k=20):
    """Ensure we always return exactly top_k items"""
    try:
        uidx = model['uenc'].transform([str(user_id)])[0]
    except ValueError:
        return [''] * top_k  # Return empty strings instead of empty list
    
    # Handle sparse matrix correctly
    user_vec = model['mat'][uidx].toarray().flatten()
    scores = model['sim'].dot(user_vec)
    scored = []
    
    for m in candidate_movies:
        try:
            midx = model['menc'].transform([str(m)])[0]
            s = scores[midx]
        except ValueError:
            s = -np.inf
        scored.append((m, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Ensure we return exactly top_k items, pad with empty strings if needed
    result = [str(x[0]) for x in scored[:top_k]]
    return (result + [''] * top_k)[:top_k]

# ---------- evaluation loop ----------
def evaluate_batch(args):
    """Ensure pop_ranked is exactly length top_k"""
    batch_df, train_df, pop_list, item_model, top_k, neg_samples = args
    results = {'pop': [], 'item_item': []}
    all_train_movies = train_df['movie_id'].astype(str).unique()
    
    # Pre-compute user training movies once
    user_train_movies = {
        uid: set(train_df[train_df['user_id'] == uid]['movie_id'].astype(str))
        for uid in batch_df['user_id'].unique()
    }
    
    for _, row in batch_df.iterrows():
        uid = str(row['user_id'])
        true_mid = str(row['movie_id'])
        
        # Use pre-computed user movies
        seen_movies = user_train_movies.get(uid, set())
        negatives = np.random.choice(
            [m for m in all_train_movies if m not in seen_movies], 
            size=min(neg_samples, len(all_train_movies)-len(seen_movies)), 
            replace=False
        ).tolist()
        
        candidates = negatives + [true_mid]
        # Ensure pop_ranked is exactly length top_k
        pop_ranked = [m for m in pop_list if m in candidates]
        pop_ranked = (pop_ranked + [''] * top_k)[:top_k]
        
        ii_ranked = score_item_item(item_model, uid, candidates, top_k=top_k)
        
        gt = {true_mid}
        results['pop'].append({
            'hit': hit_rate(pop_ranked, gt, top_k),
            'precision': precision_at_k(pop_ranked, gt, top_k),
            'recall': recall_at_k(pop_ranked, gt, top_k),
            'ndcg': ndcg_at_k(pop_ranked, gt, top_k)
        })
        results['item_item'].append({
            'hit': hit_rate(ii_ranked, gt, top_k),
            'precision': precision_at_k(ii_ranked, gt, top_k),
            'recall': recall_at_k(ii_ranked, gt, top_k),
            'ndcg': ndcg_at_k(ii_ranked, gt, top_k)
        })
    
    return results

def evaluate(train_df, test_df, top_k=20, neg_samples=500, n_jobs=4, batch_size=100, parallel=True):
    """Memory-efficient parallel evaluation"""
    pop_list = popular_recommender(train_df, top_k=2000)
    
    # Build item-item model with reduced size
    item_model = build_item_item_model(train_df, max_items=5000)
    
    # Process in smaller chunks
    chunk_size = min(batch_size, 1000)
    n_chunks = (len(test_df) + chunk_size - 1) // chunk_size
    
    all_results = {'pop': [], 'item_item': []}
    
    for chunk_idx in tqdm(range(n_chunks), desc='Processing chunks'):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(test_df))
        chunk_df = test_df.iloc[start_idx:end_idx]
        
        # Split chunk into batches for parallel processing
        n_batches = max(1, len(chunk_df) // (chunk_size // n_jobs))
        batches = np.array_split(chunk_df, n_batches)
        
        eval_args = [
            (batch, train_df, pop_list, item_model, top_k, neg_samples)
            for batch in batches
        ]
        
        # Process batches in parallel
        if parallel:
            with Pool(n_jobs) as pool:
                batch_results = list(pool.imap(evaluate_batch, eval_args))
        else:
            batch_results = [evaluate_batch(a) for a in eval_args]

        
        # Combine results from batches
        for batch_res in batch_results:
            all_results['pop'].extend(batch_res['pop'])
            all_results['item_item'].extend(batch_res['item_item'])
    
    # Aggregate means
    agg = {}
    for name, recs in all_results.items():
        dfm = pd.DataFrame(recs)
        agg[name] = dfm.mean().to_dict()
    
    return agg

def evaluate_user_segments(train_df, test_df, segments, top_k=20, parallel=True):
    """Evaluate performance across different user segments"""
    results = {}
    for segment_name, segment_filter in segments.items():
        segment_test = test_df[segment_filter(test_df)]
        if len(segment_test) == 0:
            continue
        res = evaluate(train_df, segment_test, top_k=top_k, parallel=parallel)
        results[segment_name] = res
    return results

def analyze_predictions(train_df, test_df, predictions, top_k=20):
    """Analyze prediction quality and biases"""
    metrics = {}
    
    # Basic metrics
    metrics.update(evaluate(train_df, test_df, top_k=top_k))
    
    # Coverage analysis
    all_items = set(train_df['movie_id'].unique())
    recommended_items = set()
    for user_recs in predictions.values():
        recommended_items.update(user_recs[:top_k])
    metrics['item_coverage'] = len(recommended_items) / len(all_items)
    
    # Popularity bias
    item_pop = train_df['movie_id'].value_counts()
    avg_pop = np.mean([item_pop.get(item, 0) for items in predictions.values() 
                       for item in items[:top_k]])
    metrics['avg_popularity'] = avg_pop
    
    return metrics

def create_user_segments(df):
    """Create user segments based on interaction count"""
    user_counts = df.groupby('user_id').size()
    # Define new users as those with <= 5 interactions
    new_users = set(user_counts[user_counts <= 5].index)
    regular_users = set(user_counts[user_counts > 5].index)
    return {
        'new_users': lambda x: x['user_id'].isin(new_users),
        'regular_users': lambda x: x['user_id'].isin(regular_users)
    }

# ---------- main ----------
def main(args):
    # Add data validation and progress reporting
    if args.dev:
        print("Running in development mode with sample data...")
        df = pd.DataFrame({
            'user_id': ['u1', 'u1', 'u2', 'u2', 'u2', 'u3', 'u3'],
            'movie_id': ['m1', 'm2', 'm1', 'm3', 'm4', 'm2', 'm3'],
            'timestamp': [1, 2, 3, 4, 5, 6, 7],
            'minutes_watched': [15, 45, 30, 60, 20, 10, 50]
        })
    else:
        # Try to find the input file in multiple locations
        possible_paths = [
            args.input,
            os.path.join(os.path.dirname(__file__), '..', '..', args.input),
            os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'training_data.csv'),
            os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'training_data.csv'),
            os.path.join(os.path.dirname(__file__), 'sample_training.csv')
        ]
        
        print("Searching for input file...")
        input_path = None
        for path in possible_paths:
            print(f"Trying path: {path}")
            if os.path.exists(path):
                input_path = path
                break
        
        if input_path is None:
            print("No input file found. Available paths were:")
            for path in possible_paths:
                print(f" - {path}")
            raise FileNotFoundError("No input file found")
        
        print(f"\nFound input file: {input_path}")
        df = load_data(input_path)
        
        if args.sample > 0:
            print(f"\nSampling {args.sample} users...")
            sampled_users = np.random.choice(df['user_id'].unique(), size=args.sample, replace=False)
            df = df[df['user_id'].isin(sampled_users)]
            print(f"Sampled data size: {len(df)} interactions")

    print("\nPreparing evaluation...")
    train_df, test_df = train_test_split_leave_one(df)
    print(f"Users in train: {train_df['user_id'].nunique()}, test interactions: {len(test_df)}")
    
    # Use smaller batch size and fewer negative samples by default
    print("\nStarting evaluation...")
    print(f"Parameters: top_k={args.k}, neg_samples={min(args.neg, 100)}, jobs={args.jobs}, batch_size={min(args.batch_size, 1000)}")
    
    res = evaluate(
        train_df, 
        test_df, 
        top_k=args.k, 
        neg_samples=min(args.neg, 100),  # Limit negative samples
        n_jobs=args.jobs,
        batch_size=min(args.batch_size, 1000),  # Limit batch size
        parallel=not args.no_parallel  # Allow disabling parallel processing
    )
    
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:  # Only create directory if output path has a directory component
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            'meta': {
                'input': str(input_path) if not args.dev else 'sample_data',
                'k': args.k,
                'neg_samples': args.neg,
                'n_jobs': args.jobs,
                'batch_size': args.batch_size,
                'sample_size': args.sample if not args.dev else 0
            },
            'results': res
        }, f, indent=2)
    print(f"\nSaved results to {args.output}")
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='data/training_data.csv')
    p.add_argument('--output', default='results/offline_eval_results.json')
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--neg', type=int, default=500)
    p.add_argument('--jobs', type=int, default=4, help='Number of parallel jobs')
    p.add_argument('--batch_size', type=int, default=100, help='Batch size for parallel processing')
    p.add_argument('--sample', type=int, default=0, help='Number of users to sample (0=all)')
    p.add_argument('--dev', action='store_true', help='Run in development mode with sample data')
    p.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    args = p.parse_args()
    main(args)