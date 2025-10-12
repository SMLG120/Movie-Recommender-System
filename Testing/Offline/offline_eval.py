# ...new file...
import argparse
import json
import math
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os
import random

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
    gains = [1 if rec in ground_truth else 0 for rec in recommended[:k]]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(np.array(gains) / discounts)
    ideal_gains = [1] * min(len(ground_truth), k)
    idcg = np.sum(np.array(ideal_gains) / discounts[: len(ideal_gains)]) if len(ideal_gains) > 0 else 0.0
    return dcg / idcg if idcg > 0 else 0.0

# ---------- data load & checks ----------
def load_data(path):
    df = pd.read_csv(path)
    # expected: user_id, movie_id, rating (optional), timestamp (optional)
    if 'timestamp' not in df.columns:
        # preserve file order as pseudo-time
        df['timestamp'] = np.arange(len(df))
    df = df.dropna(subset=['user_id', 'movie_id'])
    # ensure string ids
    df['user_id'] = df['user_id'].astype(str)
    df['movie_id'] = df['movie_id'].astype(str)
    return df

def train_test_split_leave_one(df, min_train_interactions=1):
    df = df.sort_values('timestamp')
    train_parts = []
    test_parts = []
    for uid, g in df.groupby('user_id'):
        if len(g) <= min_train_interactions:
            train_parts.append(g)
            continue
        test_parts.append(g.tail(1))
        train_parts.append(g.iloc[:-1])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df, test_df

# ---------- simple recommenders ----------
def popular_recommender(train_df, top_k=1000):
    pop = train_df.groupby('movie_id').size().sort_values(ascending=False)
    return list(pop.index)[:top_k]

def build_item_item_model(train_df):
    users = train_df['user_id'].unique()
    movies = train_df['movie_id'].unique()
    uenc = LabelEncoder().fit(users)
    menc = LabelEncoder().fit(movies)
    rows = uenc.transform(train_df['user_id'])
    cols = menc.transform(train_df['movie_id'])
    vals = train_df['rating'].fillna(1).astype(float).values
    mat = np.zeros((len(users), len(movies)), dtype=float)
    for r, c, v in zip(rows, cols, vals):
        mat[r, c] = v
    sim = cosine_similarity(mat.T)  # movies x movies
    model = {'sim': sim, 'users': list(uenc.classes_), 'movies': list(menc.classes_), 'mat': mat, 'uenc': uenc, 'menc': menc}
    return model

def score_item_item(model, user_id, candidate_movies, top_k=20):
    try:
        uidx = model['uenc'].transform([str(user_id)])[0]
    except ValueError:
        return []
    user_vec = model['mat'][uidx]
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
    return [str(x[0]) for x in scored[:top_k]]

# ---------- evaluation loop ----------
def evaluate(train_df, test_df, top_k=20, neg_samples=500):
    results = {}
    all_train_movies = train_df['movie_id'].astype(str).unique().tolist()
    pop_list = popular_recommender(train_df, top_k=2000)
    item_model = build_item_item_model(train_df)

    metrics_store = {'pop': [], 'item_item': []}
    # For each test interaction (one per user), sample negatives then evaluate
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Evaluate'):
        uid = str(row['user_id'])
        true_mid = str(row['movie_id'])
        user_train_movies = set(train_df[train_df['user_id'] == uid]['movie_id'].astype(str).tolist())
        negatives = list(set(all_train_movies) - user_train_movies)
        if len(negatives) > neg_samples:
            negatives = random.sample(negatives, neg_samples)
        candidates = negatives + [true_mid]
        # popularity ranking restricted to candidates (preserve ordering)
        pop_ranked = [m for m in pop_list if m in candidates]
        # pad if needed
        if len(pop_ranked) < top_k:
            extras = [m for m in candidates if m not in pop_ranked]
            pop_ranked += extras[: max(0, top_k - len(pop_ranked))]
        pop_ranked = pop_ranked[:top_k]
        # item-item ranking
        ii_ranked = score_item_item(item_model, uid, candidates, top_k=top_k)
        # metrics
        gt = {true_mid}
        metrics_store['pop'].append({
            'hit': hit_rate(pop_ranked, gt, top_k),
            'precision': precision_at_k(pop_ranked, gt, top_k),
            'recall': recall_at_k(pop_ranked, gt, top_k),
            'ndcg': ndcg_at_k(pop_ranked, gt, top_k)
        })
        metrics_store['item_item'].append({
            'hit': hit_rate(ii_ranked, gt, top_k),
            'precision': precision_at_k(ii_ranked, gt, top_k),
            'recall': recall_at_k(ii_ranked, gt, top_k),
            'ndcg': ndcg_at_k(ii_ranked, gt, top_k)
        })

    # aggregate means
    agg = {}
    for name, recs in metrics_store.items():
        dfm = pd.DataFrame(recs)
        agg[name] = dfm.mean().to_dict()
    return agg

# ---------- main ----------
def main(args):
    assert os.path.exists(args.input), f"Input not found: {args.input}"
    df = load_data(args.input)
    train_df, test_df = train_test_split_leave_one(df)
    print(f"Users in train: {train_df['user_id'].nunique()}, test interactions: {len(test_df)}")
    res = evaluate(train_df, test_df, top_k=args.k, neg_samples=args.neg)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({'meta': {'input': args.input, 'k': args.k, 'neg_samples': args.neg}, 'results': res}, f, indent=2)
    print(f"Saved results to {args.output}")
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default='Data/training_data.csv')
    p.add_argument('--output', default='Testing/Offline/offline_eval_results.json')
    p.add_argument('--k', type=int, default=20)
    p.add_argument('--neg', type=int, default=500)
    args = p.parse_args()
    main(args)