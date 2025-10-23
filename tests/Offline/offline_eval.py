import argparse
import json
import os
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from tqdm import tqdm
import xgboost as xgb

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ---------------------------
# Metrics
# ---------------------------
def precision_at_k(recommended, ground_truth, k):
    return len(set(recommended[:k]).intersection(ground_truth)) / k


def recall_at_k(recommended, ground_truth, k):
    return len(set(recommended[:k]).intersection(ground_truth)) / max(1, len(ground_truth))


def hit_rate(recommended, ground_truth, k):
    return 1 if len(set(recommended[:k]).intersection(ground_truth)) > 0 else 0


def ndcg_at_k(recommended, ground_truth, k):
    recommended_k = (recommended[:k] + [""] * k)[:k]
    gains = [1 if rec in ground_truth and rec != "" else 0 for rec in recommended_k]
    discounts = np.log2(np.arange(2, k + 2))
    dcg = np.sum(gains / discounts)
    n_relevant = min(len(ground_truth), k)
    ideal_gains = [1] * n_relevant + [0] * (k - n_relevant)
    idcg = np.sum(ideal_gains / discounts) if n_relevant > 0 else 0.0
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------
# Data loading and preprocessing
# ---------------------------
def load_data(path="data/training_data.csv"):
    df = pd.read_csv(path)
    required_cols = ["user_id", "movie_id", "rating"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df = df.dropna(subset=required_cols)
    df["user_id"] = df["user_id"].astype(str)
    df["movie_id"] = df["movie_id"].astype(str)
    if "timestamp" not in df.columns:
        df["timestamp"] = np.arange(len(df))
    return df


def feature_encode(df):
    user_enc = LabelEncoder().fit(df["user_id"])
    movie_enc = LabelEncoder().fit(df["movie_id"])
    df["user_idx"] = user_enc.transform(df["user_id"])
    df["movie_idx"] = movie_enc.transform(df["movie_id"])
    return df, user_enc, movie_enc


def leave_one_out_split(df):
    df = df.sort_values("timestamp")
    train_parts, test_parts = [], []
    for uid, g in df.groupby("user_id"):
        if len(g) == 1:
            train_parts.append(g)
        else:
            train_parts.append(g.iloc[:-1])
            test_parts.append(g.iloc[-1:])
    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True) if test_parts else pd.DataFrame(columns=df.columns)
    return train_df, test_df


# ---------------------------
# Simple recommenders
# ---------------------------
def popular_recommender(train_df, top_k=1000):
    pop = train_df.groupby("movie_id").size().sort_values(ascending=False)
    result = list(pop.index)[:top_k]
    return (result + [""] * top_k)[:top_k]


def build_item_item_model(train_df, max_items=5000):
    pop_items = train_df.groupby("movie_id").size().nlargest(max_items).index
    train_df = train_df[train_df["movie_id"].isin(pop_items)]
    users = train_df["user_id"].unique()
    movies = pop_items
    uenc = LabelEncoder().fit(users)
    menc = LabelEncoder().fit(movies)
    rows = uenc.transform(train_df["user_id"])
    cols = menc.transform(train_df["movie_id"])
    vals = train_df["rating"].fillna(1).astype(float).values
    mat = csr_matrix((vals, (rows, cols)), shape=(len(users), len(movies)))
    sim = cosine_similarity(mat.T)
    return {"users": list(uenc.classes_), "movies": list(menc.classes_), "uenc": uenc, "menc": menc, "mat": mat, "sim": sim}


def score_item_item(model, user_id, candidate_movies, top_k=20):
    try:
        uidx = model["uenc"].transform([str(user_id)])[0]
    except ValueError:
        return [""] * top_k
    user_vec = model["mat"][uidx].toarray().flatten()
    scores = model["sim"].dot(user_vec)
    scored = []
    for m in candidate_movies:
        try:
            midx = model["menc"].transform([str(m)])[0]
            s = scores[midx]
        except ValueError:
            s = -np.inf
        scored.append((m, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    result = [str(x[0]) for x in scored[:top_k]]
    return (result + [""] * top_k)[:top_k]


# ---------------------------
# XGBoost
# ---------------------------
def train_xgboost(train_df):
    X_train = train_df[["user_idx", "movie_idx"]].values
    y_train = train_df["rating"].values
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)
    return model


def rank_xgboost(xgb_model, user_enc, movie_enc, train_df, top_k=20, all_movies=None):
    if all_movies is None:
        all_movies = train_df["movie_id"].unique()
    user_ranked = {}
    for uid in tqdm(train_df["user_id"].unique(), desc="Ranking XGBoost"):
        seen_movies = set(train_df[train_df["user_id"] == uid]["movie_id"])
        candidates = [m for m in all_movies if m not in seen_movies]
        if not candidates:
            user_ranked[uid] = [""] * top_k
            continue
        X_candidates = pd.DataFrame({
            "user_idx": [user_enc.transform([uid])[0]] * len(candidates),
            "movie_idx": movie_enc.transform(candidates)
        })
        preds = xgb_model.predict(X_candidates)
        ranked = [m for _, m in sorted(zip(preds, candidates), reverse=True)]
        user_ranked[uid] = (ranked + [""] * top_k)[:top_k]
    return user_ranked


def evaluate_xgboost_ranking(user_ranked, test_df, top_k=20):
    results = []
    for _, row in test_df.iterrows():
        uid = row["user_id"]
        true_movie = row["movie_id"]
        recs = user_ranked.get(uid, [""] * top_k)
        gt = {true_movie}
        results.append({
            "hit": hit_rate(recs, gt, top_k),
            "precision": precision_at_k(recs, gt, top_k),
            "recall": recall_at_k(recs, gt, top_k),
            "ndcg": ndcg_at_k(recs, gt, top_k)
        })
    dfm = pd.DataFrame(results)
    return dfm.mean().to_dict()


# ---------------------------
# Evaluation
# ---------------------------
def evaluate_model(train_df, test_df, user_enc, movie_enc, top_k=20):
    agg = {}

    # Popular
    pop_list = popular_recommender(train_df, top_k=2000)
    results_pop = []
    for _, row in test_df.iterrows():
        gt = {row["movie_id"]}
        recs = (pop_list[:top_k] + [""] * top_k)[:top_k]
        results_pop.append({
            "hit": hit_rate(recs, gt, top_k),
            "precision": precision_at_k(recs, gt, top_k),
            "recall": recall_at_k(recs, gt, top_k),
            "ndcg": ndcg_at_k(recs, gt, top_k),
        })
    agg["pop"] = pd.DataFrame(results_pop).mean().to_dict()

    # Item-Item
    item_model = build_item_item_model(train_df)
    results_ii = []
    for _, row in test_df.iterrows():
        gt = {row["movie_id"]}
        recs = score_item_item(item_model, row["user_id"], [row["movie_id"]], top_k=top_k)
        results_ii.append({
            "hit": hit_rate(recs, gt, top_k),
            "precision": precision_at_k(recs, gt, top_k),
            "recall": recall_at_k(recs, gt, top_k),
            "ndcg": ndcg_at_k(recs, gt, top_k),
        })
    agg["item_item"] = pd.DataFrame(results_ii).mean().to_dict()

    # XGBoost
    xgb_model = train_xgboost(train_df)
    user_ranked = rank_xgboost(xgb_model, user_enc, movie_enc, train_df, top_k=top_k)
    agg["xgboost"] = evaluate_xgboost_ranking(user_ranked, test_df, top_k=top_k)

    return agg


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_data.csv")
    parser.add_argument("--output", default="results/offline_eval_with_xgb.json")
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()

    # Create results directory
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {args.input}...")
    df = load_data(args.input)
    df, user_enc, movie_enc = feature_encode(df)
    train_df, test_df = leave_one_out_split(df)

    print(f"Training and evaluating on {len(train_df)} interactions...")
    metrics = evaluate_model(train_df, test_df, user_enc, movie_enc, top_k=args.k)
    print(json.dumps(metrics, indent=2))

    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
