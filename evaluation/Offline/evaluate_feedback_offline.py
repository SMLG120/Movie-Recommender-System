from pathlib import Path
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

# Paths
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_data"


# Data classes

@dataclass
class PopularityStats:
    n_movies: int
    corr_popularity_interactions: float
    top_share_fraction: float
    top_share_percent: float
    top_k_fraction: float = 0.1  


@dataclass
class GroupQualityStats:
    group: str
    n_interactions: int
    mean_rating: float
    mae: float | None = None  


# Feedback loop
def analyze_popularity_feedback() -> PopularityStats:
    df = pd.read_csv(DATA_DIR / "training_data_v2.csv")
    # interactions per movie in training data
    inter_per_movie = df.groupby("movie_id").size().rename("interaction_count")

    # average popularity score per movie
    pop_per_movie = df.groupby("movie_id")["popularity"].mean()
    stats = pd.concat([inter_per_movie, pop_per_movie], axis=1).dropna()
    n_movies = len(stats)
    corr = stats["interaction_count"].corr(stats["popularity"])

    # how concentrated are interactions
    k = max(1, int(0.1 * n_movies))
    top = stats.sort_values("interaction_count", ascending=False).head(k)
    share = top["interaction_count"].sum() / stats["interaction_count"].sum()

    return PopularityStats(
        n_movies=n_movies,
        corr_popularity_interactions=float(corr),
        top_share_fraction=float(share),
        top_share_percent=float(share * 100.0),
        top_k_fraction=0.1,
    )


# Fairness

def analyze_user_group_fairness(group_col: str = "gender") -> list[GroupQualityStats]:
    df = pd.read_csv(DATA_DIR / "training_data_v2.csv")

    if group_col not in df.columns:
        raise ValueError(f"{group_col} not in training_data_v2.csv columns")
    
    has_pred = "estimated_rating" in df.columns or "prediction" in df.columns
    pred_col = "estimated_rating" if "estimated_rating" in df.columns else "prediction"

    rating_col = "rating" if "rating" in df.columns else None

    group_stats: list[GroupQualityStats] = []
    for group_value, gdf in df.groupby(group_col):
        n = len(gdf)
        mean_rating = float(gdf[rating_col].mean()) if rating_col else float("nan")
        if has_pred and rating_col:
            mae = float((gdf[pred_col] - gdf[rating_col]).abs().mean())
        else:
            mae = None
        group_stats.append(
            GroupQualityStats(
                group=str(group_value),
                n_interactions=int(n),
                mean_rating=mean_rating,
                mae=mae,  
            )
        )

    return group_stats


# Genre diversity

def analyze_genre_diversity():

    df = pd.read_csv(DATA_DIR / "training_data_v2.csv", usecols=["user_id", "movie_id"])
    movies = pd.read_csv(RAW_DIR / "movies.csv", usecols=["id", "genres"])

    merged = df.merge(movies, left_on="movie_id", right_on="id", how="left")
    merged["genres_list"] = merged["genres"].fillna("").apply(
        lambda g: [s.strip() for s in g.split(",") if s.strip()]
    )

    def distinct_genres(lists):
        return len({g for sub in lists for g in sub})

    user_div = (
        merged.groupby("user_id")["genres_list"]
        .apply(distinct_genres)
        .rename("n_genres")
        .reset_index()
    )

    return {
        "n_users": int(user_div.shape[0]),
        "mean_genres": float(user_div["n_genres"].mean()),
        "median_genres": float(user_div["n_genres"].median()),
        "p10_genres": float(user_div["n_genres"].quantile(0.10)),
        "p90_genres": float(user_div["n_genres"].quantile(0.90)),
    }


# Offline “log-based” dataset overview 
def analyze_logs():
    df = pd.read_csv(DATA_DIR / "training_data_v2.csv")

    n_rows = len(df)
    n_users = df["user_id"].nunique()
    n_movies = df["movie_id"].nunique()

    # rating stats 
    rating_stats = {}
    if "rating" in df.columns:
        rating_stats = {
            "mean_rating": float(df["rating"].mean()),
            "std_rating": float(df["rating"].std()),
            "min_rating": float(df["rating"].min()),
            "max_rating": float(df["rating"].max()),
        }

    # interactions per user / per movie
    inter_per_user = df.groupby("user_id").size()
    inter_per_movie = df.groupby("movie_id").size()

    offline_summary = {
        "n_rows": int(n_rows),
        "n_users": int(n_users),
        "n_movies": int(n_movies),
        "mean_interactions_per_user": float(inter_per_user.mean()),
        "median_interactions_per_user": float(inter_per_user.median()),
        "mean_interactions_per_movie": float(inter_per_movie.mean()),
        "median_interactions_per_movie": float(inter_per_movie.median()),
    }
    offline_summary.update(rating_stats)

    return offline_summary

def main(out_path: Path | None = None):
    pop_stats = analyze_popularity_feedback()
    fairness_gender = analyze_user_group_fairness("gender")
    genre_div = analyze_genre_diversity()
    offline_log_stats = analyze_logs()

    summary = {
        "popularity_stats": asdict(pop_stats),
        "fairness_by_gender": [asdict(g) for g in fairness_gender],
        "genre_diversity": genre_div,
        "offline_log_stats": offline_log_stats,
    }

    if out_path is None:
        out_path = (
            REPO_ROOT
            / "evaluation"
            / "Offline"
            / "offline_feedback_analysis.json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[analysis] Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
