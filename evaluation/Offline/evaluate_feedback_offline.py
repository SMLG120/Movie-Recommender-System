from pathlib import Path
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_data"
ONLINE_LOG = REPO_ROOT / "evaluation" / "Online" / "logs" / "online_metrics.json"


# ---------- Data classes for clean, serializable summaries ----------
@dataclass
class PopularityStats:
    n_movies: int
    corr_popularity_interactions: float
    top_share_fraction: float
    top_share_percent: float
    top_k_fraction: float = 0.1  # top 10%


@dataclass
class GroupQualityStats:
    group: str
    n_interactions: int
    mean_rating: float
    mae: float | None = None  # you can fill this later from offline eval


# ---------- Feedback loop: popularity amplification ----------

def analyze_popularity_feedback() -> PopularityStats:
    df = pd.read_csv(DATA_DIR / "training_data_v2.csv")

    # interactions per movie in training data
    inter_per_movie = df.groupby("movie_id").size().rename("interaction_count")

    # average popularity score per movie
    pop_per_movie = df.groupby("movie_id")["popularity"].mean()

    stats = pd.concat([inter_per_movie, pop_per_movie], axis=1).dropna()

    n_movies = len(stats)
    corr = stats["interaction_count"].corr(stats["popularity"])

    # how concentrated are interactions? e.g., top 10% movies by interaction count
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


# ---------- Fairness: quality by user group (e.g., gender) ----------

def analyze_user_group_fairness(group_col: str = "gender") -> list[GroupQualityStats]:
    df = pd.read_csv(DATA_DIR / "training_data_v2.csv")

    if group_col not in df.columns:
        raise ValueError(f"{group_col} not in training_data_v2.csv columns")

    # use explicit rating if available
    rating_col = "rating" if "rating" in df.columns else None

    group_stats: list[GroupQualityStats] = []
    for group_value, gdf in df.groupby(group_col):
        n = len(gdf)
        mean_rating = float(gdf[rating_col].mean()) if rating_col else float("nan")
        group_stats.append(
            GroupQualityStats(
                group=str(group_value),
                n_interactions=int(n),
                mean_rating=mean_rating,
                mae=None,  # you can plug in per-group MAE later
            )
        )

    return group_stats


# ---------- Optional: genre diversity as a “filter bubble” proxy ----------

def analyze_genre_diversity():
    """
    Approximate filter-bubble risk by looking at genre diversity per user
    in the training data (no timestamps, so this is a snapshot).
    """
    df = pd.read_csv(DATA_DIR / "training_data_v2.csv", usecols=["user_id", "movie_id"])
    movies = pd.read_csv(RAW_DIR / "movies.csv", usecols=["id", "genres"])

    merged = df.merge(movies, left_on="movie_id", right_on="id", how="left")
    merged["genres_list"] = merged["genres"].fillna("").apply(
        lambda g: [s.strip() for s in g.split(",") if s.strip()]
    )

    def distinct_genres(lists):
        # flatten list-of-lists and take unique genres
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


# ---------- Log-based overview (small now, scalable later) ----------

def analyze_logs():
    """
    Very lightweight example using evaluation/Online/logs/online_metrics.json.
    In real deployment you'll just have more events and can extend this.
    """
    if not ONLINE_LOG.exists():
        return {}

    with open(ONLINE_LOG) as f:
        logs = json.load(f)

    recs = logs.get("recommendations", [])
    qualities = logs.get("recommendation_quality", [])

    return {
        "n_recommendation_events": len(recs),
        "n_quality_events": len(qualities),
    }


# ---------- CLI entry point ----------

def main(out_path: Path | None = None):
    pop_stats = analyze_popularity_feedback()
    fairness_gender = analyze_user_group_fairness("gender")
    genre_div = analyze_genre_diversity()
    log_stats = analyze_logs()

    summary = {
        "popularity_stats": asdict(pop_stats),
        "fairness_by_gender": [asdict(g) for g in fairness_gender],
        "genre_diversity": genre_div,
        "online_log_stats": log_stats,
    }

    if out_path is None:
        out_path = REPO_ROOT / "evaluation" / "Offline" / "offline_feedback_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[analysis] Wrote summary to {out_path}")


if __name__ == "__main__":
    main()
