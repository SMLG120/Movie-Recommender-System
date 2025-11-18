"""
Feedback loop & fairness detector for online metrics.

Schema of evaluation/Online/logs/online_metrics.json:

{
  "recommendations": [
    {"timestamp": "...", "user_id": "u1", "items": ["m1", "m2", "m3"]},
    ...
  ],
  "user_interactions": [
    {"timestamp": "...", "user_id": "u1", "item_id": "m2",
     "action_type": "click" | "watch", "watch_time": 20},
    ...
  ],
  "recommendation_quality": [
    {"timestamp": "...", "user_id": "u1",
     "recommendations": ["m1","m2","m3"],
     "selected": "m2", "satisfaction": 0.8},
    ...
  ],
  ...
}

Usage (from repo root):
    python monitoring/feedback_detection.py

Exit code:
    0 -> no issue detected
    1 -> feedback loop or fairness issue detected
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ----------------- CONFIG -----------------

# Thresholds – tune these if you want
CTR_THRESHOLD = 0.10           # if CTR < 10% -> feedback issue
SAT_RATIO_THRESHOLD = 0.85     # if sat_F / sat_M < 0.85 -> fairness issue

REPO_ROOT = Path(__file__).resolve().parents[1]
ONLINE_LOG = REPO_ROOT / "evaluation" / "Online" / "logs" / "online_metrics.json"
FEEDBACK_EVENTS_LOG = REPO_ROOT / "evaluation" / "Online" / "logs" / "feedback_events.json"
TRAINING_DATA = REPO_ROOT / "data" / "training_data_v2.csv"   # needs columns: user_id, gender


# ----------------- DATA CLASS -----------------

@dataclass
class DetectionResult:
    issue: Optional[str]
    ctr: float
    ctr_threshold: float
    sat_male: Optional[float]
    sat_female: Optional[float]
    sat_ratio: Optional[float]
    sat_ratio_threshold: float
    n_rec_events: int
    n_interaction_events: int
    n_quality_events: int


# ----------------- LOADERS -----------------

def load_online_metrics() -> Dict[str, Any]:
    if not ONLINE_LOG.exists():
        print(f"[feedback_detection] No online metrics log at {ONLINE_LOG}", file=sys.stderr)
        return {"recommendations": [], "user_interactions": [], "recommendation_quality": []}

    with ONLINE_LOG.open() as f:
        data = json.load(f)

    data.setdefault("recommendations", [])
    data.setdefault("user_interactions", [])
    data.setdefault("recommendation_quality", [])
    return data


def load_user_gender_mapping() -> Dict[str, str]:
    """
    Build a mapping user_id (as STRING) -> gender from training_data_v2.csv.
    Assumes columns: user_id, gender.
    """
    if not TRAINING_DATA.exists():
        print(f"[feedback_detection] No training data at {TRAINING_DATA}", file=sys.stderr)
        return {}

    df = pd.read_csv(TRAINING_DATA, usecols=["user_id", "gender"])
    df = df.dropna(subset=["user_id"])
    df["user_id"] = df["user_id"].astype(str)
    df = df.drop_duplicates(subset=["user_id"], keep="first")

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        uid = str(row["user_id"])
        g = row.get("gender", "unknown")
        mapping[uid] = str(g) if pd.notna(g) else "unknown"
    return mapping


# ----------------- METRIC COMPUTATION -----------------

def compute_ctr(
    recommendations: List[Dict[str, Any]],
    interactions: List[Dict[str, Any]],
) -> float:
    """
    CTR = (# interactions where item was recommended to that user)
          / (total number of recommended items)

    Very simple approximation based on your JSON schema.
    """

    # Build user -> set/list of recommended items
    user_to_recommended: Dict[str, List[str]] = {}
    total_recommended = 0

    for rec in recommendations:
        uid = str(rec.get("user_id"))
        items = rec.get("items", []) or []
        items = [str(x) for x in items]
        total_recommended += len(items)
        user_to_recommended.setdefault(uid, []).extend(items)

    if total_recommended == 0:
        return 0.0

    # Count interactions where item_id is in that user's recommended items
    hits = 0
    for ev in interactions:
        uid = str(ev.get("user_id"))
        item_id = str(ev.get("item_id"))
        rec_items = user_to_recommended.get(uid, [])
        if item_id in rec_items:
            hits += 1

    return hits / total_recommended


def compute_group_satisfaction(
    quality_events: List[Dict[str, Any]],
    user_gender: Dict[str, str],
) -> Dict[str, float]:
    """
    Compute average satisfaction per gender.

    We look up gender by user_id using training_data_v2.csv.
    If a user_id is not found, it is grouped as 'unknown'.
    """

    sats_by_gender: Dict[str, List[float]] = {}

    for ev in quality_events:
        uid = str(ev.get("user_id"))
        g = user_gender.get(uid, "unknown")
        sat_val = ev.get("satisfaction")
        if sat_val is None:
            continue
        try:
            s = float(sat_val)
        except Exception:
            continue

        sats_by_gender.setdefault(g, []).append(s)

    avg_sats: Dict[str, float] = {}
    for g, vals in sats_by_gender.items():
        if vals:
            avg_sats[g] = sum(vals) / len(vals)

    return avg_sats


# ----------------- DETECTION -----------------

def detect_feedback_and_fairness() -> DetectionResult:
    data = load_online_metrics()
    recs = data["recommendations"]
    inter = data["user_interactions"]
    qual = data["recommendation_quality"]

    ctr = compute_ctr(recs, inter)
    print(f"[feedback_detection] Global CTR = {ctr:.4f} (threshold {CTR_THRESHOLD})")

    user_gender_map = load_user_gender_mapping()
    avg_sats = compute_group_satisfaction(qual, user_gender_map)

    sat_m = avg_sats.get("M")
    sat_f = avg_sats.get("F")

    sat_ratio = None
    if sat_m is not None and sat_m > 0 and sat_f is not None:
        sat_ratio = sat_f / sat_m

    if sat_m is not None:
        print(f"[feedback_detection] avg satisfaction (M) = {sat_m:.4f}")
    if sat_f is not None:
        print(f"[feedback_detection] avg satisfaction (F) = {sat_f:.4f}")
    if sat_ratio is not None:
        print(
            f"[feedback_detection] satisfaction ratio F/M = "
            f"{sat_ratio:.4f} (threshold {SAT_RATIO_THRESHOLD})"
        )

    issue: Optional[str] = None

    # Rule 1: feedback / degradation based on CTR
    if ctr < CTR_THRESHOLD:
        issue = "feedback_low_ctr"

    # Rule 2: fairness degradation based on gender satisfaction
    if sat_ratio is not None and sat_ratio < SAT_RATIO_THRESHOLD:
        issue = (issue + "+fairness_gender") if issue else "fairness_gender"

    return DetectionResult(
        issue=issue,
        ctr=ctr,
        ctr_threshold=CTR_THRESHOLD,
        sat_male=sat_m,
        sat_female=sat_f,
        sat_ratio=sat_ratio,
        sat_ratio_threshold=SAT_RATIO_THRESHOLD,
        n_rec_events=len(recs),
        n_interaction_events=len(inter),
        n_quality_events=len(qual),
    )


def append_feedback_event(result: DetectionResult) -> None:
    FEEDBACK_EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)

    if FEEDBACK_EVENTS_LOG.exists():
        with FEEDBACK_EVENTS_LOG.open() as f:
            events = json.load(f)
    else:
        events = []

    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "issue": result.issue,
        "details": asdict(result),
    }
    events.append(event)

    with FEEDBACK_EVENTS_LOG.open("w") as f:
        json.dump(events, f, indent=2)

    print(f"[feedback_detection] Logged event to {FEEDBACK_EVENTS_LOG}")


def abort_canary_release() -> None:
    # For the milestone, printing is enough; CI can also use exit code 1.
    print("[feedback_detection] ABORTING CANARY RELEASE (issue detected).")


def trigger_retraining_pipeline() -> None:
    # Hook this into your real pipeline if you want.
    print("[feedback_detection] TRIGGERING RETRAINING PIPELINE (placeholder).")


def main() -> None:
    result = detect_feedback_and_fairness()

    if result.issue is None:
        print("[feedback_detection] No feedback or fairness issue detected.")
        append_feedback_event(result)
        sys.exit(0)

    print(f"[feedback_detection] ISSUE DETECTED: {result.issue}")
    append_feedback_event(result)
    abort_canary_release()
    trigger_retraining_pipeline()
    sys.exit(1)

if __name__ == "__main__":
    main()
