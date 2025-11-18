"""
Feedback loop & fairness detector for online metrics.

Usage (from repo root):
    python monitoring/feedback_detection.py

Exit code:
    0  -> no issue detected
    1  -> feedback loop or fairness issue detected (caller should abort canary / trigger retrain)
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd


# ---------- CONFIG ----------

# Thresholds (you can tune these)
CTR_THRESHOLD = 0.10              # if global CTR < 10%, treat as feedback-loop issue
SAT_RATIO_THRESHOLD = 0.85        # if sat_female / sat_male < 0.85, treat as fairness issue

# Paths (adjust if your structure differs)
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
ONLINE_LOG = REPO_ROOT / "evaluation" / "Online" / "logs" / "online_metrics.json"
FEEDBACK_EVENTS_LOG = REPO_ROOT / "evaluation" / "Online" / "logs" / "feedback_events.json"

TRAINING_DATA = DATA_DIR / "training_data_v2.csv"  # used to map user_id -> gender


# ---------- DATA CLASSES ----------

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
    n_quality_events: int


# ---------- HELPERS TO LOAD DATA ----------

def load_online_metrics() -> Dict[str, Any]:
    if not ONLINE_LOG.exists():
        print(f"[feedback_detection] No online metrics log found at {ONLINE_LOG}", file=sys.stderr)
        return {"recommendations": [], "recommendation_quality": []}

    with ONLINE_LOG.open() as f:
        data = json.load(f)

    # Ensure structure
    data.setdefault("recommendations", [])
    data.setdefault("recommendation_quality", [])
    return data


def load_user_gender_mapping() -> Dict[int, str]:
    """
    Build a mapping user_id -> gender from training_data_v2.csv.
    Assumes columns: user_id, gender.
    """
    if not TRAINING_DATA.exists():
        print(f"[feedback_detection] No training data found at {TRAINING_DATA}", file=sys.stderr)
        return {}

    df = pd.read_csv(TRAINING_DATA, usecols=["user_id", "gender"])
    df = df.dropna(subset=["user_id"])
    # If multiple rows per user, just take the first non-null gender
    df = df.drop_duplicates(subset=["user_id"], keep="first")
    mapping = {}
    for _, row in df.iterrows():
        try:
            uid = int(row["user_id"])
        except Exception:
            continue
        g = row.get("gender", "unknown")
        mapping[uid] = str(g) if pd.notna(g) else "unknown"
    return mapping


# ---------- METRIC COMPUTATION ----------

def compute_global_ctr(recommendations: List[Dict[str, Any]]) -> float:
    """
    Compute CTR from recommendations events.

    Assumes each recommendation event has either:
      - 'hits' and 'k' fields, OR
      - 'hits' and 'items' (list of item ids)

    If your schema is different, adjust this function accordingly.
    """
    total_hits = 0
    total_recommended = 0

    for ev in recommendations:
        hits = ev.get("hits")
        if hits is None:
            # If you log per-item info, you could define hits differently
            # For simplicity, fall back to 0 if missing
            hits = 0
        total_hits += int(hits)

        if "k" in ev:
            total_recommended += int(ev["k"])
        elif "items" in ev and isinstance(ev["items"], list):
            total_recommended += len(ev["items"])
        else:
            # If you store a pre-computed CTR in the event itself,
            # you could aggregate it differently.
            pass

    if total_recommended == 0:
        return 0.0

    return total_hits / total_recommended


def compute_group_satisfaction(
    quality_events: List[Dict[str, Any]],
    user_gender: Dict[int, str],
) -> Dict[str, float]:
    """
    Compute average satisfaction per gender using recommendation_quality events.

    Assumes each quality event has:
      - 'user_id'
      - 'satisfaction' (float)

    If your schema uses a different field (e.g. 'quality_score'),
    adjust accordingly.
    """
    sats_by_gender: Dict[str, List[float]] = {}

    for ev in quality_events:
        uid = ev.get("user_id")
        if uid is None:
            continue
        try:
            uid_int = int(uid)
        except Exception:
            continue

        g = user_gender.get(uid_int, "unknown")
        sat_val = ev.get("satisfaction")

        if sat_val is None:
            # You could also define satisfaction from (actual_rating, estimated_rating) here.
            continue

        try:
            s = float(sat_val)
        except Exception:
            continue

        sats_by_gender.setdefault(g, []).append(s)

    avg_sats: Dict[str, float] = {}
    for g, vals in sats_by_gender.items():
        if len(vals) > 0:
            avg_sats[g] = sum(vals) / len(vals)

    return avg_sats


# ---------- DETECTION LOGIC ----------

def detect_feedback_and_fairness() -> DetectionResult:
    metrics = load_online_metrics()
    recs = metrics.get("recommendations", [])
    qual = metrics.get("recommendation_quality", [])

    ctr = compute_global_ctr(recs)
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

    # Rule 1: feedback loop / degradation based on CTR
    if ctr < CTR_THRESHOLD:
        issue = "feedback_popularity"

    # Rule 2: fairness degradation based on group satisfaction
    if sat_ratio is not None and sat_ratio < SAT_RATIO_THRESHOLD:
        # if there is already a feedback issue, append; else create fairness issue
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
        n_quality_events=len(qual),
    )


def append_feedback_event(result: DetectionResult) -> None:
    """
    Append a detection event to feedback_events.json.
    """
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

    print(f"[feedback_detection] Logged feedback event to {FEEDBACK_EVENTS_LOG}")


# ---------- PLACEHOLDERS FOR ACTIONS (RETRAIN / ABORT CANARY) ----------

def abort_canary_release() -> None:
    """
    Placeholder: implement your actual canary abort logic here.
    For the milestone, printing a clear message is enough, and CI
    can treat non-zero exit codes as abort signals.
    """
    print("[feedback_detection] ABORTING CANARY: issue detected.")


def trigger_retraining_pipeline() -> None:
    """
    Placeholder: implement your actual retraining trigger here.
    Example options:
      - Call a shell script: subprocess.run(["bash", "scripts/run_pipeline.sh"])
      - Call a Python entrypoint.
      - Trigger a GitLab CI job via API (advanced).
    """
    print("[feedback_detection] TRIGGERING RETRAINING PIPELINE (placeholder).")


# ---------- MAIN ----------

def main() -> None:
    result = detect_feedback_and_fairness()

    if result.issue is None:
        print("[feedback_detection] No feedback loop or fairness issue detected.")
        # Still log a “clean” event if you want
        append_feedback_event(result)
        sys.exit(0)

    print(f"[feedback_detection] ISSUE DETECTED: {result.issue}")
    append_feedback_event(result)

    # Take actions (for the milestone, printing is enough)
    abort_canary_release()
    trigger_retraining_pipeline()

    # Non-zero exit so CI/canary pipeline can stop
    sys.exit(1)


if __name__ == "__main__":
    main()