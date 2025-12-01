from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Thresholds – tunable
CTR_THRESHOLD = 0.10           # if CTR < 10% -> feedback issue
CANARY_CTR_RATIO_THRESHOLD = 0.70 
SAT_RATIO_THRESHOLD = 0.85     # if sat_F / sat_M < 0.85 -> fairness issue

REPO_ROOT = Path(__file__).resolve().parents[1]
ONLINE_LOG = REPO_ROOT / "evaluation" / "Online" / "logs" / "online_metrics.json"
FEEDBACK_EVENTS_LOG = REPO_ROOT / "evaluation" / "Online" / "logs" / "feedback_events.json"
TRAINING_DATA = REPO_ROOT / "data" / "training_data_v2.csv"   

CANARY_CONFIG = REPO_ROOT / "tests" / "deployment" / "config" / "canary_config.json"
DEFAULT_CANARY_PERCENTAGE = 0.0 

# Data classes

@dataclass
class DetectionResult:
    issue: Optional[str]
    ctr: float
    ctr_threshold: float
    canary_ctr_ratio_threshold: float
    sat_male: Optional[float]
    sat_female: Optional[float]
    sat_ratio: Optional[float]
    sat_ratio_threshold: float
    n_rec_events: int
    n_interaction_events: int
    n_quality_events: int
    ctr_by_model: Dict[str, float] = field(default_factory=dict)

# ADDED: helpers to read/write canary percentage
def get_current_canary_percentage() -> float:
    """
    Read current canary percentage from CANARY_CONFIG.
    Returns DEFAULT_CANARY_PERCENTAGE if file or field is missing.
    """
    if not CANARY_CONFIG.exists():
        return DEFAULT_CANARY_PERCENTAGE
    try:
        with CANARY_CONFIG.open() as f:
            cfg = json.load(f)
        pct = float(cfg.get("canary_percentage", DEFAULT_CANARY_PERCENTAGE))
        # Clamp to [0, 100] to avoid crazy values
        return max(0.0, min(100.0, pct))
    except Exception as e:
        print(f"[feedback_detection] Failed to read canary config: {e}", file=sys.stderr)
        return DEFAULT_CANARY_PERCENTAGE


def set_canary_percentage(pct: float) -> None:
    """
    Write new canary percentage into CANARY_CONFIG.
    This is the main control knob that app.py reads to decide v1 vs v2 traffic.
    """
    CANARY_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    pct_clamped = max(0.0, min(100.0, float(pct)))
    cfg = {"canary_percentage": pct_clamped}
    with CANARY_CONFIG.open("w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[feedback_detection] Updated canary percentage to {pct_clamped:.1f}%")



# Loader

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


# Metrics

def compute_ctr(
    recommendations: List[Dict[str, Any]],
    interactions: List[Dict[str, Any]],
) -> float:
    # Build user 
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

    # Count interactions
    hits = 0
    for ev in interactions:
        uid = str(ev.get("user_id"))
        item_id = str(ev.get("item_id"))
        rec_items = user_to_recommended.get(uid, [])
        if item_id in rec_items:
            hits += 1

    return hits / total_recommended

# ADDED: per-model CTR, assuming each recommendation has "model_version"
def compute_ctr_per_model(
    recommendations: List[Dict[str, Any]],
    interactions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute CTR per model_version (e.g., "v1", "v2").

    CTR(model) = hits(model) / total_items_shown_by_model

    We approximate hits(model) by checking, for each interaction,
    whether the clicked item appears in any recommendation list
    previously shown to that user by that model.
    """
    # Map: model -> user -> list of items recommended by that model
    model_user_to_items: Dict[str, Dict[str, List[str]]] = {}
    total_by_model: Dict[str, int] = {}

    for rec in recommendations:
        model = str(rec.get("model_version", "unknown"))
        uid = str(rec.get("user_id"))
        items = rec.get("items", []) or []
        items = [str(x) for x in items]

        total_by_model[model] = total_by_model.get(model, 0) + len(items)
        user_items = model_user_to_items.setdefault(model, {}).setdefault(uid, [])
        user_items.extend(items)

    if not total_by_model:
        return {}

    hits_by_model: Dict[str, int] = {m: 0 for m in total_by_model.keys()}

    for ev in interactions:
        uid = str(ev.get("user_id"))
        item_id = str(ev.get("item_id"))
        for model, user_to_items in model_user_to_items.items():
            rec_items = user_to_items.get(uid, [])
            if item_id in rec_items:
                hits_by_model[model] += 1

    ctr_by_model: Dict[str, float] = {}
    for model, total in total_by_model.items():
        if total > 0:
            ctr_by_model[model] = hits_by_model[model] / total
        else:
            ctr_by_model[model] = 0.0

    return ctr_by_model


def compute_group_satisfaction(
    quality_events: List[Dict[str, Any]],
    user_gender: Dict[str, str],
) -> Dict[str, float]:
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


# Detection of feedback loop and fairness issues

def detect_feedback_and_fairness() -> DetectionResult:
    data = load_online_metrics()
    recs = data["recommendations"]
    inter = data["user_interactions"]
    qual = data["recommendation_quality"]

    ctr_global = compute_ctr(recs, inter)
    print(f"[feedback_detection] Global CTR = {ctr_global:.4f} (threshold {CTR_THRESHOLD})")
    
    # ADDED: per-model CTR
    ctr_by_model = compute_ctr_per_model(recs, inter)
    if ctr_by_model:
        for model, ctr in ctr_by_model.items():
            print(f"[feedback_detection] CTR[{model}] = {ctr:.4f}")
    else:
        print("[feedback_detection] No per-model CTR data (no recommendations logged).")

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

    # feedback based on CTR
    if ctr_global < CTR_THRESHOLD:
        issue = "feedback_low_ctr"

    # original & canary model CTR comparison
    ctr_og = ctr_by_model.get("v1")
    ctr_canary = ctr_by_model.get("v2")

    if ctr_canary is not None:
        # Absolute canary failure
        if ctr_canary < CTR_THRESHOLD:
            issue = (issue + "+canary_low_ctr_abs") if issue else "canary_low_ctr_abs"

        # Relative canary underperformance vs v1
        if ctr_og is not None and ctr_og > 0:
            if ctr_canary < CANARY_CTR_RATIO_THRESHOLD * ctr_og:
                issue = (issue + "+canary_low_ctr_rel") if issue else "canary_low_ctr_rel"


    # Feedback based on gender satisfaction
    if sat_ratio is not None and sat_ratio < SAT_RATIO_THRESHOLD:
        issue = (issue + "+fairness_gender") if issue else "fairness_gender"

    return DetectionResult(
        issue=issue,
        ctr=ctr_global,
        ctr_threshold=CTR_THRESHOLD,
        canary_ctr_ratio_threshold=CANARY_CTR_RATIO_THRESHOLD,
        sat_male=sat_m,
        sat_female=sat_f,
        sat_ratio=sat_ratio,
        sat_ratio_threshold=SAT_RATIO_THRESHOLD,
        n_rec_events=len(recs),
        n_interaction_events=len(inter),
        n_quality_events=len(qual),
        ctr_by_model=ctr_by_model,
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
    print("[feedback_detection] ABORTING CANARY RELEASE (issue detected).")


def trigger_retraining_pipeline() -> None:
    print("[feedback_detection] TRIGGERING RETRAINING PIPELINE (placeholder).")


def main() -> None:
    result = detect_feedback_and_fairness()

    if result.issue is None:
        print("[feedback_detection] No feedback or fairness issue detected.")
        append_feedback_event(result)

        # Simple progressive rollout logic based on current percentage
        current = get_current_canary_percentage()
        print(f"[feedback_detection] Current canary percentage = {current:.1f}%")

        # NOTE: This is a simple staircase policy; you can tune thresholds & steps.
        if current < 5.0:
            new_pct = 5.0
        elif current < 10.0:
            new_pct = 10.0
        elif current < 25.0:
            new_pct = 25.0
        elif current < 50.0:
            new_pct = 50.0
        elif current < 100.0:
            new_pct = 100.0
        else:
            new_pct = current  # already fully rolled out

        if new_pct != current:
            print(f"[feedback_detection] Increasing canary percentage to {new_pct:.1f}%")
            set_canary_percentage(new_pct)
        else:
            print("[feedback_detection] Keeping canary percentage unchanged.")        
        
        sys.exit(0)


    print(f"[feedback_detection] ISSUE DETECTED: {result.issue}")
    append_feedback_event(result)
    # Set 0% traffic to canary model - automatic rollback
    print("[feedback_detection] Setting canary percentage to 0.0% due to detected issue.")
    set_canary_percentage(0.0)
    abort_canary_release()
    trigger_retraining_pipeline()
    sys.exit(1)

if __name__ == "__main__":
    main()
