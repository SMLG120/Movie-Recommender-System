from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import shutil
import os

# Thresholds
CTR_THRESHOLD = 0.10           # if CTR < 10% -> feedback issue
CANARY_CTR_RATIO_THRESHOLD = 0.70  # v2 must be at least 70% as good as v1
RATING_RATIO_THRESHOLD = 0.85  # if avg_rating_F / avg_rating_M < 0.85 -> fairness issue

# Path plumbing: repo root is two levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_LOG = REPO_ROOT / "evaluation" / "Online" / "logs" / "online_metrics.json"
FEEDBACK_EVENTS_LOG = REPO_ROOT / "evaluation" / "Online" / "logs" / "feedback_events.json"
TRAINING_DATA = REPO_ROOT / "data" / "training_data_v2.csv"

# Canary configuration file (read / written by evaluator and app)
CANARY_CONFIG = REPO_ROOT / "tests" / "deployment" / "config" / "canary_config.json"
# Start canary at 10% by default for safer incremental rollout
DEFAULT_CANARY_PERCENTAGE = 10.0


# Data classes
@dataclass
class DetectionResult:
    issue: Optional[str]
    ctr: float
    ctr_threshold: float
    # Canary thresholds + per-model ctr
    canary_ctr_ratio_threshold: float
    rating_male: Optional[float]
    rating_female: Optional[float]
    rating_ratio: Optional[float]
    rating_ratio_threshold: float
    n_rec_events: int
    n_interaction_events: int
    n_quality_events: int
    ctr_by_model: Dict[str, float] = field(default_factory=dict)


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


# Helpers to read/write canary percentage
def get_current_canary_percentage() -> float:
    # If config is missing, prefer safe behavior: assume 0% canary
    if not CANARY_CONFIG.exists():
        return 0.0
    try:
        with CANARY_CONFIG.open() as f:
            cfg = json.load(f)
        pct = float(cfg.get("canary_percentage", DEFAULT_CANARY_PERCENTAGE))
        return max(0.0, min(100.0, pct))
    except Exception as e:
        print(f"[feedback_detection] Failed to read canary config: {e}", file=sys.stderr)
        return 0.0


def set_canary_percentage(pct: float) -> None:
    CANARY_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    pct_clamped = max(0.0, min(100.0, float(pct)))
    cfg = {"canary_percentage": pct_clamped}
    with CANARY_CONFIG.open("w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[feedback_detection] Updated canary percentage to {pct_clamped:.1f}%")




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


def compute_ctr_per_model(
    recommendations: List[Dict[str, Any]],
    interactions: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Compute CTR per model_version (e.g., "v1", "v2").

    CTR(model) = hits(model) / total_items_shown_by_model
    """
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


def compute_group_rating(
    quality_events: List[Dict[str, Any]],
    user_gender: Dict[str, str],
) -> Dict[str, float]:
    ratings_by_gender: Dict[str, List[float]] = {}

    for ev in quality_events:
        uid = str(ev.get("user_id"))
        g = user_gender.get(uid, "unknown")
        rating_val = ev.get("rating")
        if rating_val is None:
            continue
        try:
            r = float(rating_val)
        except Exception:
            continue

        ratings_by_gender.setdefault(g, []).append(r)

    avg_ratings: Dict[str, float] = {}
    for g, vals in ratings_by_gender.items():
        if vals:
            avg_ratings[g] = sum(vals) / len(vals)

    return avg_ratings


# Detection of feedback loop and fairness issues

def detect_feedback_and_fairness() -> DetectionResult:
    data = load_online_metrics()
    recs = data["recommendations"]
    inter = data["user_interactions"]
    qual = data["recommendation_quality"]
    ctr = compute_ctr(recs, inter)
    print(f"[feedback_detection] Global CTR = {ctr:.4f} (threshold {CTR_THRESHOLD})")

    # Per-model CTR (for canary vs stable)
    ctr_by_model = compute_ctr_per_model(recs, inter)
    if ctr_by_model:
        for model, v in ctr_by_model.items():
            print(f"[feedback_detection] CTR[{model}] = {v:.4f}")
    else:
        print("[feedback_detection] No per-model CTR data (no recommendations logged).")

    user_gender_map = load_user_gender_mapping()
    avg_ratings = compute_group_rating(qual, user_gender_map)

    rating_m = avg_ratings.get("M")
    rating_f = avg_ratings.get("F")

    rating_ratio = None
    if rating_m is not None and rating_m > 0 and rating_f is not None:
        rating_ratio = rating_f / rating_m

    if rating_m is not None:
        print(f"[feedback_detection] avg rating (M) = {rating_m:.4f}")
    if rating_f is not None:
        print(f"[feedback_detection] avg rating (F) = {rating_f:.4f}")
    if rating_ratio is not None:
        print(
            f"[feedback_detection] rating ratio F/M = "
            f"{rating_ratio:.4f} (threshold {RATING_RATIO_THRESHOLD})"
        )

    issue: Optional[str] = None

    # Global CTR feedback
    if ctr < CTR_THRESHOLD:
        issue = "feedback_low_ctr"

    # Compare original vs canary CTR
    ctr_og = ctr_by_model.get("v1")
    ctr_canary = ctr_by_model.get("v2")

    if ctr_canary is not None:
        # Absolute canary failure
        if ctr_canary < CTR_THRESHOLD:
            issue = (issue + "+canary_low_ctr_abs") if issue else "canary_low_ctr_abs"

        # Relative underperformance vs v1
        if ctr_og is not None and ctr_og > 0:
            if ctr_canary < CANARY_CTR_RATIO_THRESHOLD * ctr_og:
                issue = (issue + "+canary_low_ctr_rel") if issue else "canary_low_ctr_rel"

    # Fairness based on rating ratio
    if rating_ratio is not None and rating_ratio < RATING_RATIO_THRESHOLD:
        issue = (issue + "+fairness_gender") if issue else "fairness_gender"

    return DetectionResult(
        issue=issue,
        ctr=ctr,
        ctr_threshold=CTR_THRESHOLD,
        canary_ctr_ratio_threshold=CANARY_CTR_RATIO_THRESHOLD,
        rating_male=rating_m,
        rating_female=rating_f,
        rating_ratio=rating_ratio,
        rating_ratio_threshold=RATING_RATIO_THRESHOLD,
        n_rec_events=len(recs),
        n_interaction_events=len(inter),
        n_quality_events=len(qual),
        ctr_by_model=ctr_by_model,
    )


def append_feedback_event(result: DetectionResult, action_taken: Optional[str] = None, reason: Optional[str] = None, pct_after: Optional[float] = None) -> None:
    FEEDBACK_EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)

    if FEEDBACK_EVENTS_LOG.exists():
        with FEEDBACK_EVENTS_LOG.open() as f:
            events = json.load(f)
    else:
        events = []
    
    pct_before = get_current_canary_percentage()
    
    # Determine event type and action
    if action_taken is None:
        if result.issue:
            action_taken = "ABORT"
            event_type = "canary_abort"
        else:
            action_taken = "RAMP"
            event_type = "canary_check"
    else:
        if action_taken == "PROMOTE_AND_RESET":
            event_type = "promotion"
        else:
            event_type = "canary_check" if action_taken == "RAMP" else "canary_abort"
    
    # Build reason if not provided
    if reason is None:
        if result.issue:
            reason = f"Issue detected: {result.issue}"
        else:
            reason = "No issues detected. Proceeding with canary rollout."
    
    # Default to current percentage if not provided
    if pct_after is None:
        pct_after = pct_before
    
    event = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "event_type": event_type,
        "issue": result.issue,
        "details": {
            "ctr": result.ctr,
            "ctr_threshold": result.ctr_threshold,
            "ctr_by_model": result.ctr_by_model,
            "canary_ctr_ratio_threshold": result.canary_ctr_ratio_threshold,
            "n_rec_events": result.n_rec_events,
            "n_interaction_events": result.n_interaction_events,
            "n_quality_events": result.n_quality_events,
            "rating_male": result.rating_male,
            "rating_female": result.rating_female,
            "rating_ratio": result.rating_ratio,
            "rating_ratio_threshold": result.rating_ratio_threshold,
        },
        "action_taken": action_taken,
        "canary_percentage_before": pct_before,
        "canary_percentage_after": pct_after,
        "reason": reason,
    }
    
    events.append(event)

    with FEEDBACK_EVENTS_LOG.open("w") as f:
        json.dump(events, f, indent=2)

    print(f"[feedback_detection] Logged event to {FEEDBACK_EVENTS_LOG}")


def abort_canary_release() -> None:
    print("[feedback_detection] ABORTING CANARY RELEASE (issue detected).")


def trigger_retraining_pipeline() -> None:
    print("[feedback_detection] TRIGGERING RETRAINING PIPELINE (placeholder).")


def _dir_has_files(p: Path) -> bool:
    try:
        if not p.exists() or not p.is_dir():
            return False
        for _ in p.iterdir():
            return True
        return False
    except Exception:
        return False


def promote_canary_to_stable() -> bool:
    """
    Promote the canary model (src/models/v2) to stable (src/models).
    Steps:
    - Validate v2 contains artifacts.
    - Move current v1 -> models_prev/<timestamp>/
    - Move v2 -> models/ (rename/move)
    - Recreate empty v2 dir for future canaries
    - Return True on success, False on failure (attempt rollback on failure)
    """
    v1_dir = REPO_ROOT / "src" / "models"
    v2_dir = REPO_ROOT / "src" / "models" / "v2"
    prev_root = REPO_ROOT / "src" / "models_prev"

    print(f"[feedback_detection] Attempting to promote canary: v2={v2_dir} -> v1={v1_dir}")

    if not _dir_has_files(v2_dir):
        print(f"[feedback_detection] No artifacts found in {v2_dir}; aborting promotion.")
        return False

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    prev_dir = prev_root / timestamp
    try:
        prev_root.mkdir(parents=True, exist_ok=True)

        # Move v1 -> prev_dir (if v1 exists)
        if v1_dir.exists():
            print(f"[feedback_detection] Backing up current v1 -> {prev_dir}")
            shutil.move(str(v1_dir), str(prev_dir))
        else:
            print(f"[feedback_detection] No existing v1 at {v1_dir}; skipping backup")

        # Move v2 -> v1 (promote)
        print(f"[feedback_detection] Promoting v2 -> v1")
        shutil.move(str(v2_dir), str(v1_dir))

        # Recreate an empty v2 dir for future canaries
        v2_dir.mkdir(parents=True, exist_ok=True)

        print("[feedback_detection] Promotion completed successfully.")
        return True

    except Exception as e:
        print(f"[feedback_detection] Promotion failed: {e}")
        # Attempt rollback: if prev_dir exists and v1 missing, restore
        try:
            if prev_dir.exists() and not v1_dir.exists():
                print(f"[feedback_detection] Attempting rollback: restore {prev_dir} -> {v1_dir}")
                shutil.move(str(prev_dir), str(v1_dir))
        except Exception as re:
            print(f"[feedback_detection] Rollback failed: {re}")
        return False


def main() -> None:
    result = detect_feedback_and_fairness()

    if result.issue is None:
        print("[feedback_detection] No feedback or fairness issue detected.")

        # If we have no v2 traffic in the logs, there's nothing to roll out.
        if "v2" not in result.ctr_by_model:
            print("[feedback_detection] No canary (v2) traffic observed in logs; "
                  "skipping canary rollout step.")
            append_feedback_event(result, action_taken="RAMP", reason="No canary traffic observed.")
            sys.exit(0)
            
        # Simple progressive rollout logic based on current percentage
        current = get_current_canary_percentage()
        print(f"[feedback_detection] Current canary percentage = {current:.1f}%")
        # Only act when we have a minimum number of interactions to avoid noise
        n_inter = result.n_interaction_events
        if n_inter < 10:
            print(f"[feedback_detection] Insufficient interactions ({n_inter}) to act; need >= 10.")
            append_feedback_event(result, action_taken="RAMP", reason=f"Insufficient interactions ({n_inter}); need >= 10.")
            sys.exit(0)

        # Act only on every 10th interaction batch to avoid frequent small changes
        if (n_inter % 10) != 0:
            print(f"[feedback_detection] Not acting until interactions reach a multiple of 10 (have {n_inter}).")
            append_feedback_event(result, action_taken="RAMP", reason=f"Waiting for interaction count multiple of 10 (have {n_inter}).")
            sys.exit(0)

        # Progressive +10% step policy
        if current < 10.0:
            new_pct = 10.0
        else:
            new_pct = min(100.0, current + 10.0)

        if new_pct != current:
            print(f"[feedback_detection] Increasing canary percentage to {new_pct:.1f}%")
            
            # Log the ramp event BEFORE setting the new percentage
            reason = f"CTR[v2]={result.ctr_by_model.get('v2', 0):.3f} > 70% of CTR[v1]={result.ctr_by_model.get('v1', 0):.3f}. Ramping to {new_pct:.0f}%."
            if new_pct >= 100.0:
                reason = f"All metrics healthy. Ramping to 100% for promotion."
            
            append_feedback_event(result, action_taken="RAMP", reason=reason, pct_after=new_pct)
            
            # Now update the canary percentage
            set_canary_percentage(new_pct)

            # If we've reached 100%, promote canary to stable and then set canary back to 0%
            if new_pct >= 100.0:
                print("[feedback_detection] Canary reached 100% — attempting promotion to stable.")
                promoted = promote_canary_to_stable()
                if promoted:
                    print("[feedback_detection] Promotion succeeded — setting canary percentage to 0%.")
                    # Log the promotion event
                    promotion_reason = f"Canary reached 100% with sustained performance (CTR[v2]={result.ctr_by_model.get('v2', 0):.3f} > 70% of v1={result.ctr_by_model.get('v1', 0):.3f}). Promotion successful."
                    append_feedback_event(result, action_taken="PROMOTE_AND_RESET", reason=promotion_reason, pct_after=0.0)
                    set_canary_percentage(0.0)
                else:
                    print("[feedback_detection] Promotion failed — setting canary percentage to 0% and aborting.")
                    set_canary_percentage(0.0)
                    sys.exit(1)
        else:
            print("[feedback_detection] Keeping canary percentage unchanged.")
            append_feedback_event(result, action_taken="RAMP", reason="Canary percentage unchanged.")

        sys.exit(0)


    print(f"[feedback_detection] ISSUE DETECTED: {result.issue}")
    reason = f"Issue detected: {result.issue}. Canary release aborted and traffic returned to v1 exclusively."
    append_feedback_event(result, action_taken="ABORT", reason=reason, pct_after=0.0)
    # Set 0% traffic to canary model - automatic rollback
    print("[feedback_detection] Setting canary percentage to 0.0% due to detected issue.")
    set_canary_percentage(0.0)
    abort_canary_release()
    trigger_retraining_pipeline()
    sys.exit(1)

if __name__ == "__main__":
    main()
