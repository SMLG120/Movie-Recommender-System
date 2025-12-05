#!/usr/bin/env python3
import argparse
import csv
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from evaluation.Online.online_eval import OnlineEvaluator  

API_URL = "http://localhost:8080"
ONLINE_LOG = REPO_ROOT / "evaluation" / "Online" / "logs" / "online_metrics.json"
TRAINING_DATA = REPO_ROOT / "data" / "training_data_v2.csv"
EVALUATOR = OnlineEvaluator(log_path=str(ONLINE_LOG))
EVAL_LOCK = threading.Lock()

def parse_recs(resp):
    try:
        if resp.status_code != 200:
            return []
        # try JSON first
        try:
            j = resp.json()
            if isinstance(j, dict) and "recommendations" in j:
                recs = j["recommendations"]
                if recs and isinstance(recs[0], dict) and "item_id" in recs[0]:
                    return [str(x["item_id"]) for x in recs]
                return [str(x) for x in recs]
            if isinstance(j, list):
                return [str(x) for x in j]
        except Exception:
            # fallback: CSV / plain
            txt = resp.text.strip()
            if "," in txt:
                return [x.strip() for x in txt.split(",") if x.strip()]
            return [txt] if txt else []
    except Exception:
        return []
    return []

def do_bad_request():
    try:
        r = random.choice([
            lambda: requests.get(f"{API_URL}/recommend", params={"top_n": -1}, timeout=3),
            lambda: requests.get(f"{API_URL}/nope", timeout=3),
            lambda: requests.post(f"{API_URL}/event/click", data="notjson", timeout=3),
        ])()
        return r.status_code
    except Exception as e:
        return f"err:{e}"


def load_user_ids():
    """Load real user_ids from training data"""
    if not TRAINING_DATA.exists():
        return []
    user_ids = []
    try:
        with TRAINING_DATA.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid_raw = row.get("user_id")
                if uid_raw is None:
                    continue
                try:
                    user_ids.append(int(float(uid_raw)))
                except Exception:
                    continue
    except Exception:
        return []
    return user_ids


def worker(user_ids, topn_choices, click_p):
    user_id = random.choice(user_ids)
    top_n = random.choice(topn_choices)
    #added for canary
    model_version = "v2" if random.random() < 0.10 else "v1"
    chosen = None  

    try:
        r = requests.get(f"{API_URL}/recommend",
                         params={"user_id": user_id, "top_n": top_n}, timeout=600)
        recs = parse_recs(r)
        resp_time = r.elapsed.total_seconds() if getattr(r, "elapsed", None) else 0.0
        if recs:
            with EVAL_LOCK:
                EVALUATOR.log_recommendation(str(user_id), recs, resp_time)
                #for canary
                if EVALUATOR.metrics.get("recommendations"):
                    EVALUATOR.metrics["recommendations"][-1]["model_version"] = model_version
    except Exception as e:
        return f"user={user_id} rec_err={e}"

    msg = [f"user={user_id}", f"rec_count={len(recs)}", f"status={getattr(r,'status_code', 'NA')}"]

    # choose an item only if we got recs
    if recs:
        chosen = random.choice(recs)

    # /event/click 
    if chosen and random.random() < click_p:
        try:
            rc = requests.post(f"{API_URL}/event/click",
                               json={"user_id": user_id, "item_id": chosen, "k": len(recs)},
                               timeout=4).status_code
            msg.append(f"click={rc}")
            if rc == 200:
                with EVAL_LOCK:
                    EVALUATOR.log_interaction(str(user_id), str(chosen), "click")
                    #for canary
                    if EVALUATOR.metrics.get("user_interactions"):
                        EVALUATOR.metrics["user_interactions"][-1]["model_version"] = model_version
        except Exception as e:
            msg.append(f"click_err={e}")

    # /event/rating 
    if chosen:
        true = random.choice([1,2,3,4,5])
        pred = max(1.0, min(5.0, true + random.uniform(-0.7, 0.7)))
        try:
            rc = requests.post(f"{API_URL}/event/rating",
                               json={"user_id": user_id, "item_id": chosen,
                                     "rating": true, "predicted": pred},
                               timeout=4).status_code
            msg.append(f"rate={rc}")
            if rc == 200 and recs:
                with EVAL_LOCK:
                    EVALUATOR.log_recommendation_quality(
                        user_id=str(user_id),
                        recommended_items=recs,
                        selected_item=str(chosen),
                        rating_score=true,
                    )
                    #for canary
                    if EVALUATOR.metrics.get("recommendation_quality"):
                        EVALUATOR.metrics["recommendation_quality"][-1]["model_version"] = model_version
        except Exception as e:
            msg.append(f"rate_err={e}")

    # a few bad requests for error-rate 
    if random.random() < 0.1:
        msg.append(f"bad={do_bad_request()}")

    return " ".join(msg)

def run(users=20, concurrency=3, delay=1.0, click_p=0.25):
    topn_choices = [5, 10, 20]
    real_user_ids = load_user_ids()
    # fallback to synthetic ids if training data is missing
    user_ids = real_user_ids if real_user_ids else list(range(1, users + 1))

    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        while True:
            futures = [pool.submit(worker, user_ids, topn_choices, click_p) for _ in range(concurrency)]
            for f in as_completed(futures):
                print(time.strftime("[%H:%M:%S]"), f.result())
            time.sleep(delay)
    except KeyboardInterrupt:
        print("stopping…")
    finally:
        pool.shutdown(cancel_futures=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--click-p", type=float, default=0.25)
    args = ap.parse_args()
    run(args.users, args.concurrency, args.delay, args.click_p)
