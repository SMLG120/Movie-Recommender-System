#!/usr/bin/env python3
"""
simulate_traffic_full.py

Generate synthetic traffic to exercise /recommend, /event/click, /event/rating,
and a few bad requests so Grafana/Prometheus panels get real data.
"""

import requests
import random
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

API_URL = "http://localhost:8080"

def do_recommend(user_id, top_n=10, timeout=6):
    """Call /recommend; return list of movie ids (strings) or None on error."""
    try:
        r = requests.get(f"{API_URL}/recommend", params={"user_id": user_id, "top_n": top_n}, timeout=timeout)
        text = r.text.strip()
        if r.status_code != 200:
            return {"status": r.status_code, "body": text}
        # Accept either JSON, CSV, or plain list
        try:
            j = r.json()
            # if returns {"recommendations": [...]}
            if isinstance(j, dict) and "recommendations" in j:
                recs = j["recommendations"]
                # if recs are dicts with item_id keys
                if recs and isinstance(recs[0], dict) and "item_id" in recs[0]:
                    return [str(x["item_id"]) for x in recs]
                # else assume list of ids
                return [str(x) for x in recs]
            # if top-level is list
            if isinstance(j, list):
                return [str(x) for x in j]
        except Exception:
            # fallback parse CSV / plain
            if "," in text:
                return [x.strip() for x in text.split(",") if x.strip()]
            # single id
            if text:
                return [text]
        return None
    except Exception as e:
        return {"error": str(e)}

def do_click(user_id, item_id):
    """POST a click event to /event/click"""
    try:
        payload = {"user_id": user_id, "item_id": item_id}
        r = requests.post(f"{API_URL}/event/click", json=payload, timeout=4)
        return r.status_code
    except Exception as e:
        return {"error": str(e)}

def do_rating(user_id, item_id, predicted=None):
    """POST a rating event to /event/rating. Provide predicted to produce realistic MAE/RMSE."""
    try:
        # create a plausible rating and predicted if not provided
        true_rating = random.choice([1.0, 2.0, 3.0, 4.0, 5.0])
        if predicted is None:
            # simulated prediction close to true_rating with noise
            predicted = max(1.0, min(5.0, true_rating + random.normalvariate(0, 0.5)))
        payload = {"user_id": user_id, "item_id": item_id, "rating": float(true_rating), "predicted": float(predicted)}
        r = requests.post(f"{API_URL}/event/rating", json=payload, timeout=4)
        return r.status_code
    except Exception as e:
        return {"error": str(e)}

def do_bad_request():
    """Hit an invalid endpoint or malformed param to generate a 4xx/5xx."""
    choices = [
        lambda: requests.get(f"{API_URL}/recommend", params={"top_n": -1}, timeout=3),
        lambda: requests.get(f"{API_URL}/this-does-not-exist", timeout=3),
        lambda: requests.post(f"{API_URL}/event/click", data="notjson", timeout=3),
    ]
    try:
        r = random.choice(choices)()
        return getattr(r, "status_code", "err")
    except Exception as e:
        return {"error": str(e)}

def worker_task(user_id, do_click_prob, do_rate_prob, top_n_choices):
    """One worker iteration: call recommend, maybe click, maybe rating."""
    # 1) recommend
    top_n = random.choice(top_n_choices)
    rec = do_recommend(user_id, top_n=top_n)
    outcome = {"user": user_id, "recommend": None, "click": None, "rating": None, "bad": None}

    if isinstance(rec, dict):
        outcome["recommend"] = rec
    elif isinstance(rec, list):
        outcome["recommend"] = {"count": len(rec)}
        # pick item for click/rating
        if rec:
            chosen = random.choice(rec)
        else:
            chosen = None
    else:
        outcome["recommend"] = rec
        chosen = None

    # 2) optionally send click (depends on probability and chosen item)
    if chosen and random.random() < do_click_prob:
        c = do_click(user_id, chosen)
        outcome["click"] = c

    # 3) optionally send rating
    if chosen and random.random() < do_rate_prob:
        # produce a predicted value near the true rating
        rcode = do_rating(user_id, chosen, predicted=(random.uniform(1.0, 5.0)))
        outcome["rating"] = rcode

    # occasionally generate a bad request
    if random.random() < 0.05:  # 5% bad requests
        outcome["bad"] = do_bad_request()

    return outcome

def simulate(n_users=20, concurrency=3, delay=1.0, run_forever=True, do_click_prob=0.25, do_rate_prob=0.1):
    top_n_choices = [5, 10, 20]
    print(f"[START] simulate users={n_users} concurrency={concurrency} delay={delay}s click_p={do_click_prob} rate_p={do_rate_prob}")
    executor = ThreadPoolExecutor(max_workers=concurrency)
    try:
        while True:
            futures = []
            for _ in range(concurrency):
                uid = random.randint(1, n_users)
                futures.append(executor.submit(worker_task, uid, do_click_prob, do_rate_prob, top_n_choices))
            for fut in as_completed(futures, timeout=delay+1):
                try:
                    out = fut.result()
                    # concise logging for visibility
                    parts = []
                    parts.append(f"user={out['user']}")
                    if out["recommend"]:
                        parts.append(f"rec={out['recommend']}")
                    if out["click"] is not None:
                        parts.append(f"click={out['click']}")
                    if out["rating"] is not None:
                        parts.append(f"rate={out['rating']}")
                    if out["bad"] is not None:
                        parts.append(f"bad={out['bad']}")
                    print(f"[{time.strftime('%H:%M:%S')}] " + " ".join(parts))
                except Exception as e:
                    print("[WARN] worker failed:", e)
            time.sleep(delay)
            if not run_forever:
                break
    except KeyboardInterrupt:
        print("Stopping traffic generator.")
    finally:
        executor.shutdown(wait=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=20, help="number of distinct user ids to simulate")
    parser.add_argument("--concurrency", type=int, default=3, help="number of parallel requests per iteration")
    parser.add_argument("--delay", type=float, default=1.0, help="sleep seconds between batches")
    parser.add_argument("--click-p", type=float, default=0.25, help="probability to emit click after recommend")
    parser.add_argument("--rate-p", type=float, default=0.10, help="probability to emit rating after recommend")
    args = parser.parse_args()

    simulate(n_users=args.users, concurrency=args.concurrency, delay=args.delay,
             do_click_prob=args.click_p, do_rate_prob=args.rate_p)
