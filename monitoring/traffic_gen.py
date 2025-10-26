#!/usr/bin/env python3
import argparse, random, time, json
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

def recommend(base, user_id, top_n=10, timeout=2.0):
    r = requests.get(f"{base}/recommend", params={"user_id": user_id, "top_n": top_n}, timeout=timeout)
    try:
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
    except Exception:
        data = {}
    recs = data.get("recommendations") or []
    return r.status_code, recs

def click(base, user_id, item_id, timeout=2.0):
    return requests.post(f"{base}/event/click",
                         json={"user_id": user_id, "item_id": item_id},
                         timeout=timeout).status_code

def rating(base, user_id, item_id, timeout=2.0):
    # Simulate realism: true rating 2–5, predicted with noise
    true_rating = random.choice([2.0, 3.0, 4.0, 5.0])
    predicted   = max(1.0, min(5.0, true_rating + random.uniform(-0.8, 0.8)))
    return requests.post(f"{base}/event/rating",
                         json={"user_id": user_id, "item_id": item_id,
                               "rating": true_rating, "predicted": predicted},
                         timeout=timeout).status_code

def bad_request(base, timeout=2.0):
    # 400 (missing user_id) or 404 (/no-such-route) to drive Error Rate panel
    if random.random() < 0.6:
        return requests.get(f"{base}/recommend", timeout=timeout).status_code  # 400
    else:
        return requests.get(f"{base}/no-such-route", timeout=timeout).status_code  # 404

def worker(base, users, topn, stop_time):
    while time.time() < stop_time:
        u = random.choice(users)
        # 10% intentionally bad to create 4xx/404 for "HTTP Error Rate (%)"
        if random.random() < 0.10:
            try: bad_request(base)
            except Exception: pass
            time.sleep(random.uniform(0.02, 0.12))
            continue

        # Normal traffic path
        try:
            sc, recs = recommend(base, u, topn)
            if recs:
                # ~30% of the time, simulate a click on a served item
                if random.random() < 0.30:
                    item = random.choice(recs)
                    try: click(base, u, item)
                    except Exception: pass

                # ~40% of the time, report a rating to drive MAE/RMSE counters
                if random.random() < 0.40:
                    item = random.choice(recs)
                    try: rating(base, u, item)
                    except Exception: pass
        except Exception:
            pass

        # short think time so we generate steady load
        time.sleep(random.uniform(0.02, 0.10))

def main():
    ap = argparse.ArgumentParser(description="Traffic generator for Movie Recommender API")
    ap.add_argument("--base", default="http://localhost:8080", help="Base URL of API")
    ap.add_argument("--users", type=int, default=30, help="Number of synthetic users (1..N)")
    ap.add_argument("--concurrency", type=int, default=8, help="Threads (controls overall RPS)")
    ap.add_argument("--duration", type=int, default=180, help="Run time in seconds")
    ap.add_argument("--topn", type=int, default=10, help="top_n for /recommend")
    args = ap.parse_args()

    users = list(range(1, args.users + 1))
    stop_time = time.time() + args.duration

    print(f"Starting traffic → {args.base}  | users=1..{args.users}  | threads={args.concurrency}  | duration={args.duration}s")
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(worker, args.base, users, args.topn, stop_time) for _ in range(args.concurrency)]
        for f in as_completed(futures):
            pass
    print("Done.")

if __name__ == "__main__":
    main()
