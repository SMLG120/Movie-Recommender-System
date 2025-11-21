#!/usr/bin/env python3
import requests, random, time, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

API_URL = "http://localhost:8080"

def parse_recs(resp):
    """Return a list of item_ids (strings) or [] on failure."""
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

def worker(n_users, topn_choices, click_p, rate_p):
    user_id = random.randint(1, n_users)
    top_n = random.choice(topn_choices)
    chosen = None  # <-- define early

    # /recommend
    try:
        r = requests.get(f"{API_URL}/recommend",
                         params={"user_id": user_id, "top_n": top_n}, timeout=600)
        recs = parse_recs(r)
    except Exception as e:
        return f"user={user_id} rec_err={e}"

    msg = [f"user={user_id}", f"rec_count={len(recs)}", f"status={getattr(r,'status_code', 'NA')}"]

    # choose an item only if we got recs
    if recs:
        chosen = random.choice(recs)

    # /event/click (only if we have a chosen item)
    if chosen and random.random() < click_p:
        try:
            rc = requests.post(f"{API_URL}/event/click",
                               json={"user_id": user_id, "item_id": chosen, "k": len(recs)},
                               timeout=4).status_code
            msg.append(f"click={rc}")
        except Exception as e:
            msg.append(f"click_err={e}")

    # /event/rating (only if we have a chosen item)
    if chosen and random.random() < rate_p:
        true = random.choice([1,2,3,4,5])
        pred = max(1.0, min(5.0, true + random.uniform(-0.7, 0.7)))
        try:
            rc = requests.post(f"{API_URL}/event/rating",
                               json={"user_id": user_id, "item_id": chosen,
                                     "rating": true, "predicted": pred},
                               timeout=4).status_code
            msg.append(f"rate={rc}")
        except Exception as e:
            msg.append(f"rate_err={e}")

    # a few bad requests for error-rate panels
    if random.random() < 0.2:
        msg.append(f"bad={do_bad_request()}")

    return " ".join(msg)

def run(users=20, concurrency=3, delay=1.0, click_p=0.25, rate_p=0.10):
    topn_choices = [5, 10, 20]
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        while True:
            futures = [pool.submit(worker, users, topn_choices, click_p, rate_p) for _ in range(concurrency)]
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
    ap.add_argument("--rate-p", type=float, default=0.10)
    args = ap.parse_args()
    run(args.users, args.concurrency, args.delay, args.click_p, args.rate_p)
