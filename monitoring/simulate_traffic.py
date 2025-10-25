import requests
import random
import time

API_URL = "http://localhost:8080"

def simulate_requests(n_users=10, delay=2):
    while True:
        # pick a random user_id and top_n
        user_id = random.randint(1, n_users)
        top_n = random.choice([5, 10, 20])

        try:
            r = requests.get(f"{API_URL}/recommend", params={"user_id": user_id, "top_n": top_n}, timeout=5)
            print(f"[{time.strftime('%H:%M:%S')}] {r.status_code} user={user_id}")
        except Exception as e:
            print(f"[WARN] request failed: {e}")
        time.sleep(delay)

if __name__ == "__main__":
    simulate_requests()
