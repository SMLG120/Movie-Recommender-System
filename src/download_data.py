import os
import re
import csv
import time
import json
import argparse
import requests
import logging
from tqdm import tqdm
from collections import defaultdict


class DataDownloader:
    def __init__(self, logfile=None,
                 user_file="user_ids.txt", movie_file="movie_ids.txt",
                 user_dir="users", movie_dir="movies",
                 base_user="http://fall2025-comp585.cs.mcgill.ca:8080/user/",
                 base_movie="http://fall2025-comp585.cs.mcgill.ca:8080/movie/",
                 loglevel=logging.INFO):

        # Configure logger
        logging.basicConfig(
            level=loglevel,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger("DataDownloader")

        self.logfile = logfile
        self.user_file = user_file
        self.movie_file = movie_file

        # IDs and watch_time
        self.user_ids = []
        self.movie_ids = []
        self.watch_time = defaultdict(lambda: defaultdict(int))

        # API endpoints
        self.base_user = base_user
        self.base_movie = base_movie

        # movie info storage dir
        self.movie_dir = movie_dir
        os.makedirs(movie_dir, exist_ok=True)

        if os.path.exists(self.user_file) and os.path.exists(self.movie_file):
            self.user_ids = self._load_ids(self.user_file)
            self.movie_ids = self._load_ids(self.movie_file)
            self.logger.info(f"Loaded {len(self.user_ids)} users and {len(self.movie_ids)} movies from files.")
        elif self.logfile:
            self.logger.info("ID files not found. Parsing logs instead...")
            self.parse_watch_logs()
        else:
            self.logger.warning("No ID files or logfile provided. Downloader initialized empty.")

    def _load_ids(self, filepath):
        with open(filepath) as f:
            return [line.strip() for line in f if line.strip()]


    def parse_watch_logs(self):
        """Parse the log file to extract user IDs, movie IDs, and watch times."""
        if not self.logfile or not os.path.exists(self.logfile):
            raise FileNotFoundError(f"Log file {self.logfile} not found")

        user_ids = set()
        movie_ids = set()
        watch_time = defaultdict(lambda: defaultdict(int))

        with open(self.logfile, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue

                user = parts[1]
                user_ids.add(user)

                m = re.search(r"/data/m/([^/]+)/(\d+)\.mpg", line)
                if m:
                    movie = m.group(1)
                    minute = int(m.group(2))
                    movie_ids.add(movie)
                    watch_time[user][movie] += 1

        # Save to instance
        self.user_ids = list(user_ids)
        self.movie_ids = list(movie_ids)
        self.watch_time = watch_time

        # Save to files
        with open(self.user_file, "w") as uf:
            uf.write("\n".join(self.user_ids))
        with open(self.movie_file, "w") as mf:
            mf.write("\n".join(self.movie_ids))

        self.logger.info(f"Parsed {len(self.user_ids)} users and {len(self.movie_ids)} movies from logs.")

    def fetch_user(self, user_id):
        """Fetch a single user's metadata and return as dict."""
        try:
            r = requests.get(self.base_user + str(user_id), timeout=5)
            if r.status_code == 200:
                return r.json()
            else:
                self.logger.error(f"Failed user {user_id}: {r.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"Exception fetching user {user_id}: {e}")
            return None

    def fetch_movie(self, movie_id, overwrite=False, fail_log="failed_movies.txt"):
        """Fetch and save movie metadata."""
        safe_id = movie_id.replace("/", "_")
        out_file = os.path.join(self.movie_dir, f"{safe_id}.json")
        if os.path.exists(out_file) and not overwrite:
            return
        try:
            r = requests.get(self.base_movie + movie_id, timeout=5)
            if r.status_code == 200:
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(r.json(), f)
                self.logger.debug(f"Saved movie {movie_id}.json")
            else:
                self.logger.error(f"Failed movie {movie_id}: {r.status_code}")
                with open(fail_log, "a") as f:   # append failed ID
                    f.write(f"{movie_id}\n")
        except Exception as e:
            self.logger.error(f"Exception fetching movie {movie_id}: {e}")
            with open(fail_log, "a") as f:       # append failed ID
                f.write(f"{movie_id}\n")

    def fetch_all_users(self, delay=0.1, output="users.csv", batch_size=1000):
        """Fetch all users and save into one CSV with batch flushing."""
        self.logger.info(f"Fetching {len(self.user_ids)} users...")

        self.logger.info(f"Opening {output} for writing...")
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["user_id", "age", "occupation", "gender"])
            writer.writeheader()

            batch = []
            for i, uid in enumerate(tqdm(self.user_ids, desc="Users", unit="user"), 1):
                user_data = self.fetch_user(uid)
                if user_data:
                    batch.append(user_data)

                if i % batch_size == 0:
                    writer.writerows(batch)
                    f.flush()
                    batch = []
                    self.logger.info(f"Processed {i} users...")

                time.sleep(delay)

            # Write any leftovers
            if batch:
                writer.writerows(batch)
                f.flush()

        self.logger.info(f"Saved all users into {output}")


    def fetch_all_movies(self, delay=0.1, overwrite=False):
        self.logger.info(f"Fetching {len(self.movie_ids)} movies...")
        for mid in tqdm(self.movie_ids):
            self.fetch_movie(mid, overwrite=overwrite)
            time.sleep(delay)

    def fetch_all(self, delay=0.1, overwrite=False):
        self.logger.info(f"Fetching {len(self.user_ids)} users and {len(self.movie_ids)} movies...")
        self.fetch_all_users(delay, output="users.csv")
        self.fetch_all_movies(delay, overwrite)
        self.logger.info("Download complete.")

    def flatten_movies_json(self, movies_folder="movies", movies_file="movies.csv"):
        """Flatten all movie JSONs in a folder into a single CSV."""
        movies = []

        for file in os.listdir(movies_folder):
            if not file.endswith(".json"):
                continue
            with open(os.path.join(movies_folder, file), "r", encoding="utf-8") as f:
                data = json.load(f)

                movie = {
                    "id": data.get("id"),
                    "title": data.get("title"),
                    "original_language": data.get("original_language"),
                    "release_date": data.get("release_date"),
                    "runtime": data.get("runtime"),
                    "popularity": data.get("popularity"),
                    "vote_average": data.get("vote_average"),
                    "vote_count": data.get("vote_count"),
                    "genres": ",".join([g.get("name", "") for g in data.get("genres", [])]),
                    "spoken_languages": ",".join([l.get("name", "") for l in data.get("spoken_languages", [])]),
                    "production_countries": ",".join([c.get("iso_3166_1", "") for c in data.get("production_countries", [])]),
                    "overview": (data.get("overview") or "").replace("\n", " ")
                }
                movies.append(movie)

        if not movies:
            raise ValueError("No movie JSON files found to process.")

        with open(movies_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=movies[0].keys())
            writer.writeheader()
            writer.writerows(movies)

        print(f"[INFO] Saved {len(movies)} movies into {movies_file}")
        

    def parse_ratings_logs(self, logfile=None, output="ratings.csv"):
        """Parse ratings from logs and save into CSV, filtered by known users/movies."""
        logfile = logfile or self.logfile
        if not logfile or not os.path.exists(logfile):
            raise FileNotFoundError(f"Log file {logfile} not found")

        ratings = []
        pattern = re.compile(r"GET /rate/([^=]+)=(\d+)")

        with open(logfile, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue

                timestamp = parts[0]
                user_id = parts[1]

                m = pattern.search(line)
                if m:
                    movie_id = m.group(1)
                    rating = int(m.group(2))

                    if (not self.user_ids or user_id in self.user_ids) and \
                       (not self.movie_ids or movie_id in self.movie_ids):
                        ratings.append({
                            "timestamp": timestamp,
                            "user_id": user_id,
                            "movie_id": movie_id,
                            "rating": rating
                        })

        if ratings:
            with open(output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "user_id", "movie_id", "rating"])
                writer.writeheader()
                writer.writerows(ratings)
            self.logger.info(f"Extracted {len(ratings)} ratings into {output}")
        else:
            self.logger.warning("No ratings found in log.")

        return ratings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse logs and download user/movie metadata")
    parser.add_argument("--logfile", type=str, default="data/event_stream.log", help="Path to event stream log file")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between API calls (seconds)")
    parser.add_argument("--loglevel", type=str, default="INFO", help="Logging level (DEBUG, INFO, ERROR)")
    parser.add_argument("--user_file", type=str, default="data/user_ids.txt", help="Unique user IDs file path")
    parser.add_argument("--movie_file", type=str, default="data/movie_ids.txt", help="Unique movie IDs file path")
    args = parser.parse_args()

    loglevel = getattr(logging, args.loglevel.upper(), logging.INFO)

    dl = DataDownloader(logfile=args.logfile, loglevel=loglevel)
    # dl.parse_watch_logs()
    # dl.fetch_all(delay=args.delay)
    # dl.flatten_movies_json()
    ratings = dl.parse_ratings_logs(output="data/ratings.csv")
