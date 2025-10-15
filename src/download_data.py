import os
import re
import csv
import glob
import json
import time
import requests
import logging
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from confluent_kafka import Consumer


# ---------------------------------------------
# 1. KafkaLogCollector — responsible for fetching logs
# ---------------------------------------------
class KafkaLogCollector:
    def __init__(self, topic="movielog6", duration=600, flush_interval=100, output_dir="data"):
        self.topic = topic
        self.duration = duration
        self.flush_interval = flush_interval
        self.log_dir = f"{output_dir}/raw_data" # store raw logs
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = logging.getLogger("KafkaLogCollector")

    def collect(self, output_log=None):
        """Stream logs from Kafka and write incrementally to disk."""
        output_log = output_log or os.path.join(self.log_dir, "event_stream.log")
        consumer = Consumer({
            'bootstrap.servers': 'fall2025-comp585.cs.mcgill.ca:9092',
            'group.id': 'recsys',
            'auto.offset.reset': 'earliest'
        })
        consumer.subscribe([self.topic])

        start_time = time.time()
        processed = 0
        self.logger.info(f"Collecting logs from {self.topic} for {self.duration}s")

        with open(output_log, "a", encoding="utf-8") as f:
            while time.time() - start_time < self.duration:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    self.logger.error(f"Kafka error: {msg.error()}")
                    continue
                line = msg.value().decode("utf-8")
                f.write(line + "\n")
                processed += 1

                if processed % self.flush_interval == 0:
                    self.logger.info(f"Processed {processed} messages...")

        consumer.close()
        self.logger.info(f"Finished consuming after {self.duration}s, total {processed} messages.")
        return output_log


# ---------------------------------------------
# 2. LogParser — responsible for extracting IDs and ratings
# ---------------------------------------------
class LogParser:
    def __init__(self, output_dir="data"):
        self.logger = logging.getLogger("LogParser")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def parse_logs(self, logfile):
        """Parse Kafka logs, append only new user-movie pairs, and return new IDs."""
        self.logger.info(f"Parsing logs from {logfile}")
        output_csv = os.path.join(self.output_dir, "raw_data/watch_time.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        existing_pairs = set()
        if os.path.exists(output_csv):
            with open(output_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_pairs = {(r["user_id"], r["movie_id"]) for r in reader}

        user_ids, movie_ids = set(), set()
        watch_minutes = defaultdict(lambda: defaultdict(set))

        with open(logfile, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                user = parts[1]
                user_ids.add(user)
                m = re.search(r"/data/m/([^/]+)/(\d+)\.mpg", line)
                if not m:
                    continue
                movie, minute = m.group(1), int(m.group(2))
                movie_ids.add(movie)
                watch_minutes[user][movie].add(minute)

        write_header = not os.path.exists(output_csv)
        with open(output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["user_id", "movie_id", "interaction_count", "max_minute_reached"])
            for user, movies in watch_minutes.items():
                for movie, minutes in movies.items():
                    if (user, movie) in existing_pairs:
                        continue
                    writer.writerow([user, movie, len(minutes), max(minutes)])

        self.logger.info(
            f"Processed {len(user_ids)} users and {len(movie_ids)} movies. \nSaved watch time interaction to {output_csv}"
        )
        return user_ids, movie_ids

    def parse_ratings(self, logfile, user_ids, movie_ids):
        """Extract ratings interactions filtered by known users/movies."""
        self.logger.info(f"Extracting ratings from {logfile}")
        output_csv = os.path.join(self.output_dir, "raw_data/ratings.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        ratings = []
        pattern = re.compile(r"GET /rate/([^=]+)=(\d+)")

        with open(logfile, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                timestamp, user_id = parts[0], parts[1]
                m = pattern.search(line)
                if m:
                    movie_id = m.group(1)
                    rating = int(m.group(2))
                    if user_id in user_ids and movie_id in movie_ids:
                        ratings.append({
                            "timestamp": timestamp,
                            "user_id": user_id,
                            "movie_id": movie_id,
                            "rating": rating
                        })

        if ratings:
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
            with open(output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "user_id", "movie_id", "rating"])
                writer.writeheader()
                writer.writerows(ratings)
            self.logger.info(f"Extracted {len(ratings)} ratings into {output_csv}")
        return ratings


# ---------------------------------------------
# 3. MetadataFetcher — fetch user/movie metadata via API
# ---------------------------------------------
class MetadataFetcher:
    def __init__(self, user_api, movie_api, output_dir="data"):
        self.output_dir = output_dir
        self.user_api = user_api
        self.movie_api = movie_api
        self.movie_dir = os.path.join(output_dir, "raw_data/movies")
        os.makedirs(self.movie_dir, exist_ok=True)

        self.logger = logging.getLogger("MetadataFetcher")

    def fetch_user(self, user_id):
        try:
            r = requests.get(self.user_api + str(user_id), timeout=5)
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            self.logger.error(f"User fetch error {user_id}: {e}")
            return None

    def fetch_movie(self, movie_id, overwrite=False):
        safe_id = movie_id.replace("/", "_")
        out_file = os.path.join(self.movie_dir, f"{safe_id}.json")
        if os.path.exists(out_file) and not overwrite:
            return
        try:
            r = requests.get(self.movie_api + movie_id, timeout=5)
            if r.status_code == 200:
                with open(out_file, "a", encoding="utf-8") as f:
                    json.dump(r.json(), f)
        except Exception as e:
            self.logger.error(f"Movie fetch error {movie_id}: {e}")


    def fetch_all_users(self, user_ids, delay=0.1):
        """Fetch users and append to users.csv."""
        output_csv = os.path.join(self.output_dir, "raw_data/users.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        existing_ids = set()
        if os.path.exists(output_csv):
            existing_ids = set(pd.read_csv(output_csv)["user_id"].astype(str))
            user_ids = [uid for uid in user_ids if uid not in existing_ids]

        if not user_ids:
            self.logger.info("No new users to fetch.")
            return

        self.logger.info(f"Fetching {len(user_ids)} users...")
        new_data = []
        for uid in tqdm(user_ids, desc="Users"):
            user_data = self.fetch_user(uid)
            if user_data:
                new_data.append(user_data)
            time.sleep(delay)

        if new_data:
            pd.DataFrame(new_data).to_csv(
                output_csv, mode="a", index=False, header=not os.path.exists(output_csv)
            )
            self.logger.info(f"Appended {len(new_data)} users → {output_csv}")

    def fetch_all_movies(self, movie_ids, delay=0.1):
        self.logger.info(f"Fetching {len(movie_ids)} movies...")
        for mid in tqdm(movie_ids, desc="Movies"):
            self.fetch_movie(mid)
            time.sleep(delay)
        self.flatten_movies_json()

    def flatten_movies_json(self):
        """Flatten all movie JSONs in a folder into a single CSV."""
        movie_csv = os.path.join(self.output_dir, "raw_data/movies.csv")
        movies = []

        for file in glob.glob(f"{self.movie_dir}/*.json"):
            with open(file, "r", encoding="utf-8") as f:
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

        with open(movie_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=movies[0].keys())
            writer.writeheader()
            writer.writerows(movies)

        self.logger.info(f"Saved {len(movies)} movies into {movie_csv}")


# ---------------------------------------------
# 4. Pipeline — orchestrates the flow end-to-end
# ---------------------------------------------
class Pipeline:
    def __init__(self, topic="movielog6", output_dir="data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.collector = KafkaLogCollector(topic, output_dir=self.output_dir, duration=60)
        self.parser = LogParser(output_dir=self.output_dir)
        self.fetcher = MetadataFetcher(
            user_api="http://fall2025-comp585.cs.mcgill.ca:8080/user/",
            movie_api="http://fall2025-comp585.cs.mcgill.ca:8080/movie/",
            output_dir=self.output_dir
        )

    def run(self):
        logfile = self.collector.collect()
        users, movies = self.parser.parse_logs(logfile)
        self.fetcher.fetch_all_users(users)
        self.fetcher.fetch_all_movies(movies)
        self.parser.parse_ratings(logfile, users, movies)
        print("Pipeline complete!")


# ---------------------------------------------
# Entry point
# ---------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pipeline = Pipeline(topic="movielog6")
    pipeline.run()
