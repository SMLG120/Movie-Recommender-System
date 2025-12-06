import os, gc
import re
import csv
import glob
import json
import time
import requests
import subprocess
import logging
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from confluent_kafka import Consumer
from typing import Iterable, Callable, List, Tuple, Dict, Any

import warnings
warnings.filterwarnings("ignore")


# 1. KafkaLogCollector — responsible for fetching logs
class KafkaLogCollector:
    def __init__(
        self,
        topic="movielog6",
        duration=60,
        flush_interval=100,
        output_dir="data",
        consumer_factory=None, 
        time_provider=None,  
    ):
        self.topic = topic
        self.duration = duration
        self.flush_interval = flush_interval
        self.log_dir = f"{output_dir}/raw_data"
        os.makedirs(self.log_dir, exist_ok=True)

        # Dependency injection points
        self._consumer_factory = consumer_factory or self._default_consumer_factory
        self._time = time_provider or time
        self.logger = logging.getLogger("KafkaLogCollector")

    def _default_consumer_factory(self):
        """Default Kafka consumer factory."""
        return Consumer({
            'bootstrap.servers': 'fall2025-comp585.cs.mcgill.ca:9092',
            'group.id': 'recsys',
            'auto.offset.reset': 'earliest'
        })

    def _handle_message(self, msg):
        """Decode and validate a Kafka message."""
        if msg.error():
            self.logger.error(f"Kafka error: {msg.error()}")
            return None
        try:
            return msg.value().decode("utf-8")
        except Exception as e:
            self.logger.warning(f"Decode error: {e}")
            return None

    def collect(self, output_log=None):
        """Stream logs from Kafka and write incrementally to disk."""
        output_log = output_log or os.path.join(self.log_dir, "event_stream.log")
        consumer = self._consumer_factory()
        consumer.subscribe([self.topic])

        start_time = self._time.time()
        processed = 0
        self.logger.info(f"Collecting logs from {self.topic} for {self.duration}s")

        with open(output_log, "w", encoding="utf-8") as f:
            while self._time.time() - start_time < self.duration:
                msg = consumer.poll(1.0)
                if msg is None:
                    continue

                line = self._handle_message(msg)
                if not line:
                    continue

                f.write(line + "\n")
                processed += 1

                if processed % self.flush_interval == 0:
                    self.logger.info(f"Processed {processed} messages...")

        consumer.close()
        del consumer
        gc.collect()
        self.logger.info(f"Finished consuming after {self.duration}s, total {processed} messages.")
        return output_log


# 2. LogParser — responsible for extracting IDs and ratings
class LogParser:
    def __init__(self, output_dir="data", file_reader=None, file_writer=None):
        """
        output_dir: directory for saving parsed data
        file_reader: injectable callable for reading file lines (for tests)
        file_writer: injectable callable for writing to CSV (for tests)
        """
        self.logger = logging.getLogger("LogParser")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.file_reader = file_reader or self._default_reader
        self.file_writer = file_writer or self._default_writer

    def _default_reader(self, path: str) -> Iterable[str]:
        with open(path, "r", encoding="utf-8") as f:
            yield from f

    def _default_writer(self, path: str, rows: List[List[str]], header: List[str] = None):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if header and write_header:
                writer.writerow(header)
            writer.writerows(rows)

    @staticmethod
    def extract_watch_event(line: str) -> Tuple[str, str, int] | None:
        """Extract user, movie, and minute from a raw log line."""
        parts = line.strip().split(",")
        if len(parts) < 3:
            return None
        user = parts[1]
        m = re.search(r"/data/m/([^/]+)/(\d+)\.mpg", line)
        if not m:
            return None
        movie, minute = m.group(1), int(m.group(2))
        return user, movie, minute

    @staticmethod
    def extract_rating_event(line: str) -> Dict[str, str] | None:
        """Extract rating info from a log line."""
        parts = line.strip().split(",")
        if len(parts) < 3:
            return None
        timestamp, user_id = parts[0], parts[1]
        m = re.search(r"GET /rate/([^=]+)=(\d+)", line)
        if not m:
            return None
        return {
            "timestamp": timestamp,
            "user_id": user_id,
            "movie_id": m.group(1),
            "rating": int(m.group(2))
        }

    def parse_logs(self, logfile: str):
        """Parse logs and return new user/movie IDs."""
        self.logger.info(f"Parsing logs from {logfile}")
        output_csv = os.path.join(self.output_dir, "raw_data/watch_time.csv")
        failed_movies_path = os.path.join(self.output_dir, "raw_data/failed_movies.txt")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        existing_pairs = set()
        if os.path.exists(output_csv):
            with open(output_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_pairs = {(r["user_id"], r["movie_id"]) for r in reader}

        with open(failed_movies_path, "r", encoding="utf-8") as f:
            failed_movies = [line.strip() for line in f if line.strip()]

        user_ids, movie_ids = set(), set()
        watch_minutes = defaultdict(lambda: defaultdict(set))

        for line in self.file_reader(logfile):
            event = self.extract_watch_event(line)
            if not event:
                continue
            user, movie, minute = event
            user_ids.add(user)
            movie_ids.add(movie)
            watch_minutes[user][movie].add(minute)

        rows = []
        new_user_ids = set()
        new_movie_ids = set()

        for user, movies in watch_minutes.items():
            for movie, minutes in movies.items():
                if (user, movie) in existing_pairs:
                    continue
                if movie in failed_movies:
                    continue
                rows.append([user, movie, len(minutes), max(minutes)])
                new_user_ids.add(user)
                new_movie_ids.add(movie)
        # for user, movies in watch_minutes.items():
        #     for movie, minutes in movies.items():
        #         if (user, movie) not in existing_pairs:
        #             rows.append([user, movie, len(minutes), max(minutes)])

        if rows:
            self.file_writer(
                output_csv, rows,
                header=["user_id", "movie_id", "interaction_count", "max_minute_reached"]
            )

        self.logger.info(
            f"Processed {len(user_ids)} users and {len(movie_ids)} movies. "
            f"Saved watch time interactions to {output_csv}"
        )
        return new_user_ids, new_movie_ids

    def parse_ratings(self, logfile: str, user_ids: set, movie_ids: set):
        """Extract ratings filtered by known users/movies."""
        self.logger.info(f"Extracting ratings from {logfile}")
        output_csv = os.path.join(self.output_dir, "raw_data/ratings.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        ratings = []
        for line in self.file_reader(logfile):
            rating = self.extract_rating_event(line)
            if rating and rating["user_id"] in user_ids and rating["movie_id"] in movie_ids:
                ratings.append(rating)

        if ratings:
            self.logger.info(f"Extracted {len(ratings)} ratings → {output_csv}")
            with open(output_csv, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp", "user_id", "movie_id", "rating"])
                # writer.writeheader()
                writer.writerows(ratings)

        # remove buplicate ratings
        ratings_df = pd.read_csv(output_csv)
        ratings_df = (
                ratings_df
                .sort_values("timestamp")
                .drop_duplicates(subset=["user_id", "movie_id"], keep="last")
            )
        ratings_df.to_csv(output_csv, index=False)

        return ratings


# 3. MetadataFetcher — fetch user/movie metadata via API
class MetadataFetcher:
    """Fetches user and movie metadata and saves structured results."""

    def __init__(
        self,
        user_api: str,
        movie_api: str,
        output_dir: str = "data",
        http_get: Callable[[str], Any] = None,
        sleep_fn: Callable[[float], None] = None,
    ):
        self.output_dir = output_dir
        self.user_api = user_api
        self.movie_api = movie_api
        self.movie_dir = os.path.join(output_dir, "raw_data/movies")
        os.makedirs(self.movie_dir, exist_ok=True)

        # injectable functions for testing
        self.http_get = http_get or requests.get
        self.sleep = sleep_fn or time.sleep

        self.logger = logging.getLogger("MetadataFetcher")

    def fetch_user(self, user_id: str) -> Dict[str, Any] | None:
        """Fetch a single user record from API."""
        try:
            r = self.http_get(self.user_api + str(user_id))
            if r.status_code == 200:
                return r.json()
            self.logger.warning(f"User {user_id} returned {r.status_code}")
        except Exception as e:
            self.logger.error(f"User fetch error {user_id}: {e}")
        return None

    def fetch_movie(self, movie_id: str, overwrite: bool = False) -> Dict[str, Any] | None:
        """Fetch and save a single movie record."""
        safe_id = movie_id.replace("/", "_")
        out_file = os.path.join(self.movie_dir, f"{safe_id}.json")

        if os.path.exists(out_file) and not overwrite:
            self.logger.debug(f"Skipping existing movie file {safe_id}")
            return None

        try:
            r = self.http_get(self.movie_api + movie_id)
            if r.status_code == 200:
                data = r.json()
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                return data
            else:
                failed_movies = "data/raw_data/failed_movies.txt"
                with open(failed_movies, "a+", encoding="utf-8") as f:
                    f.seek(0)
                    if movie_id not in {line.strip() for line in f}:
                        f.write(f"{movie_id}\n")
                self.logger.warning(f"Movie {movie_id} returned {r.status_code}")
        except Exception as e:
            self.logger.error(f"Movie fetch error {movie_id}: {e}")
        return None

    def fetch_all_users(self, user_ids: List[str], delay: float = 0.1) -> List[Dict[str, Any]]:
        """Fetch multiple users and append to CSV."""
        output_csv = os.path.join(self.output_dir, "raw_data/users.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        existing_ids = set()
        if os.path.exists(output_csv):
            existing_ids = set(pd.read_csv(output_csv)["user_id"].astype(str))
            user_ids = [uid for uid in user_ids if uid not in existing_ids]

        if not user_ids:
            self.logger.info("No new users to fetch.")
            return []

        new_data = []
        for uid in tqdm(user_ids, desc="Users"):
            data = self.fetch_user(uid)
            if data:
                new_data.append(data)
            self.sleep(delay)

        if new_data:
            df = pd.DataFrame(new_data)
            df.to_csv(
                output_csv,
                mode="a",
                index=False,
                header=not os.path.exists(output_csv),
            )
            self.logger.info(f"Appended {len(new_data)} users → {output_csv}")
        return new_data

    def fetch_all_movies(self, movie_ids: List[str], delay: float = 0.1) -> None:
        """Fetch all movies and flatten results to CSV."""
        self.logger.info(f"Fetching {len(movie_ids)} movies...")
        for mid in tqdm(movie_ids, desc="Movies"):
            self.fetch_movie(mid)
            self.sleep(delay)
        self.flatten_movies_json()

    @staticmethod
    def flatten_movie_json(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert movie JSON dict into a flat CSV-friendly dict."""
        return {
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
            "overview": (data.get("overview") or "").replace("\n", " "),
        }

    def flatten_movies_json(self) -> List[Dict[str, Any]]:
        """Flatten all movie JSON files into one CSV."""
        movie_csv = os.path.join(self.output_dir, "raw_data/movies.csv")
        json_files = glob.glob(f"{self.movie_dir}/*.json")

        movies = []
        for file in json_files:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                movies.append(self.flatten_movie_json(data))

        if not movies:
            raise ValueError("No movie JSON files found to process.")

        with open(movie_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=movies[0].keys())
            # writer.writeheader()
            writer.writerows(movies)

        self.logger.info(f"Saved {len(movies)} movies into {movie_csv}")
        return movies


# 4. DataPipeline — orchestrates the flow end-to-end
class DataPipeline:
    """End-to-end data pipeline orchestrator."""

    def __init__(
        self,
        topic: str = "movielog6",
        output_dir: str = "data",
        collector=None,
        parser=None,
        fetcher=None,
        logger=None,
    ):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Dependency injection (for tests)
        self.collector = collector
        self.parser = parser
        self.fetcher = fetcher

        # Initialization only if not provided (real runtime)
        if self.collector is None:
            # from recommender.data_download import KafkaLogCollector
            self.collector = KafkaLogCollector(topic, output_dir=self.output_dir, duration=60, flush_interval=100)
            print("Initialized KafkaLogCollector")
        if self.parser is None:
            # from recommender.log_parser import LogParser
            self.parser = LogParser(output_dir=self.output_dir)
        if self.fetcher is None:
            # from recommender.metadata import MetadataFetcher
            self.fetcher = MetadataFetcher(
                user_api="http://fall2025-comp585.cs.mcgill.ca:8080/user/",
                movie_api="http://fall2025-comp585.cs.mcgill.ca:8080/movie/",
                output_dir=self.output_dir,
            )

        self.logger = logger or logging.getLogger("Pipeline")

    def run(self):
        """Execute full pipeline; return summary dict for testing."""
        self.logger.info("Starting data pipeline run...")
        print("Starting pipeline run...")
        logfile = self.collector.collect()
        # subprocess.run(["python", "-c", "from download_data import KafkaLogCollector; KafkaLogCollector(topic, output_dir=self.output_dir, duration=900, flush_interval=100).collect()"])
        # logfile = "event_stream.log"
        self.logger.info(f"Log collection complete: {logfile}")
        print(f"Log collection complete: {logfile}")
        users, movies = self.parser.parse_logs(logfile)
        print("Log parser complete.")
        self.logger.info(f"Parsed {len(users)} users and {len(movies)} movies from logs")

        if users:
            self.fetcher.fetch_all_users(users)
        
        if movies:
            self.fetcher.fetch_all_movies(movies)
            
        self.parser.parse_ratings(logfile, users, movies)
        print("Metadata fetching and rating parsing complete.")

        self.logger.info("Pipeline complete!")
        return {"users": len(users), "movies": len(movies)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    pipeline = Pipeline(topic="movielog6")
    pipeline.run()
