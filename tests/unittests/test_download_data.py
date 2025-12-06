import os
import io
import json
import csv
import pytest
from unittest.mock import MagicMock
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

from src.download_data import KafkaLogCollector, LogParser, MetadataFetcher, DataPipeline


# KafkaLogCollector Tests
def test_handle_message_valid_and_error(caplog):
    collector = KafkaLogCollector()
    msg = MagicMock()
    msg.error.return_value = None
    msg.value.return_value = b"test line"
    assert collector._handle_message(msg) == "test line"

    # error path
    msg.error.return_value = "bad"
    assert collector._handle_message(msg) is None
    assert "Kafka error" in caplog.text


def test_collect_writes_mocked_messages(tmp_path):
    fake_msg = MagicMock()
    fake_msg.error.return_value = None
    fake_msg.value.return_value = b"hi"

    fake_consumer = MagicMock()
    fake_consumer.poll.side_effect = [fake_msg] + [None] * 10  # no StopIteration
    fake_time = MagicMock()
    fake_time.time.side_effect = [0, 0.5, 1.5]  # loop exits after ~1.5s

    collector = KafkaLogCollector(
        duration=1,
        consumer_factory=lambda: fake_consumer,
        time_provider=fake_time,
        output_dir=str(tmp_path),
    )

    output_log = collector.collect()
    content = Path(output_log).read_text().strip()
    assert "hi" in content
    fake_consumer.close.assert_called_once()


# LogParser Tests
def test_extract_watch_event_valid():
    line = "t1,user1,/data/m/movieA/10.mpg"
    assert LogParser.extract_watch_event(line) == ("user1", "movieA", 10)

def test_extract_watch_event_invalid():
    assert LogParser.extract_watch_event("garbage") is None

def test_extract_rating_event_valid():
    line = "t1,user1,GET /rate/movieX=5"
    rating = LogParser.extract_rating_event(line)
    assert rating["movie_id"] == "movieX" and rating["rating"] == 5

def test_parse_logs_and_file_write(tmp_path):
    lines = [
        "t1,u1,/data/m/m1/1.mpg",
        "t2,u1,/data/m/m1/2.mpg",
        "t3,u2,/data/m/m2/1.mpg",
    ]

    # ensure failed_movies.txt exists
    failed = tmp_path / "raw_data/failed_movies.txt"
    failed.parent.mkdir(parents=True, exist_ok=True)
    failed.write_text("")

    parser = LogParser(output_dir=str(tmp_path), file_reader=lambda _: lines)
    users, movies = parser.parse_logs("fake.log")

    assert users == {"u1", "u2"}
    assert movies == {"m1", "m2"}

    output_csv = tmp_path / "raw_data/watch_time.csv"
    assert output_csv.exists()

    with open(output_csv) as f:
        rows = list(csv.DictReader(f))

    assert {"user_id", "movie_id", "interaction_count", "max_minute_reached"} <= set(rows[0].keys())


def test_parse_ratings_filters_known_entities(tmp_path):
    lines = [
        "t1,u1,GET /rate/m1=4",
        "t2,u2,GET /rate/m2=5",
    ]

    # ensure ratings.csv exists before pandas reads it
    ratings_csv = tmp_path / "raw_data/ratings.csv"
    ratings_csv.parent.mkdir(parents=True, exist_ok=True)
    ratings_csv.write_text("timestamp,user_id,movie_id,rating\n")

    parser = LogParser(output_dir=str(tmp_path), file_reader=lambda _: lines)
    ratings = parser.parse_ratings("fake.log", {"u1"}, {"m1"})

    assert len(ratings) == 1
    assert ratings[0]["user_id"] == "u1"
    assert ratings[0]["movie_id"] == "m1"

# MetadataFetcher Tests
def test_fetch_user_success():
    fake_resp = MagicMock(status_code=200, json=lambda: {"user_id": "u1"})
    fetcher = MetadataFetcher("u/", "m/", http_get=lambda url: fake_resp)
    result = fetcher.fetch_user("u1")
    assert result == {"user_id": "u1"}

def test_fetch_movie_saves_json(tmp_path):
    fake_resp = MagicMock(status_code=200, json=lambda: {"id": "m1"})
    fetcher = MetadataFetcher("u/", "m/", http_get=lambda url: fake_resp, output_dir=str(tmp_path))
    fetcher.fetch_movie("m1")
    files = list((tmp_path / "raw_data/movies").glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["id"] == "m1"

def test_flatten_movie_json_simplifies_data():
    data = {"id": "m1", "genres": [{"name": "Drama"}]}
    flat = MetadataFetcher.flatten_movie_json(data)
    assert flat["genres"] == "Drama"

def test_flatten_movies_json_reads_jsons(tmp_path):
    mdir = tmp_path / "raw_data/movies"
    mdir.mkdir(parents=True)
    (mdir / "a.json").write_text('{"id": "a", "title": "A"}')
    (mdir / "b.json").write_text('{"id": "b", "title": "B"}')
    fetcher = MetadataFetcher("u/", "m/", output_dir=str(tmp_path))
    result = fetcher.flatten_movies_json()
    assert len(result) == 2

# Download data Pipeline Tests
def test_pipeline_orchestration(tmp_path):
    fake_collector = MagicMock()
    fake_collector.collect.return_value = "fake.log"

    fake_parser = MagicMock()
    fake_parser.parse_logs.return_value = ({"u1"}, {"m1"})
    fake_parser.parse_ratings.return_value = None

    fake_fetcher = MagicMock()

    pipeline = DataPipeline(
        output_dir=str(tmp_path),
        collector=fake_collector,
        parser=fake_parser,
        fetcher=fake_fetcher,
    )

    result = pipeline.run()
    fake_collector.collect.assert_called_once()
    fake_parser.parse_logs.assert_called_once_with("fake.log")
    fake_fetcher.fetch_all_users.assert_called_once_with({"u1"})
    fake_fetcher.fetch_all_movies.assert_called_once_with({"m1"})
    fake_parser.parse_ratings.assert_called_once_with("fake.log", {"u1"}, {"m1"})
    assert result == {"users": 1, "movies": 1}
