import os
import sys
import tempfile
import json
import pytest
from unittest.mock import MagicMock
# Add project root to path for src imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from src.download_data import KafkaLogCollector, LogParser, MetadataFetcher, DataPipeline


@pytest.fixture
def temp_dir(tmp_path):
    """Temporary directory for integration outputs."""
    return str(tmp_path)


# KafkaLogCollector Integration
def test_kafka_logcollector_writes_file(temp_dir):
    """Integration test: KafkaLogCollector writes log file correctly."""

    def poll_generator():
        msg = MagicMock()
        msg.value = lambda: b"2025-10-01,user1,GET /data/m/MovieX/12.mpg"
        msg.error.return_value = None 
        yield msg
        while True:
            yield None

    fake_consumer = MagicMock()
    fake_consumer.poll.side_effect = poll_generator()
    fake_consumer.close = MagicMock()

    collector = KafkaLogCollector(
        topic="test_topic",
        duration=1,
        consumer_factory=lambda: fake_consumer,
        output_dir=temp_dir,
    )

    output_file = collector.collect()

    assert os.path.exists(output_file)
    content = open(output_file).read().strip()
    assert "MovieX" in content, f"Expected 'MovieX' in log, got: {content!r}"
    fake_consumer.close.assert_called_once()


# LogParser Integration
def test_logparser_extracts_watch_and_ratings(temp_dir):
    log_path = os.path.join(temp_dir, "event.log")
    lines = [
        "2025-10-01,user1,GET /data/m/MovieX/12.mpg",
        "2025-10-01,user1,GET /data/m/MovieX/13.mpg",
        "2025-10-01,user1,GET /rate/MovieX=4"
    ]
    with open(log_path, "w") as f:
        f.write("\n".join(lines))

    parser = LogParser(output_dir=temp_dir)
    users, movies = parser.parse_logs(log_path)
    assert "user1" in users
    assert "MovieX" in movies

    ratings = parser.parse_ratings(log_path, users, movies)
    assert ratings[0]["rating"] == 4
    assert ratings[0]["movie_id"] == "MovieX"


def test_metadatafetcher_user_and_movie_integration(temp_dir):
    # Mock user response
    fake_user = {"user_id": "user1", "age": 30, "occupation": "engineer", "gender": "M"}
    user_response = MagicMock(status_code=200)
    user_response.json.return_value = fake_user

    # Mock movie response
    fake_movie = {
        "id": "MovieA",
        "title": "The Great Movie",
        "original_language": "en",
        "release_date": "2020-01-01",
        "runtime": 100,
        "popularity": 7.5,
        "vote_average": 8.0,
        "vote_count": 120,
        "genres": [{"name": "Drama"}],
        "spoken_languages": [{"name": "English"}],
        "production_countries": [{"iso_3166_1": "US"}],
        "overview": "A test movie.",
    }
    movie_response = MagicMock(status_code=200)
    movie_response.json.return_value = fake_movie

    # Mock http_get that returns proper response depending on URL
    def fake_http_get(url):
        if "user" in url:
            return user_response
        elif "movie" in url:
            return movie_response
        raise ValueError(f"Unexpected URL: {url}")

    fetcher = MetadataFetcher(
        user_api="http://mockserver/user/",
        movie_api="http://mockserver/movie/",
        output_dir=temp_dir,
        http_get=fake_http_get,
    )

    users = fetcher.fetch_all_users(["user1"])
    user_csv = os.path.join(temp_dir, "raw_data", "users.csv")
    assert os.path.exists(user_csv)
    df_users = open(user_csv).read()
    assert "user1" in df_users and "engineer" in df_users

    fetcher.fetch_all_movies(["MovieA"])
    movie_json = os.path.join(temp_dir, "raw_data/movies/MovieA.json")
    assert os.path.exists(movie_json)

    # Flatten JSONs into CSV
    fetcher.flatten_movies_json()
    movie_csv = os.path.join(temp_dir, "raw_data/movies.csv")
    assert os.path.exists(movie_csv)
    df_movies = open(movie_csv).read()
    assert "The Great Movie" in df_movies
    assert "Drama" in df_movies

def test_metadatafetcher_handles_failures_gracefully(temp_dir):
    """Ensure MetadataFetcher doesn't crash on 404s or bad responses."""
    bad_user_response = MagicMock(status_code=404)
    bad_movie_response = MagicMock(status_code=500)
    broken_json_response = MagicMock(status_code=200)
    broken_json_response.json.side_effect = ValueError("Malformed JSON")

    def fake_http_get(url):
        if "user" in url:
            return bad_user_response
        elif "movie404" in url:
            return bad_movie_response
        elif "movie_badjson" in url:
            return broken_json_response
        raise ValueError(f"Unexpected URL: {url}")

    fetcher = MetadataFetcher(
        user_api="http://mockserver/user/",
        movie_api="http://mockserver/movie/",
        output_dir=temp_dir,
        http_get=fake_http_get,
    )

    user_result = fetcher.fetch_user("user404")
    assert user_result is None

    result_404 = fetcher.fetch_movie("movie404")
    assert result_404 is None

    result_bad = fetcher.fetch_movie("movie_badjson")
    assert result_bad is None

    with pytest.raises(ValueError, match="No movie JSON files found"):
        fetcher.flatten_movies_json()


# DataPipeline Integration
def test_pipeline_end_to_end(temp_dir):
    # Fake collector writes a local log
    fake_collector = MagicMock()
    fake_log = os.path.join(temp_dir, "pipeline.log")
    with open(fake_log, "w") as f:
        f.write("2025-10-01,user1,GET /data/m/MovieZ/10.mpg\n")
        f.write("2025-10-01,user1,GET /rate/MovieZ=5\n")
    fake_collector.collect.return_value = fake_log

    # Real parser (to test integration)
    parser = LogParser(output_dir=temp_dir)

    # Fake fetcher (skip API)
    fake_fetcher = MagicMock()
    fake_fetcher.fetch_all_users.return_value = [{"user_id": "user1"}]
    fake_fetcher.fetch_all_movies.return_value = None

    pipeline = DataPipeline(
        output_dir=temp_dir,
        collector=fake_collector,
        parser=parser,
        fetcher=fake_fetcher
    )

    summary = pipeline.run()

    assert summary["users"] == 1
    assert summary["movies"] == 1
    assert os.path.exists(os.path.join(temp_dir, "raw_data", "watch_time.csv"))
    assert os.path.exists(os.path.join(temp_dir, "raw_data", "ratings.csv"))
