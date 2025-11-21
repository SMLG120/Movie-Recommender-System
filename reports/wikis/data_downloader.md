## Downloading data using the Data Downloader

This system builds and updates the dataset for our recommender model by:
1. Streaming interaction logs from Kafka.
2. Parsing logs to extract user–movie interactions and ratings.
3. Fetching user and movie metadata via APIs.
4. Saving results into structured CSV/JSON files for training.

---

## Pipeline Flow
| Step | Component | Output |
|------|------------|---------|
| 1️⃣ | KafkaLogCollector | `event_stream.log` |
| 2️⃣ | LogParser | `watch_time.csv`, `ratings.csv` |
| 3️⃣ | MetadataFetcher | `users.csv`, `movies.csv` |
| 4️⃣ | Pipeline | Runs end-to-end |

---

## Directory Structure
```
data/
├── raw_data/
│   ├── event_stream.log
│   ├── watch_time.csv
│   ├── ratings.csv
│   ├── users.csv
│   ├── movies.csv
│   └── movies/*.json
```

---

## Components

### KafkaLogCollector
- Streams Kafka topic `movielog6`.
- Appends to `data/raw_data/event_stream.log`.

### LogParser
- Extracts `(user_id, movie_id)` pairs and ratings.
- Avoids duplicates; appends new data incrementally.

### MetadataFetcher
- Fetches user and movie metadata from APIs.
- Appends only new users or movies.

### Pipeline
- Orchestrates full process:
  1. Collect logs
  2. Parse logs
  3. Fetch metadata
  4. Extract ratings