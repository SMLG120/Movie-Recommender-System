# Movie Recommendations – COMP 585

**Team 6**  
Samuel Ha · Sanghyun Hong · Jessica Ojo · Eric Qiu

---

## Overview

This project implements a **movie recommendation system** for a simulated streaming service with approximately **1 million customers** and **27,000 movies**.

Across multiple milestones, we designed, deployed, and evaluated a **production-ready recommendation service** that interacts with provided APIs and Kafka event streams. The system ingests user activity logs (movie requests, ratings, and recommendation requests) and returns **personalized movie recommendations** via an inference service.

---

## Project Structure

.
├── src/ # Source code: data prep, features, training, inference
│ ├── experiments/ # Model experiments and analysis scripts
│ ├── configs.py # Global configuration and constants
│ ├── feature_builder.py
│ ├── trainer.py # Training pipeline entrypoint
│ ├── inference.py # Recommender inference service
│ ├── download_data.py
│ └── inference_readme.md
│
├── data/
│ ├── raw_data/ # Original CSV files
│ └── prepared data, ID lists
│
├── models/ # Saved trained models
├── train_results/ # JSON outputs from training and tuning
├── reports/ # Reports and meeting notes
├── docker/ # Dockerfile for containerization
└── requirements.txt # Python dependencies

yaml
Copier le code

---

## Key Files

- `requirements.txt` — Python dependencies  
- `src/configs.py` — Global configuration and constants  
- `src/feature_builder.py` — Feature extraction and preprocessing  
- `src/trainer.py` — Training pipeline entrypoint  
- `src/inference.py` — Inference service and recommender logic  
- `src/download_data.py` — Data preparation utilities  
- `src/inference_readme.md` — Notes on inference usage  
- `src/experiments/` — Experimental training and analysis scripts (XGBoost, MLP, logistic regression)

---

## Quick Start

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
2. Upgrade pip and install dependencies
bash
Copier le code
pip install --upgrade pip
pip install -r requirements.txt
3. Prepare the data
Place the raw CSV files under:

text
Copier le code
data/raw_data/
If needed, see src/download_data.py for helper utilities.

How to Run
Train the Default Pipeline
bash
Copier le code
source .venv/bin/activate
python src/trainer.py
Run Experiment Scripts
Scripts under src/experiments/ evaluate different models and require command-line arguments.

XGBoost
bash
Copier le code
python src/experiments/train_model_xgb.py <ratings_csv> <out_model>
MLP
bash
Copier le code
python src/experiments/train_model_mlp.py <ratings_csv> <out_model>
Logistic Regression
bash
Copier le code
python src/experiments/train_model_logistic_regression.py <ratings_csv> <out_model>
Inference
src/inference.py defines a RecommenderEngine class and includes a runnable example under:

python
Copier le code
if __name__ == "__main__":
Current Behavior
Running:

bash
Copier le code
python src/inference.py
Loads the model from src/models/xgb_recommender.joblib

Reads movie metadata from data/raw_data/movies.csv

Runs a hard-coded example (user_id = 13262)

Prints a comma-separated list of recommended movie_ids

Example Usage
python
Copier le code
from src.inference import RecommenderEngine

engine = RecommenderEngine(
    model_path='src/models/xgb_recommender.joblib',
    movies_file='data/raw_data/movies.csv',
    mode='dev'
)

print(engine.recommend(12345, top_n=10))
Docker
Build and run the container using the provided Dockerfile:

bash
Copier le code
docker build -f docker/Dockerfile -t movie-recommender:v1.0 .
Example run with log rotation and port mapping:

bash
Copier le code
docker run -it \
  --log-opt max-size=50m \
  --log-opt max-file=5 \
  -p 8080:8080 \
  movie-recommender:v1.0
Monitoring Stack
To start the monitoring services:

bash
Copier le code
cd monitoring
docker compose up --build
This hosts:

Recommender API — http://localhost:8080

Prometheus — http://localhost:9090

Grafana — http://localhost:3000

Grafana login

Username: admin

Password: admin

Import the dashboard from:

text
Copier le code
monitoring/grafana-dashboard.json
Models & Artifacts
Trained models: src/models/

Example: xgb_recommender.joblib, xgb_recommender.pkl

Training & tuning results: src/train_results/ (JSON files)

Load a Saved Model
python
Copier le code
import joblib

model = joblib.load('src/models/xgb_recommender.joblib')
Ethical Considerations
The team acknowledges ethical considerations related to recommendation systems. No personally identifiable information (PII) is collected or shared beyond the provided simulated dataset.

We also recognize that recommendation algorithms can amplify popularity bias. This system is developed solely for COMP 585 evaluation purposes and is not intended for real-world deployment.

