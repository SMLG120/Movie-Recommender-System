## Movie Recommendations \- Comp585

Team 6: Samuel Ha, Sanghyun Hong, Jessica Ojo, Marie Check, Eric Qiu

### Overview

This project implements a movie recommendation system for a simulated streaming service with around 1M customers and around 27K movies. Over the first milestones, we designed, deployed, and evaluated a production-ready recommendation service that interacts with the provided APIs and Kafka event streams. The system receives user activity logs (movie requests, ratings, and recommendation requests) and responds with personalized movie recommendations through an inference service.

### Project Structure

\- \`src/\` — source code: data download, feature building, training, inference, experiment scripts  
\- \`data/\` — prepared data, id lists, and \`raw_data/\` with original CSVs  
\- \`models/\` — saved trained models  
\- \`reports/\` — report write-ups and meeting notes  
\- \`train_results/\` — JSON results from training and hyperparameter tuning  
\- \`docker/\` — \`Dockerfile\` for containerizing the project

### Key files:

\- \`requirements.txt\` — Python dependencies  
\- \`src/configs.py\` — global configuration and constants  
\- \`src/feature\_builder.py\` — feature extraction and preprocessing  
\- \`src/trainer.py\` — training entrypoint / pipeline  
\- \`src/inference.py\` — recommender service/inference utilities  
\- \`src/download\_data.py\` — helpers to prepare or download data  
\- \`src/inference\_readme.md\` — notes about inference/service usage  
\- \`src/experiments/\` — experiment scripts (XGBoost, MLP, logistic regression, analysis)

### Quick start

1\. Create and activate a virtual environment and install dependencies:  
```bash  
**create virtual environment**  
python3 -m venv .venv  
source .venv/bin/activate
```
**upgrade pip and install requirements**  
```bash
pip install --upgrade pip  
pip install -r requirements.txt  
```

2\. Prepare data. Place the raw CSVs under \`data/raw_data/\` if they aren't already present. See \`src/download_data.py\` for helper utilities.

### How to Run

**Train the default pipeline:**

```bash  
source .venv/bin/activate  
python src/trainer.py  
```

**Run experiment scripts** 

The scripts under \`src/experiments/\` are small, training/analysis scripts that evaluates the performance of different models. They require command-line arguments; here are the exact usages that the scripts expect:

\- XGBoost experiment  
```bash  
python src/experiments/train_model_xgb.py <ratings_csv> <out_model>  
```

\- MLP experiment 

```bash  
python src/experiments/train_model_mlp.py <ratings_csv> <out_model>  
```

\- Logistic regression experiment

```bash  
python src/experiments/train_model_logistic_regression.py <ratings_csv> <out_model>  
```

**Inference**  
\`src/inference.py\` includes a small \`RecommenderEngine\` class and a runnable example in the \`if __name__ == "__main__"\` block. Current behaviour:

\- Running \`python src/inference.py\` will load the model at \`src/models/xgb\_recommender.joblib\` (default path), read \`data/raw_data/movies.csv\` for movie metadata, and run a hard-coded example (user_id 13262\) — it prints a comma-separated list of recommended \`movie_id\`s.

```python  
from src.inference import RecommenderEngine  
engine = RecommenderEngine(model_path='src/models/xgb_recommender.joblib', movies_file='data/raw_data/movies.csv', mode='dev')  
print(engine.recommend(12345, top_n=10))  
```

### Models & artifacts

\- Trained models: \`src/models/\` (e.g., \`xgb_recommender.joblib\`, \`xgb_recommender.pkl\`)  
\- Training/tuning outputs: \`src/train_results/\` (JSON files)

Load a saved model programmatically:

```python  
import joblib  
model = joblib.load('src/models/xgb_recommender.joblib')  
```
## Docker

Build and run a container using the provided Dockerfile:

```bash
docker build -f docker/Dockerfile -t movie-recommender:v1.0 .
# Example run (with log rotation limits and port mapping):
docker run -it --log-opt max-size=50m --log-opt max-file=5 -p 8080:8080 movie-recommender:v1.0
```
### Ethical Considerations

As a team, we have acknowledged the existence of ethical issues in this project and have made efforts to minimize them. For the data gathered, we have ensured that no personally identifiable information (PII) is collected or shared beyond what is provided in the simulated dataset. Furthermore, we recognize that recommendation algorithms can amplify existing popularity biases; therefore, this project is intended solely for the evaluation of the Comp585 project and it is not intended for deployment in real-world systems.

