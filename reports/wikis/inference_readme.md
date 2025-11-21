## Recommender Engine

This module runs a trained **movie recommendation engine**.  
It generates personalized movie recommendations for a given user ID by fetching their profile, combining it with all candidate movies, and ranking them by predicted preference.

---

### Installation

Clone the repository and install dependencies:

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
```

Requirements: **Python 3.10+**

---

### Modes

The engine supports two modes for loading the model:

#### 1. Prod Mode (`mode="prod"`)

- The model is stored on Hugging Face Hub.
- Requires a Hugging Face token (`huggingface-cli login`).
- The engine automatically downloads the model from the repo when initialized.

**Example:**

```python
from inference import RecommenderEngine

engine = RecommenderEngine(
    model_path="src/models/xgb_recommender.joblib", 
    repo_id="comp585Team6/recommender_model1", 
    mode="prod"
)
print(engine.recommend(user_id=13262)) 
# 13262 is a sample user_id. Test ids are provided in 'data/test_ids'
```

---

#### 2. Dev Mode (`mode="dev"`)

- The model must already exist locally in `src/models/xgb_recommender.joblib`.
- Does **not** require a Hugging Face token.

**Example:**

```python
from inference import RecommenderEngine

engine = RecommenderEngine(
    model_path="src/models/xgb_recommender.joblib", 
    mode="dev"
)
print(engine.recommend(user_id=13262))
```

---

### Inputs & Outputs

- **Input** - `user_id` → integer.
- **Output** - A single line containing up to **20 movie IDs**, comma-separated, ordered by recommendation strength.

**Example output:**

```
schindlers+list+1993,the+shawshank+redemption+1994,moolaad+2004,night+and+fog+1955
```

---

### How It Works

1. Fetch user info from the API.  
2. Build a candidate pool of all movies (`movies.csv`).  
3. Run candidate pairs through the **FeatureBuilder** to generate features.  
4. Pass features through the trained **XGBoost model** (inside a pipeline with preprocessing).  
5. Rank movies by predicted score.  
6. Return top 20 IDs.  

---

#### Notes

- If a user is **not found**, the engine defaults to a cold-start profile:

```json
{"age": -1, "occupation": "other or not specified", "gender": "U"}
```

- The engine expects a local copy of **`movies.csv`** in `data/raw_data/`.
