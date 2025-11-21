# Model Improvement 

## 1. Overview
This document describes the **model improvement phase** for the recommender system, focusing on:
- **Feature extraction** and preprocessing  
- **Handling of missing values and outliers**  
- **Training pipeline architecture (XGBoost)**  
- **Evaluation metrics**

The final model combines **metadata-based features** (user and movie attributes) with **latent collaborative embeddings** derived from explicit and implicit feedback (see *Collaborative Filtering Wiki* for details).  

## 2. Feature Extraction and Preprocessing

### 2.1 Data Merging
The `FeatureBuilder` integrates:
- **User data** (`users.csv`)
- **Movie metadata** (`movies.csv`)
- **Ratings** (explicit preferences)
- **Collaborative embeddings** (explicit + implicit CF outputs)

Merging strategy:
- `ratings` → `users` (via `user_id`)
- Result → `movies` (via `movie_id`)
- Then merges all available embeddings on matching IDs  
This ensures both behavioral (CF) and contextual (metadata) signals are represented.

### 2.2 Missing Value Strategy
| Feature | Handling | Rationale |
|----------|-----------|------------|
| `age` | Filled with median | Retains population distribution without skewing extremes |
| `runtime` | Filled with median, then clipped [30, 720] | Removes extreme outliers (trailers, data errors) |
| `popularity`, `vote_average`, `vote_count` | Filled with 0 | Missing implies “no available data” |
| `release_year` | Extracted from `movie_id` if missing, else median | Many missing years exist in `release_date`; ID suffix recovers most |
| `occupation`, `gender`, `original_language` | Filled with `"unknown"` / `"U"` | Keeps categorical integrity |
| `spoken_languages` | Filled with `original_language` | Default to primary movie language |

Additional columns (`id`, `title`, `overview`, etc.) are ignored since they’re not predictive.

### 2.3 Feature Engineering
**1. Categorical Encoding**
- Columns: `age_bin`, `occupation`, `gender`, `original_language`
- Encoded via `OneHotEncoder(handle_unknown="ignore")`  

**2. Multi-Hot Expansion**
- `genres`, `production_countries`, and `spoken_languages` expanded into binary columns (`genre_Action`, `country_USA`, etc.)
- `spoken_languages` normalized using a `LANGUAGE_MAP` dictionary  

**3. Age Binning**
- Discretized into 5 brackets: `[0-18, 19-25, 26-35, 36-50, 50+]`
- Provides generalizable demographic categories

**4. Outlier Clipping**
| Column | Range | Reason |
|---------|--------|--------|
| `age` | [5, 100] | Removes unrealistic user ages |
| `runtime` | [30, 720] | Filters short clips or data errors |
| `release_year` | [1500, current_year+1] | Ensures realistic historical span |

Outliers are **clipped** rather than dropped to retain samples while maintaining stability.

## 3. XGBoost Training Pipeline

### 3.1 Data Preparation
- The `Trainer` class loads the final dataset built by `FeatureBuilder`.
- Features split into:
  - **Categorical** → one-hot encoded  
  - **Numeric** → passed through directly (multi-hot, CF embeddings, numeric movie features)

ColumnTransformer ensures consistent transformations during both training and inference.

### 3.2 Training Process
The pipeline:
```text
Load data → Split → Preprocess → Tune → Train (XGBoost) → Evaluate → Save
```
- **Objective:** Regression task predicting user ratings.  
- **Eval metric:** Root Mean Squared Error (RMSE).  
- **Parallelized** across CPU cores (`n_jobs=-1`).  

Model persistence is modular:
- Preprocessor: `preprocessor.joblib`
- Model: `xgb_model_only.joblib`

This separation allows future replacement or retraining without re-encoding features.

## 4. Evaluation Metrics

| Metric | Purpose | Description |
|---------|----------|-------------|
| **RMSE** | Primary | Penalizes large deviations between predicted and true ratings |
| **MAE** | Complementary | Measures average absolute prediction error |
| **R²** | Fit Quality | Indicates variance explained by model |
| **Spearman Correlation** | Ranking sanity check | Tests monotonic ranking consistency |
| **Pearson Correlation** | Linearity check | Measures direct score correlation |

These metrics collectively ensure:
- Strong generalization (low RMSE/MAE)
- Rank-preserving quality (Spearman ≥ 0.7 for good coherence)
- Stable interpretability across runs

## 5. Summary
The pipeline improves over baseline recommenders by:
- Integrating **metadata + CF latent embeddings**  
- Applying **robust preprocessing** for missing and outlier data  
- Using **interpretable gradient-boosted trees (XGBoost)**  
- Tracking comprehensive **performance metrics**  

Future work will add:
- CI/CD integration for model regression checks  
- Live monitoring of model drift and inference latency  
