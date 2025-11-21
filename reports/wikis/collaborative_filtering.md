# 🤝 Collaborative Filtering Wiki

## 1. Overview
Collaborative Filtering (CF) is the backbone of modern recommender systems. It leverages **user–item interaction patterns** rather than explicit metadata to infer latent preferences. While metadata-based models rely on movie features (e.g., genre, language), CF captures **hidden behavioral relationships** — users with similar watch patterns tend to like similar movies.

Matrix Factorization (MF)-based CF, such as **SVD** and **ALS**, has become the dominant paradigm due to:
- **Scalability** on large sparse datasets (e.g., MovieLens).
- **Ability to model latent dimensions** such as taste or novelty-seeking.
- **Empirical superiority** over neighborhood-based CF in sparse regimes (Huang & Singh, 2020).

## 2. Explicit vs Implicit CF
| Aspect | Explicit CF (SVD) | Implicit CF (ALS) |
|--------|------------------|------------------|
| Input signal | User ratings (e.g., 1–5 stars) | User behavior (e.g., views, watch time) |
| Feedback type | Direct user preference | Indirect engagement (implicit) |
| Loss objective | Predict observed rating | Maximize confidence-weighted co-occurrence |
| Matrix type | Dense where ratings exist | Weighted sparse (confidence matrix) |
| Key library | `surprise.SVD` | `implicit.als.AlternatingLeastSquares` |

**Explicit CF (SVD):** Learns latent factors from user ratings via **Singular Value Decomposition**. Useful when user–item ratings are trustworthy and capture explicit sentiment.

**Implicit CF (ALS):** Designed for **behavioral data** where explicit feedback is missing or sparse. Models *confidence* in each observation — e.g., a user who replays or watches longer has higher affinity. ALS alternates between user and item embedding updates with regularization (Zhou et al., 2008).

Together, they form a **hybrid latent representation** combining explicit and implicit preferences.

## 3. Confidence Modeling (for ALS)
Implicit CF relies on a **confidence matrix** `C = 1 + α * score`, where `score` derives from user activity:

```python
completion = max_minute_reached / movie_duration
freq_norm = log1p(interaction_count) / log1p(movie_duration)
score = 0.7 * completion + 0.25 * freq_norm
confidence = 1 + 80 * score
```

**Design rationale:**
- *Completion ratio (70%)* → proxy for engagement quality.  
- *Frequency normalization (25%)* → reflects repeated interest.  
- *α = 80* → empirically stable scaling for MovieLens-scale data (Hu et al., 2008).

## 4. Model Architecture
### Explicit CF (SVD)
```python
n_factors = 50
n_epochs = 30
lr_all = 0.005
reg_all = 0.02
```
These hyperparameters balance convergence speed and generalization. Larger `n_factors` improve expressiveness but risk overfitting.

### Implicit CF (ALS)
```python
factors = 50
regularization = 0.01
iterations = 20
use_cg = True
```
This configuration mirrors findings where ~50 latent factors yield optimal tradeoffs between accuracy and runtime (Huang & Singh, 2020).

Both models output:
- `user_factors_*` and `movie_factors_*` embeddings.  
- `maps/*.json` to preserve user–item ID mapping.

## 5. Why Combine Explicit + Implicit CF
Combining both mitigates individual weaknesses:
- **SVD** struggles with sparse ratings → complemented by implicit behavioral signals.
- **ALS** ignores explicit sentiment intensity → corrected by SVD.  
- Hybrid embeddings enrich downstream models (e.g., XGBoost) with **behavioral** and **expressive** features, improving generalization in cold-start and mixed-feedback settings.

## 6. Parameter Rationale
| Setting | Value | Justification |
|----------|--------|----------------|
| `n_factors = 50` | Balance expressiveness vs memory cost for 10⁴–10⁵-scale datasets. |
| `alpha = 80` | Scales confidence to emphasize higher engagement without overfitting. |
| `reg = 0.01–0.02` | Prevents overfitting, stabilizes training. |
| `n_epochs = 30` | Converges near-optimal RMSE while remaining efficient. |
| `use_cg = True` | Conjugate gradient solver accelerates ALS convergence. |

## 7. Outputs and Integration
Each CF model produces:
- **Embeddings:**  
  `user_factors_explicit.csv`, `movie_factors_explicit.csv`,  
  `user_factors_implicit.csv`, `movie_factors_implicit.csv`
- **Mapping files:**  
  `maps/explicit_maps.json`, `maps/implicit_maps.json`

These embeddings are merged during feature building and treated as numeric features, enabling the hybrid recommender (e.g., XGBoost) to learn from both **latent collaborative signals** and **explicit metadata**.

---

## 8. References
- Hu, Y., Koren, Y., & Volinsky, C. (2008). *Collaborative Filtering for Implicit Feedback Datasets.* IEEE International Conference on Data Mining.  
- Zhou, Y., Wilkinson, D., Schreiber, R., & Pan, R. (2008). *Large-Scale Parallel Collaborative Filtering for the Netflix Prize.* Lecture Notes in Computer Science.  
- Huang, M., & Singh, A. (2020). *The Comparison Study of Matrix Factorization on Collaborative Filtering Recommender System.* Journal of Intelligent Systems Research.