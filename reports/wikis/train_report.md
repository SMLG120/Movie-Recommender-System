# Training Report

## Overview
We trained an XGBoost regressor on the feature-engineered dataset to predict user ratings (1–5 scale).  
The model leverages user demographics (age, gender, occupation), movie metadata (runtime, popularity, vote metrics, release year, genres, languages, production countries), and explicit rating labels.

## Training Setup
- **Model:** XGBoost Regressor (`n_estimators=300`, `max_depth=8`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`)
- **Split:** 80% train / 20% test
- **Evaluation metrics:** RMSE, MAE, R²
- **Logs:** Training metadata (metrics, timings, parameter count, dataset sizes, hyperparameters) stored in `src/train_results/training_results.json`.

## Results
- **RMSE:** 0.7152  
- **MAE:** 0.5840  
- **R²:** 0.0805  
- **Train time:** 1.69 seconds 
- **Parameter count:** 300 trees  

## Interpretation
This model represents a **content-based baseline** recommender. It leverages user demographics (age, occupation, gender) and movie metadata (genres, languages, runtime, release year, etc.) to predict ratings.  

Performance is modest (RMSE ≈ 0.72, R² ≈ 0.08), which is expected since content-based features alone cannot fully capture complex user–movie interaction patterns. Users with similar demographics may still have very different preferences, and movies with similar metadata may not always be rated consistently.  

The strength of this baseline lies in its ability to handle **cold-start cases** (new users/movies) where only metadata is available. However, for stronger personalization and accuracy, further integration of collaborative filtering signals will be needed.


## Observations
- Performance is decent for a first-pass model.  
- R² is a limited metric here; RMSE/MAE are better indicators of quality.  
- Cold-start issues (new users/movies) remain a challenge and will need additional strategies (e.g., content embeddings or hybrid models).  

## Next Steps
1. **Potential feature expansions**:  
   - Watch time (implicit feedback).  
   - Text embeddings from movie overviews.  
   - Interaction-based features (e.g., user–genre affinity).  
2. **Hybrid modeling** by combining content-based features with collaborative filtering signals to leverage both metadata and user–item interactions.

## Conclusion
The current XGBoost model serves as a **minimal, content-based baseline** that establishes a reference point for evaluation. While its predictive power is limited, it provides useful insights into how metadata alone can support recommendations and cold-start scenarios.  

The next stage should focus on moving toward a **hybrid recommender system** that balances metadata-driven predictions with collaborative signals derived from user–movie interactions. This progression will enable more personalized, accurate, and scalable recommendations.
