# COMP 585 Milestone 1 Report - Team 6

## Learning Process

We implemented an XGBoost model to predict movie watch times based on user-movie interactions. XGBoost is a gradient boosting framework that uses decision trees as base learners, providing high performance and interpretability. This approach was chosen for its ability to handle tabular data effectively and capture non-linear relationships in collaborative filtering, with predicting watch time as a ranking metric for recommendations.

The model was trained on a dataset of user-movie watch times, with the target variable scaled to [0,1] using MinMaxScaler. Training used default XGBoost parameters with 100 estimators, max depth 6, and learning rate 0.1, with early stopping on validation set. The trained model achieved reasonable performance.

The training script is located at `src/train_model_xgb.py`. It reads the watch time data from `Data/watch_time.csv`, encodes users and movies to indices, builds and trains the XGBoost regressor, and saves the model and mappings.

As a result of the training process, the script created:
- `Data/model_watch_time_xgb.pkl`: The trained XGBoost model file containing the ensemble of decision trees.
- `Data/model_watch_time_mappings.pkl`: A pickle file containing the user and movie ID mappings, unique IDs, and the MinMaxScaler used for watch time normalization.

## Algorithm Implementations

### Logistic Regression
The Logistic Regression implementation is in `src/train_model_logistic_regression.py`. It loads watch time data from `Data/watch_time.csv`, encodes user and movie IDs to integer indices, and prepares features as a concatenation of user and movie indices. The target is binarized: 1 if watch time > 30 minutes, else 0. The data is split into train/test sets (80/20), and a LogisticRegression model from scikit-learn is trained. The model and mappings (including the threshold) are saved to `Data/model_watch_time_logistic.pkl` and `Data/model_watch_time_mappings.pkl`.

**Pros:**
- Simple and fast to train and infer.
- Interpretable coefficients for feature importance.
- Effective for binary classification tasks with linear separability.

**Cons:**
- Assumes linear relationships, may underperform on complex, non-linear data.
- Outputs binary predictions, not continuous watch time values.
- Sensitive to feature scaling and outliers.

### Multi-Layer Perceptron (MLP)
The MLP implementation is in `src/train_model_mlp.py`. It uses TensorFlow/Keras to build a neural network for regression. User and movie IDs are encoded to indices and fed into embedding layers (dimension 50), which are concatenated and passed through dense layers (128 units with ReLU, 64 units with ReLU, and a final output layer). The model is compiled with Adam optimizer and MSE loss. Watch times are scaled to [0,1] using MinMaxScaler. Training occurs for 10 epochs with batch size 32. The model is saved as `Data/model_watch_time_mlp.h5`, and mappings with the scaler are pickled.

**Pros:**
- Capable of learning complex, non-linear relationships through embeddings and layers.
- Scalable to larger datasets and additional features.
- Can capture latent factors in user-movie interactions.

**Cons:**
- Computationally intensive, requiring more resources and time for training.
- Prone to overfitting without proper regularization or early stopping.
- Less interpretable than tree-based or linear models.

### XGBoost
The XGBoost implementation is in `src/train_model_xgb.py`. It treats user and movie indices as numerical features in a tabular format. Watch times are scaled to [0,1]. An XGBRegressor is trained with default parameters (100 estimators, max depth 6, learning rate 0.1). The model is saved as `Data/model_watch_time_xgb.pkl`, with mappings and scaler pickled separately.

**Pros:**
- Handles tabular data well, with built-in handling of sparsity and feature interactions.
- Fast inference and robust to outliers.
- Provides feature importance for interpretability.

**Cons:**
- May not capture deep non-linear interactions as effectively as neural networks.
- Requires hyperparameter tuning to avoid overfitting.
- Performance can degrade with high-cardinality categorical features if not properly encoded.

## Data Analysis and Preprocessing

After training the XGBoost model, the `src/analyze_watch_time.py` script serves a crucial purpose by processing the raw watch time data (`Data/watch_time.csv`) to generate `Data/popular_movies.csv`. This file aggregates total watch time per movie, ranking them by popularity. It acts as a pre-computed summary for quick insights into overall movie engagement, derived from user interactions.

### Analogy for popular_movies.csv
Imagine a bookstore tracking reading times for books. Raw logs record individual sessions (e.g., "Reader A read Book X for 30 minutes"). The `popular_movies.csv` is the bookstore's "bestseller summary" – it sums times per book to rank them (e.g., "Book Z has 5,000 total minutes, most popular"). This enables fast decisions without re-analyzing logs. In the recommender, it provides a fallback for popularity-based suggestions when personalized predictions aren't available.

### Comparison to Other Data Files
- **Raw data (e.g., `watch_time.csv`)**: Granular user-movie interactions; source for training.
- **Metadata (e.g., `movies.csv`)**: Static movie details; external references.
- **popular_movies.csv**: Aggregated popularity metric; efficient for non-personalized insights.

## Inference Service

The recommendation service is implemented as a Flask API in `src/app.py`, running on port 8082. It loads the trained XGBoost model and mappings, and provides a `/recommend/<user_id>` endpoint that returns up to 20 movie IDs as a comma-separated string, ordered by predicted watch time descending.

Recommendations are derived by predicting watch times for all movies not rated by the user, ranking them by the predicted values, and selecting the top 20. If the user is not in the training data or the model fails to load, it falls back to a static list of popular movie IDs from `popular_movies.csv`.

The service also consumes Kafka messages from the 'movielog6' stream to collect real-time ratings, which are used to exclude already-rated movies from recommendations.

## Algorithm Comparison: Logistic Regression, XGBoost, and MLP

Logistic Regression, XGBoost, and MLP (Multi-Layer Perceptron) are used for predicting movie watch times, with Logistic Regression treating it as binary classification (watch time > 30 minutes), while XGBoost and MLP handle it as regression. They differ in approach, strengths, and application in recommendation systems.

### How the Algorithms Are Used Differently
- **Logistic Regression**: A linear model that treats user and movie IDs as numerical features. It predicts binary outcomes (e.g., high vs. low engagement) based on linear combinations of features. In this project, it's used for simple, interpretable classification on user-movie pairs, suitable for quick predictions when continuous values aren't needed. It's ideal for baseline models or when data is linearly separable.
- **XGBoost**: A tree-based ensemble method that treats user and movie IDs as numerical features directly. It excels in handling tabular data with built-in feature interactions and sparsity, making it suitable for quick, interpretable predictions. In this project, XGBoost is used for straightforward collaborative filtering on user-movie pairs, providing fast training and inference with minimal hyperparameter tuning. It's ideal for smaller datasets or when interpretability (e.g., feature importance) is key.
- **MLP**: A neural network that uses embedding layers to learn dense vector representations of categorical user and movie IDs, capturing latent factors and non-linear relationships. It's used here for more complex pattern recognition in recommendation systems, potentially scaling better with larger data or additional features (e.g., genres). MLP requires more computational resources and tuning (e.g., epochs, layers) but can model nuanced interactions better than trees.

### Time and Space Complexity Comparison
Empirical complexity is measured using `src/compare_complexities.py`, which trains the models on the same data and reports metrics. Theoretical complexities:
- **Logistic Regression**: Training ~O(N * D) (N=samples, D=features); Inference ~O(D); Space ~O(D).
- **XGBoost**: Training ~O(T * N * D log N) (T=trees, N=samples, D=features); Inference ~O(T * D); Space ~O(T * D).
- **MLP**: Training ~O(E * B * L * H^2) (E=epochs, B=batch size, L=layers, H=hidden units); Inference ~O(L * H); Space ~O(params).

Run `python src/compare_complexities.py Data/watch_time.csv` to get empirical results (training/inference time, peak memory, model size, param count).

