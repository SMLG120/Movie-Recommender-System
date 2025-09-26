# COMP 585 Milestone 1 Report - Team 6

## Learning Process

We implemented a Multi-Layer Perceptron (MLP) model using TensorFlow and Keras to predict movie watch times based on user-movie interactions. The MLP architecture consists of embedding layers for users and movies, followed by dense layers to learn latent factors influencing watch time. This approach was chosen because collaborative filtering via embeddings can capture complex user-item relationships, and predicting watch time provides a ranking metric for recommendations.

The model was trained on a dataset of user-movie watch times, with the target variable scaled to [0,1] using MinMaxScaler. Training used the Adam optimizer and Mean Squared Error (MSE) loss, with 10 epochs and a batch size of 32. The trained model achieved reasonable performance on the validation set.

The training script is located at `src/train_model_mlp.py`. It reads the watch time data from `Data/watch_time.csv`, encodes users and movies to indices, builds and trains the MLP, and saves the model and mappings.

As a result of the training process, the script created:
- `Data/model_watch_time_mlp.h5`: The trained TensorFlow/Keras model file containing the MLP weights and architecture.
- `Data/model_watch_time_mappings.pkl`: A pickle file containing the user and movie ID mappings, unique IDs, and the MinMaxScaler used for watch time normalization.

## Data Analysis and Preprocessing

After training the MLP model, the `src/analyze_watch_time.py` script serves a crucial purpose by processing the raw watch time data (`Data/watch_time.csv`) to generate `Data/popular_movies.csv`. This file aggregates total watch time per movie, ranking them by popularity. It acts as a pre-computed summary for quick insights into overall movie engagement, derived from user interactions.

### Analogy for popular_movies.csv
Imagine a bookstore tracking reading times for books. Raw logs record individual sessions (e.g., "Reader A read Book X for 30 minutes"). The `popular_movies.csv` is the bookstore's "bestseller summary" – it sums times per book to rank them (e.g., "Book Z has 5,000 total minutes, most popular"). This enables fast decisions without re-analyzing logs. In the recommender, it provides a fallback for popularity-based suggestions when personalized predictions aren't available.

### Comparison to Other Data Files
- **Raw data (e.g., `watch_time.csv`)**: Granular user-movie interactions; source for training.
- **Metadata (e.g., `movies.csv`)**: Static movie details; external references.
- **popular_movies.csv**: Aggregated popularity metric; efficient for non-personalized insights.

## Inference Service

The recommendation service is implemented as a Flask API in `src/app.py`, running on port 8082. It loads the trained MLP model and mappings, and provides a `/recommend/<user_id>` endpoint that returns up to 20 movie IDs as a comma-separated string, ordered by predicted watch time descending.

Recommendations are derived by predicting watch times for all movies not rated by the user, ranking them by the predicted values, and selecting the top 20. If the user is not in the training data or the model fails to load, it falls back to a static list of popular movie IDs from `popular_movies.csv`.

The service also consumes Kafka messages from the 'movielog6' stream to collect real-time ratings, which are used to exclude already-rated movies from recommendations.

