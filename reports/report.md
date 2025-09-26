# COMP 585 Milestone 1 Report - Team 6

## Learning Process

We implemented a Multi-Layer Perceptron (MLP) model using TensorFlow and Keras to predict movie watch times based on user-movie interactions. The MLP architecture consists of embedding layers for users and movies, followed by dense layers to learn latent factors influencing watch time. This approach was chosen because collaborative filtering via embeddings can capture complex user-item relationships, and predicting watch time provides a ranking metric for recommendations.

The model was trained on a dataset of user-movie watch times, with the target variable scaled to [0,1] using MinMaxScaler. Training used the Adam optimizer and Mean Squared Error (MSE) loss, with 10 epochs and a batch size of 32. The trained model achieved reasonable performance on the validation set.

The training script is located at `src/train_model_mlp.py`. It reads the watch time data from `Data/watch_time.csv`, encodes users and movies to indices, builds and trains the MLP, and saves the model and mappings.

As a result of the training process, the script created:
- `Data/model_watch_time_mlp.h5`: The trained TensorFlow/Keras model file containing the MLP weights and architecture.
- `Data/model_watch_time_mappings.pkl`: A pickle file containing the user and movie ID mappings, unique IDs, and the MinMaxScaler used for watch time normalization.

## Inference Service

The recommendation service is implemented as a Flask API in `src/app.py`, running on port 8082. It loads the trained MLP model and mappings, and provides a `/recommend/<user_id>` endpoint that returns up to 20 movie IDs as a comma-separated string, ordered by predicted watch time descending.

Recommendations are derived by predicting watch times for all movies not rated by the user, ranking them by the predicted values, and selecting the top 20. If the user is not in the training data or the model fails to load, it falls back to a static list of popular movie IDs.

The service also consumes Kafka messages from the 'movielog6' stream to collect real-time ratings, which are used to exclude already-rated movies from recommendations.

