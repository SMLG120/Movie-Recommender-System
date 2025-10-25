import argparse
import json
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import xgboost as xgb

# Import trainer.py and run for offline evaluation
from inference import recommender

# create a testset
# run the cf trainer to get test embeddings
# run it through the feature builder, pass the test embeddins to the feature builder also
# load eaxisting model
# run prediction and evaluation

# train
# start_train = time.time()
# self.pipeline.fit(X_train, y_train)
# train_time = time.time() - start_train

# preds = self.pipeline.predict(X_test)
# infer_time = (time.time() - start_train) / max(len(X_test), 1)

# mse = mean_squared_error(y_test, preds)
# rmse = np.sqrt(mse)
# mae = mean_absolute_error(y_test, preds)
# r2 = r2_score(y_test, preds)



from src.trainer import Trainer
from sklearn.pipeline import Pip    eline
from sklearn.model_selection import train_test_split


def evaluate(self, preproc_path, model_path):
    """Run evaluation of recommenders on test data"""
    eval_data = "data/training_data_v2.csv"

    preprocessor = joblib.load(preproc_path)
    xgb_model = joblib.load(model_path)
    self.pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb_model)])

    trainer = Trainer() # trainer initialization
    trainer.df = pd.read_csv(eval_data)
    X, y, _, _, _ = trainer.prepare_features()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results, y_test, preds = trainer._evaluate(self.pipeline, X_test, y_test)
    print("Evaluation Results:", results) # contains mse, rmse, mae, r2
    
    # other metric calculations
    metrics = self.metrics(y_test, preds)
    print("Additional Metrics:", metrics)

    def metrics(self, y_test, preds):
        """Compute regression and classification metrics"""
        reg_metrics = regression_metrics(y_test, preds)
        class_metrics = classification_metrics((y_test >= 3).astype(int), preds)
        return {reg_metrics, class_metrics}
