import joblib
from trainer import Trainer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np


def evaluate(preproc_path, model_path):
    """Run evaluation of recommenders on test data"""
    eval_data = "data/training_data_v2.csv"

    preprocessor = joblib.load(preproc_path)
    xgb_model = joblib.load(model_path)

    print("Loaded model and preprocessor.")
    pipeline = Pipeline([("preprocessor", preprocessor), ("model", xgb_model)])

    trainer = Trainer()

    print("Loading evaluation data...")
    trainer.df = pd.read_csv(eval_data)[:10]#.sample(frac=0.01, random_state=42).reset_index(drop=True)
    print("Preparing features...")
    X, y, _, _, _ = trainer.prepare_features()
    print("Evaluating model...")
    results, y_test, preds = trainer._evaluate(pipeline, X, y)
    print("Evaluation Results:", results)


evaluate("src/models/preprocessor.joblib", "src/models/xgb_model.joblib")