import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import sys

# usage: python train_model.py ratings.csv model.pkl
ratings_csv = sys.argv[1]
out_model = sys.argv[2]

df = pd.read_csv(ratings_csv)  # columns: user_id, movie_id, rating

# pivot to user-item matrix (users rows, movies cols)
uim = df.pivot_table(index='user_id', columns='movie_id', values='rating', fill_value=0)

# compute item-item cosine similarity
item_matrix = uim.T  # movies x users
sim = cosine_similarity(item_matrix)  # movies x movies

model = {
    'movie_ids': list(item_matrix.index),   # order of movies
    'movie_sim': sim,                       # numpy array
    'user_item': uim                         # pandas DataFrame
}

with open(out_model, 'wb') as f:
    pickle.dump(model, f)

print('Saved model to', out_model)