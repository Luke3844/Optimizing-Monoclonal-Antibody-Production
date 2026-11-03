# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================

# ============
### Imports
# ============

import pandas as pd
import os
import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()
from src.classes import Batch  # This allows you to import individual Classes from classes.py
from src.constants import *  # This imports all constants from constants.py


### Dataset Imports
batch_ids = [
    "0343", "0367", "0391", "0349", "0364", "0493",
    "0511", "0514", "0517", "0535", "0496", "0532",
    "0643", "0661", "0679", "0658", "0637", "0385",
    "0685", "0490", "0220", "0808", "0472", "0556",
    "0508", "0520", "0073", "0955", "0451", "0523",
]

dfs = []
for bid in batch_ids:
    df = pd.read_csv(os.path.join("datasets", f"batch_{bid}.csv"))
    df.insert(0, "batch_id", bid)  # add as leftmost column
    dfs.append(df)

df_all = pd.concat(dfs, ignore_index=True)

# =======================
# Setup Model Parameters
# =======================

p_max = 5
cv_splits = 3
poly_degree_max = 3
knn_k_max = 5
model_types = ["poly", "knn"]
param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1, poly_degree_max + 1)])
param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(1, knn_k_max + 1)],
                      kneighborsregressor__weights=["uniform", "distance"])
