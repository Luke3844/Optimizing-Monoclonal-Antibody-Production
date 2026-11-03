# =====================================================================
# MAIN PROGRAM
# Your code starts here. This is the file that will be run.
# =====================================================================

# ============
### Imports
# ============

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from src.classes import StandardScalerVAR
import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()
from src.classes import Batch  # This allows you to import individual Classes from classes.py
from src.constants import *  # This imports all constants from constants.py


### Dataset Imports
batch_ids = [
    "0343", "0367", "0391", "0349", "0364", "0493", "0511", "0514", "0517", "0535", "0496", "0532", "0643", "0661", "0679", "0658", "0637", "0385",
    "0685", "0490", "0220", "0808", "0472", "0556", "0508", "0520", "0073", "0955", "0451", "0523",
] #Choices justified in readme

dfs = []
for batch_id in batch_ids:
    df = pd.read_csv(os.path.join("datasets", f"batch_{batch_id}.csv"))
    df.insert(0, "Batch ID", batch_id)  # add as leftmost column
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

# ==================================
# Setup Model And Analysis Parameters
# ===================================
p_max = 5
cv_splits = 3
poly_degree_max = 3
knn_k_max = 5
model_types = ["poly", "knn"]
param_grid_poly = dict(polynomialfeatures__degree=[i for i in range(1, poly_degree_max + 1)])
param_grid_knn = dict(kneighborsregressor__n_neighbors=[i for i in range(1, knn_k_max + 1)],
                      kneighborsregressor__weights=["uniform", "distance"])

features = df.columns[2:]
feature_plot = "VCD [cells/mL]"
idx_fp = np.argmax(features == feature_plot)
units = feature_plot.split(" [")[-1][:-1]
prediction_interval = 0.95
alpha = 1 - prediction_interval
x_left = np.min(df.loc[:, "Time [h]"]) - 1
x_right = np.max(df.loc[:, "Time [h]"]) + 1

# ===============
# Model Training
# ===============
for model_type in model_types:
    for p in range(1, p_max + 1):
        x_lag, x_cur, t_cur, b_cur = sf.generate_x_lag_cur_var(p=p, batch_ids=batch_ids, df=df, features=features)

        # Create the model
        if model_type == "poly":
            model = make_pipeline(
                StandardScalerVAR(p=p),
                PolynomialFeatures(include_bias=False),
                LinearRegression()
            )
            param_grid = param_grid_poly
        elif model_type == "knn":
            model = make_pipeline(
                StandardScalerVAR(p=p),
                KNeighborsRegressor()
            )
            param_grid = param_grid_knn
        else:
            model = None
            param_grid = {}

        # Keep the batches separate in GridSearchCV
        gkf = GroupKFold(n_splits=cv_splits)
        model_gscv = GridSearchCV(model, param_grid=param_grid, cv=gkf, scoring="neg_root_mean_squared_error",
                                  return_train_score=True)

        # Fit the model
        model_gscv.fit(x_lag, x_cur, groups=b_cur)

        # Note: Here we predicted using the training data because we are evaluating whether the model was able to learn
        # the time series patterns in the dataset. We are NOT evaluating how well it generalizes to new data.
        x_cur_pred = model_gscv.predict(x_lag)

        # Calculate the residual errors (noise)
        x_cur_res = x_cur_pred - x_cur

        # Determine the unique time points in the dataset
        t_unique = np.unique(t_cur)

        # We will store the residual mean and prediction intervals at each time point
        res_mean = []
        res_pi_upper = []
        res_pi_lower = []

        # Analyze the residual distribution at each time point for the feature of interest
        for t in t_unique:
            mask = t_cur == t
            x_cur_res_t = x_cur_res[mask, idx_fp]
            n_mean = np.mean(x_cur_res_t)
            n_sd = np.std(x_cur_res_t)

            # Calculate the prediction interval using Student's t distribution
            q = alpha / 2
            dof = x_cur_res_t.shape[0] - 1
            pi_sd = stats.t.isf(q=q, df=dof)

            res_mean.append(n_mean)
            res_pi_upper.append(n_mean + pi_sd * n_sd)
            res_pi_lower.append(n_mean - pi_sd * n_sd)

        res_mean = np.array(res_mean)
        res_pi_upper = np.array(res_pi_upper)
        res_pi_lower = np.array(res_pi_lower)

        # Determine the plotting range needed to display all residual statistics symmetrically about zero
        y_max = max(np.max(np.abs(res_mean)),
                    np.max(np.abs(res_pi_upper)),
                    np.max(np.abs(res_pi_lower)),
                    np.max(np.abs(x_cur_res[:, idx_fp])))

        fig, ax = plt.subplots(1, 1, figsize=((14.39 / 2.54), (12.09 / 2.54)), dpi=300, layout="constrained")

        ax.scatter(t_cur, x_cur_res[:, idx_fp], label="Noise", color="tab:orange", s=12, edgecolors="black",
                   linewidth=0.5,
                   alpha=0.7, zorder=3)
        ax.plot(t_unique, res_mean, label="Mean", color="black", zorder=4)
        ax.fill_between(t_unique, res_pi_lower, res_pi_upper, label=f"{prediction_interval:.0%} PI", color="tab:blue",
                        edgecolors="none", alpha=0.3, zorder=2)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5, zorder=1)

        ax.set_xlabel("Time [h]", fontsize=14)
        ax.set_ylabel(f"{feature_plot} (Residuals)", fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.legend(loc="upper left", fontsize=12)

        ax.xaxis.set_major_locator(MultipleLocator(4))
        ax.set_xlim(left=x_left, right=x_right)
        ax.set_ylim(bottom=-1.05 * y_max, top=1.05 * y_max)

        fig.savefig(os.path.join("figures", f"figure_white_noise_{model_type}_p{p:02d}.png"))
        plt.close(fig)

        if model_type == "poly":
            best_degree = model_gscv.best_params_["polynomialfeatures__degree"]
            print(f"{p = } | Degree: {best_degree}")
        elif model_type == "knn":
            best_n_neighbors = model_gscv.best_params_["kneighborsregressor__n_neighbors"]
            best_weight = model_gscv.best_params_["kneighborsregressor__weights"]
            print(f"{p = } | k={best_n_neighbors}, {best_weight}")