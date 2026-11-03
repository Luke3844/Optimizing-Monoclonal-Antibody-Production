# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================
import numpy as np
from sklearn.preprocessing import StandardScaler


class Batch:
    def __init__(self, batch_id, ph_opt, temperature_opt):
        self.batch_id = batch_id
        self.ph_opt = ph_opt
        self.temperature_opt = temperature_opt

    def display_info(self):
        print(f"Batch ID: {self.batch_id:4d} | "
              f"Optimal pH: {self.ph_opt:4.1f} | "
              f"Optimal Temperature: {self.temperature_opt:4.1f} C")

class StandardScalerVAR:

    def __init__(self, p):
        self.p = p

        self.mean_ = None
        self.scale_ = None
        self.var_ = None

        self.n_samples_seen_ = None
        self.n_features_in_ = None

        self.feature_names_in_ = None

    def fit(self, x, y=None):
        # Determine how many process variables are present.
        n_var_features = x.shape[1] // self.p

        # Ensure the number of columns is compatible with the specified autoregressive order.
        assert n_var_features * self.p == x.shape[1]

        # Separate each lag block:
        #
        #     Lag 1 --> columns corresponding to x(t-1)
        #     Lag 2 --> columns corresponding to x(t-2)
        #     ...
        #
        # Each block contains all process variables.
        x_sf = []
        for i in range(self.p):
            idx_start = i * n_var_features
            idx_end = idx_start + n_var_features
            x_sf.append(x[:, idx_start:idx_end])

        # Concatenate the lag blocks vertically (i.e., by rows).
        #
        # This creates a matrix where each column corresponds to a process variable and
        # all lagged observations contribute to the scaling statistics.
        x_sf = np.concatenate(x_sf, axis=0)

        # Calculate scaling statistics independently for each process variable.
        x_sf_mean = np.mean(x_sf, axis=0)
        x_sf_sd = np.std(x_sf, axis=0)
        x_sf_var = np.var(x_sf, axis=0)

        # Tile the statistics so that every lagged copy of a process variable uses identical scaling parameters.
        self.mean_ = np.tile(x_sf_mean, (1, self.p))
        self.scale_ = np.tile(x_sf_sd, (1, self.p))
        self.var_ = np.tile(x_sf_var, (1, self.p))

        # Store metadata expected by sklearn-like estimators.
        self.n_samples_seen_ = x.shape[0]
        self.n_features_in_ = x.shape[1]

        return self

    def transform(self, x):
        x_tr = (x - self.mean_) / self.scale_
        return x_tr

    def inverse_transform(self, x):
        x_original = self.scale_ * x + self.mean_
        return x_original

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x)

    def get_params(self, deep=True):
        return {"p": self.p}

    def set_params(self, **params):

        if "p" in params:
            self.p = params["p"]

        return self