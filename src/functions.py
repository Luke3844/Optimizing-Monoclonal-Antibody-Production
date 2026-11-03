# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================

import numpy as np
# Downtime between batches is one day
downtime = 1

def calculate_average(values):
    return sum(values) / len(values)

# Economics Functions
def culture_cost(GlucoseInitial, GlutamineInitial, InitialVCD):
    return 10*GlucoseInitial + 20*GlutamineInitial + 0.0005 * InitialVCD

def culture_value(mAb_final):
    return 1000*mAb_final

def profit(cost, value):
    return value - cost

def space_time_profit(sum_profit, harvest_time):
    harvest_time_d = harvest_time/24.0
    return sum_profit / (harvest_time + downtime)

def space_time_yield(mAb_final, harvest_time):
    harvest_time_d = harvest_time/24.0
    return mAb_final / (harvest_time + downtime)

def generate_x_lag_cur_var(p, batch_ids, df, features, col_batch="Batch ID", col_time="Time [h]"):
    x_lag = []
    x_cur = []
    t_cur = []
    b_cur = []

    # Process each batch independently.
    for batch_id in batch_ids:
        # Select rows belonging to the current batch and extract all variables included in the VAR model.
        mask = df.loc[:, col_batch] == batch_id
        x_batch = df.loc[mask, features].to_numpy()
        t_batch = df.loc[mask, col_time].to_numpy()
        b_batch = df.loc[mask, col_batch].to_numpy()

        # Current observations x(t).
        # Each row contains the simultaneous values of all variables at a particular time point.
        current_values = x_batch[p:]
        current_times = t_batch[p:]
        current_batches = b_batch[p:]

        # Build lagged state vectors:
        #
        #     x(t-1), x(t-2), ..., x(t-p)
        #
        # where each lag contains all process variables.
        lagged_values = []
        for i in range(1, p + 1):
            # Extract the i-th lagged state.
            lagged_values.append(x_batch[p - i:-i])

        # Combine all lagged state vectors into a single feature matrix.
        #
        # For example, with two variables and p = 3:
        #
        #     [x1(t-1), x2(t-1), x1(t-2), x2(t-2), x1(t-3), x2(t-3)]
        #
        # Each row therefore contains the lag history used to predict the current state.
        lagged_values = np.concatenate(lagged_values, axis=1)

        # Store this batch's feature matrix, target matrix, and time values.
        x_lag.append(lagged_values)
        x_cur.append(current_values)
        t_cur.append(current_times)
        b_cur.append(current_batches)

    # Concatenate the independently generated batches into overall input and output arrays.
    x_lag = np.concatenate(x_lag, axis=0)
    x_cur = np.concatenate(x_cur, axis=0)
    t_cur = np.concatenate(t_cur, axis=0)
    b_cur = np.concatenate(b_cur, axis=0)

    return x_lag, x_cur, t_cur, b_cur