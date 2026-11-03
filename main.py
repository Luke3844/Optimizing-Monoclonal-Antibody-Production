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
df_ic = pd.read_csv(os.path.join("datasets", "initial_conditions.csv"))
df_1 = pd.read_csv(os.path.join("datasets", "batch_0343.csv"))
df_2 = pd.read_csv(os.path.join("datasets", "batch_0367.csv"))
df_3 = pd.read_csv(os.path.join("datasets", "batch_0391.csv"))
df_4 = pd.read_csv(os.path.join("datasets", "batch_0349.csv"))
df_5 = pd.read_csv(os.path.join("datasets", "batch_0364.csv"))
df_6 = pd.read_csv(os.path.join("datasets", "batch_0493.csv"))
df_7 = pd.read_csv(os.path.join("datasets", "batch_0511.csv"))
df_8 = pd.read_csv(os.path.join("datasets", "batch_0514.csv"))
df_9 = pd.read_csv(os.path.join("datasets", "batch_0517.csv"))
df_10 = pd.read_csv(os.path.join("datasets", "batch_0535.csv"))
df_11 = pd.read_csv(os.path.join("datasets", "batch_0496.csv"))
df_12 = pd.read_csv(os.path.join("datasets", "batch_0532.csv"))
df_13 = pd.read_csv(os.path.join("datasets", "batch_0643.csv"))
df_14 = pd.read_csv(os.path.join("datasets", "batch_0661.csv"))
df_15 = pd.read_csv(os.path.join("datasets", "batch_0679.csv"))
df_16 = pd.read_csv(os.path.join("datasets", "batch_0658.csv"))
df_17 = pd.read_csv(os.path.join("datasets", "batch_0637.csv"))
df_18 = pd.read_csv(os.path.join("datasets", "batch_0385.csv"))
df_19 = pd.read_csv(os.path.join("datasets", "batch_0685.csv"))
df_20 = pd.read_csv(os.path.join("datasets", "batch_0490.csv"))
df_21 = pd.read_csv(os.path.join("datasets", "batch_0220.csv"))
df_22 = pd.read_csv(os.path.join("datasets", "batch_0808.csv"))
df_23 = pd.read_csv(os.path.join("datasets", "batch_0472.csv"))
df_24 = pd.read_csv(os.path.join("datasets", "batch_0556.csv"))
df_25 = pd.read_csv(os.path.join("datasets", "batch_0508.csv"))
df_26 = pd.read_csv(os.path.join("datasets", "batch_0520.csv"))
df_27 = pd.read_csv(os.path.join("datasets", "batch_0073.csv"))
df_28 = pd.read_csv(os.path.join("datasets", "batch_0955.csv"))
df_29 = pd.read_csv(os.path.join("datasets", "batch_0451.csv"))
df_30 = pd.read_csv(os.path.join("datasets", "batch_0523.csv"))

