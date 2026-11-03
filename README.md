# Optimizing mAb Production
This code helps to analyze and optimize the space-time profit from production of an mAb production process.

## Highlights
- Selected optimal experiments to analyze from limited data.
- Built python code to create a variety of models and provide data from them.
- Used created models to predict potential economics.
- Analyzed output of models in order to determine optimal initial condtions and harvest time.

## Environment
- Python Version 3.14.7
- Matplotlib 3.11.0
- Numpy 2.5.2
- Scikit-learn 1.9.0

## Code Design
### Data Loading
The program reads 30 batch run CSVs from the datasets directory, each containing an 8-hour-resolution time course of seven process variables (glucose, glutamine, lactate, ammonia, VCD, TCD, mAb) over 672 hours. 
A Batch ID column is inserted as the leftmost column of each dataframe, and the 30 dataframes are concatenated into a single long-format dataframe df with 30 × 85 ≈ 2,550 rows.
Every downstream stage operates on one unified table; separating the batches at load time and concatenating immediately gives a single object that GroupKFold can split by batch identity. 
The Batch ID column is essential not only for grouping but also for joining each row back to its initial conditions when computing economics.
The assembled df is the sole data source for the rest of the pipeline. 

### Model Setup and Analysis Parameters
For each candidate autoregressive order *p* ∈ {1,…,5}, the function generate_x_lag_cur_var builds a supervised learning problem. 
For each batch, the time series is converted into (input, target) pairs where the input is the concatenation of the previous *p* states — [x(t−1), x(t−2), …, x(t−p)], each of length seven — and the target is the current state x(t), also length seven. 
The first *p* rows of each batch are consumed as history; the remaining rows become samples. 
The result is x_lag (n_samples × 7p), x_cur (n_samples × 7), t_cur, and b_cur (batch labels for each sample).

### Model Training


## Analysis
### Overview
- recommended initial glucose concentration [mM]: 20
- recommended initial glutamine concentration [mM]: 4
- recommended initial viable cell density [cells/mL]:  
- recommended harvest time [d]: 4
- estimated cell culture cost [$/L]: 400000
- estimated cell culture value at harvest [$/L] with 95% prediction intervals:
- estimated space-time profit [$ / (L ⋅ d)] with 95% prediction intervals: 
- estimated mAb space-time yield [mg / (L ⋅ d)] with 95% prediction intervals:

### Choice of Experimental Batches
Choosing 30 datasets from a source of 1029 batch datasets to represent the economic limits of testing.
I elected to choose 30 primarily from the center of the dataset as this is where I expected the optimal conditions to be.
The values for low glucose would result in low growth which is likely to make them suboptimal, and high values are linearly more costly without performance necessarily increasing at the same rate.
Low initial biomass means a potentially longer lag phase, and higher means large increase to cost.

### Analysis of Chosen Model
As can be seen from the figures lablled "white_noise" in the "figures" folder, all of the knn models have an increasing white noise over time, which means there is something that they are missing.
The polynomial prediction models by contrast, have no such increase or decrease over time. 

### Analysis of Initial Condition



