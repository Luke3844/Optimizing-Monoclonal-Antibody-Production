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
The program loops over two model families — polynomial regression with PolynomialFeatures and k-nearest-neighbours regression — and over *p* ∈ {1,…,5}. For each (family, *p*) pair, it constructs a scikit-learn pipeline consisting of StandardScalerVAR(p) followed by the estimator. 
The custom scaler computes per-variable mean and standard deviation across all lag blocks and tiles them so that every lag of the same variable is scaled identically; this is essential because glucose and VCD differ by six orders of magnitude and a naive scaler would be dominated by VCD. 
GroupKFold(n_splits=3) ensures folds are batch-disjoint. GridSearchCV tunes the polynomial degree (1–3) or the number of neighbours (1–5) and the weighting scheme, scoring by negative root-mean-squared error. 
The white-noise diagnostic in this stage tests whether the model has learned the one-step dynamics well enough that residuals are structureless. 
If they are not, no downstream forecast can be trusted. The comparison across two model families guards against committing to a single inductive bias: polynomial regression captures smooth global trends but cannot represent regime changes; k-NN captures local dynamics but extrapolates poorly outside the training manifold.
The best (family, p, hyperparameters) triple is retained as best_overall_model, which is the object used for recursive forecasting in the final stage. 
The residual plots saved to figures/figure_white_noise_*.png are the primary evidence that the model is adequate.

### Space Time Profit Analysis
For each of the 30 batches, the fitted pipeline is used by simulate_batch to recursively forecast the seven variables forward from the batch's observed initial state. 
At each step, the model predicts the next state from the previous *p* states; predictions are fed back as inputs until the full 672-hour time grid is filled. 
For each batch, the predicted mAb trajectory is combined with the batch's initial conditions to compute cost, revenue, space-time profit STP(*t*) and space-time yield STY(*t*) at every time point. 
The harvest time t* is the time that maximizes STP; the harvest time maximizing STY is recorded separately. 
Results are ranked by STP* and exported to figures/stp_summary.csv. 
This stage produces the deliverable of the project: a ranked recommendation of initial conditions and harvest times under the stated economics. 
It also produces the diagnostic evidence (end-of-run predicted vs. observed mAb) needed to judge whether the recommendation is trustworthy.

## Analysis
### Overview
- recommended initial glucose concentration [mM]: 20
- recommended initial glutamine concentration [mM]: 4
- recommended initial viable cell density [cells/mL]:  
- recommended harvest time [d]: 4
- estimated cell culture cost [$/L]: 400000
- estimated cell culture value at harvest [$/L] with 95% prediction intervals: 867.79 (706.78, 971.71)
- estimated space-time profit [$ / (L ⋅ d)] with 95% prediction intervals: 86.53 (52.02, 108.79)
- estimated mAb space-time yield [mg / (L ⋅ d)] with 95% prediction intervals: 0.227 (0.151, 0.208)

### Choice of Experimental Batches
Choosing 30 datasets from a source of 1029 batch datasets to represent the economic limits of testing.
I elected to choose 30 primarily from the center of the dataset as this is where I expected the optimal conditions to be.
The values for low glucose would result in low growth which is likely to make them suboptimal, and high values are linearly more costly without performance necessarily increasing at the same rate.
Low initial biomass means a potentially longer lag phase, and higher means large increase to cost.

### Analysis of Chosen Model
The process is modelled as a vector autoregression of order *p*: the state at time *t*, comprising all seven measured variables, is predicted from the states at *t*−1, …, *t*−*p*. 
This embedding converts a multivariate time-series forecasting problem into a standard supervised regression problem, which allows the use of well-understood scikit-learn estimators.

Two estimator families were compared: polynomial regression (degrees 1–3 applied to the scaled lagged features, followed by ordinary least squares) and k-nearest-neighbour regression (*k* = 1–5, uniform and distance weighting). 
Both were wrapped in a pipeline with a custom StandardScalerVAR(p) preprocessor, which computes per-variable mean and standard deviation across all lag blocks and tiles them so that every lag of the same variable is scaled identically. 
This is essential because glucose (~5 mM) and VCD (~10⁶ cells/mL) differ by six orders of magnitude; a naively fitted scaler would be dominated by VCD and would distort the regression.

Polynomial regression was expected to be the safer choice for this application. The model is used recursively — predictions are fed back as inputs for 84 consecutive steps to reach the harvest time — so it must remain stable far from any single training point. 
A low-degree polynomial cannot produce the wild extrapolations that an unconstrained model might, whereas k-NN has no extrapolation capability at all and degenerates once a predicted state leaves the training manifold. 
The comparison against k-NN therefore tests whether the richer local structure of the data justifies the extrapolation risk.

Hyperparameters were searched by GridSearchCV with GroupKFold(n_splits=3), using negative root-mean-squared error as the scoring function. 
Grouped folds were used because samples from the same batch are serially correlated; random splitting would leak information from later to earlier time points of the same batch and inflate the apparent accuracy. 
The hyperparameter grid was deliberately modest (polynomial degree ≤ 3, *k* ≤ 5, autoregressive order ≤ 5) because the design has only 30 batches and roughly 2,500 samples; higher-capacity models would overfit the batch-level structure.

### Analysis of Initial Condition



