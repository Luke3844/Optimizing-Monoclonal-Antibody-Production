# Optimizing mAb Production
This code helps to analyze and optimize the space-time profit from production of an mAb production process.

## Highlights
- Selected optimal experiments to analyze from limited data.
- Built python code to create a variety of models and provide data from them.
- Analyzed created models for accuracy and economic efficiency.

## Environment
- Python Version 3.14.7
- Add rest after project finished

## Code Design
- Explain each section of code.

## Analysis
### Overview
- recommended initial glucose concentration [mM].
- recommended initial glutamine concentration [mM].
- recommended initial viable cell density [cells/mL]. 
- recommended harvest time [d] (i.e., when to stop the production).
- estimated cell culture cost [$/L]. 
- estimated cell culture value at harvest [$/L] with 95% prediction intervals. 
- estimated space-time profit [$ / (L ⋅ d)] with 95% prediction intervals. 
- estimated mAb space-time yield [mg / (L ⋅ d)] with 95% prediction intervals.

### Choice of Batches
Choosing 30 datasets from a source of 1029 batch datasets to represent the economic limits of testing.
I elected to choose 30 primarily from the center of the dataset as this is where I expected the optimal conditions to be.
The values for low glucose would result in low growth which is likely to make them suboptimal, and high values are linearly more costly without performance necessarily increasing at the same rate.
Low initial biomass means a potentially longer lag phase, and higher means large increase to cost.



### Analysis of Chosen Model
Justify chosen model

### Analysis of Initial Condition
Analyze effects of different initial conditions
