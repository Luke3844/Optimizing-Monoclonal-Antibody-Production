# Optimizing mAb Production
This code helps to analyze and optimize the space-time profit from production of an mAb production process.

## Highlights
- Selected optimal experiments to analyze from limited data.
- Built python code to create a variety of models and provide data from them.
- Analyzed created models for accuracy and economic efficiency.

## Environment
- Python Version 3.14.7
- Matplotlib 3.11.0
- Numpy 2.5.2
- Scikit-learn 1.9.0

## Code Design
- Explain each section of code.

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



