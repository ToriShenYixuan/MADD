# MADD

Behavior fairness analysis metric for model outputs.\
\
**Data:** predicted probabilities, actual target, and sensitive attributes.\
**Output:** 
- MADD value
- p-value from MADD permutation test (if number of bootstraps specified)
- side-by-side graph of actual normalized density vectors and kernel-smoothed visualization (for comparing learnt bias)

## madd.py
The file that has the madd, graph, and bootstrap functions

## execute_madd.ipynb
The notebook that executes madd.py with guides.
