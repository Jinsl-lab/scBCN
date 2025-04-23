This is an introduction document about the `scBCN` project. The following is a description of the purpose of each file:
### scBCN.py
This python script implements the scBCN algorithm, which is used for integrating multi-source single-cell data, correcting batch effects, and conservative biological variation.
### metrics.py
This Python script contains functions to calculate various evaluation metrics, which assess the performance of the model in data integration tasks. It calls `batchKL.R` and `callLISI.R` for calculating specific metrics related to batch effects and clustering quality.
