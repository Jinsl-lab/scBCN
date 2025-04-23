# scBCN
This work proposes `scBCN`, a batch correction network designed to integrate single-cell data from multiple batches.
## How to install
Before installing scBCN, please install the following packages manually.
### Python Dependencies
`scBCN` depends on the following Python packages:
```text
anndata 0.10.4
anndata2ri 1.3.1
hnswlib 0.8.0
joblib 1.3.2
networkx 3.2.1
numpy 1.26.3
pandas 2.1.4
rpy2 3.5.11
scanpy 1.9.6
scipy 1.11.4
scikit-learn 1.3.2
torch 2.1.2
tqdm 4.66.1
```

### R Dependencies
`scBCN` depends on the following R packages:
```text
batchelor 1.18.1
harmony 1.2.0
ggplot2 3.5.1
rliger 2.1.0
Seurat 4.4.0
SeuratWrappers 0.2.0
SingleCellExperiment 1.24.0
scMC 4.3.2
```

### Installation
`scBCN` is developed under Python (version >= 3.10) and R (version >= 4.3). To use `scBCN`, you can clone this repository:
```text
git clone https://github.com/Jinsl-lab/scBCN.git 
cd scBCN
```
Then install it.
```text
pip install
```
