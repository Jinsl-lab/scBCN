import os
import scanpy as sc
import pandas as pd
import numpy as np 
from numpy import linalg as LA
from numpy.linalg import matrix_power
import scipy
from scipy.sparse import issparse
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph 
from anndata import AnnData
import networkx as nx

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, miners, reducers, distances

import random
import hnswlib
import itertools
import logging
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')


######################## 1、Data Preprocessing ########################
###### 1.0 Utility Functions
### 1.0.1 Logger Configuration
def create_logger(name='', ch=True, fh='', levelname=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(levelname)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if ch:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    if fh:
        fh = logging.FileHandler(fh, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    return logger

### 1.0.2 Dataset Information Print
def print_dataset_information(adata, batch_key="BATCH", celltype_key="celltype", log=None):
    if batch_key is not None and batch_key not in adata.obs.columns:
        print('Please check whether there is a {} column in adata.obs to identify batch information!'.format(batch_key))
        raise IOError

    if celltype_key is not None and batch_key not in adata.obs.columns:
        print('Please check whether there is a {} column in adata.obs to identify celltype information!'.format(celltype_key))
        raise IOError

    if(log is not None):
        log.info("====== print brief infomation of dataset ======")
        log.info("====== there are {} batchs in this dataset ======".format(len(adata.obs[batch_key].value_counts())))
        log.info("====== there are {} celltypes with this dataset ======".format(len(adata.obs[celltype_key].value_counts())))
    else:
        print("====== print brief infomation of dataset ======")
        print("====== there are {} batchs in this dataset ======".format(len(adata.obs[batch_key].value_counts())))
        print("====== there are {} celltypes with this dataset ======".format(len(adata.obs[celltype_key].value_counts())))
    data_info = pd.crosstab(adata.obs[batch_key], adata.obs[celltype_key], margins=True, margins_name="Total")
    display(data_info)


###### 1.1 Data Normalization
def Normalization(adata, batch_key="BATCH", n_high_var=2000, hvg_list=None, normalize_samples=True, target_sum=1e4, log_normalize=True, normalize_features=True, scale_value=10.0, verbose=True, log=None):
    n, p = adata.shape
    if normalize_samples:
        if verbose:
            log.info("Normalize counts per cell(sum={})".format(target_sum))
        sc.pp.normalize_total(adata, target_sum=target_sum)
        
    if log_normalize:
        if verbose:
            log.info("Log1p data")
        sc.pp.log1p(adata)
    
    if hvg_list is None:      
        if verbose:
            log.info("Select HVG(n_top_genes={})".format(n_high_var))
        sc.pp.highly_variable_genes(adata, n_top_genes=n_high_var, subset=True)
    else:
        log.info("Select HVG from given highly variable genes list")
        adata = adata[:, hvg_list]
    
    adata.obs["batch"] = "1"
    if normalize_features:
        if len(adata.obs[batch_key].value_counts())==1:  
            if verbose:
                log.info("Scale batch(scale_value={})".format(scale_value))
            sc.pp.scale(adata, max_value=scale_value)
            adata.obs["batch"] = 1
        else:
            if verbose:
                log.info("Scale batch(scale_value={})".format(scale_value))
            adata_sep = []
            for batch in np.unique(adata.obs[batch_key]):
                sep_batch = adata[adata.obs[batch_key]==batch]
                sc.pp.scale(sep_batch, max_value=scale_value)
                adata_sep.append(sep_batch)
            adata = sc.AnnData.concatenate(*adata_sep)
    
    return adata
  
###### 1.2 Dimension Reduction
def dimension_reduction(adata, dim=100, verbose=True, log=None):
    if verbose:
        log.info("Calculate PCA(n_comps={})".format(dim))
        
    if adata.shape[0]>300000:
        adata.X = scipy.sparse.csr_matrix(adata.X)
    sc.tl.pca(adata, n_comps=dim)
    emb = sc.AnnData(adata.obsm["X_pca"])
    
    return emb

###### 1.3 Initial Clustering
def init_clustering(emb, reso=3.0, cluster_method="leiden", verbose=False, log=None):
    if cluster_method=="leiden":
        sc.pp.neighbors(emb, random_state=0)
        sc.tl.leiden(emb, resolution=reso, key_added="init_cluster")
        if verbose:
            log.info("Apply leiden clustring(resolution={})  initization".format(reso))
            log.info("Number of Cluster ={}".format(len(emb.obs["init_cluster"].value_counts())))
            log.info("clusters={}".format([i for i in range(len(emb.obs["init_cluster"].value_counts()))]))
    else:
        if verbose:
            log.info("Not implemented!!!")
        raise IOError


######################## 2、Cross-batch Cell Clustering ########################
###### 2.0 Input Data Format Check   
def checkInput(adata, batch_key, log):
    if not isinstance(adata, AnnData):
        log.info("adata is not an object of AnnData, please convert Input data to Anndata")
        raise IOError
    
    if batch_key is not None and batch_key not in adata.obs.columns:
        log.info('Please check whether there is a {} column in adata.obs to identify batch information!'.format(batch_key))
        raise IOError
    elif batch_key is None:
        log.info("scBCN cretate \"BATCH\" column to set all cell to one batch!!!")
        batch_key = "BATCH"
        adata.obs[batch_key] = "1"
    
    if adata.obs[batch_key].dtype.name!="categroy":
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    return batch_key

###### 2.1 Mutual Nearest Neighbors (MNN) Pairs Identification
### 2.1.1 Exact Nearest Neighbors Search
def nn(ds1, ds2, names1, names2, k):
    nn_ = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b[0:]:
            match.add((names1[a], names2[b_i]))
    
    return match

### 2.1.2 Approximate Nearest Neighbors Search
def nn_approx(ds1, ds2, names1, names2, k):
    dim = ds2.shape[1]
    tree = hnswlib.Index(space="cosine", dim=dim)
    tree.init_index(max_elements=ds2.shape[0], ef_construction=200, M=32)
    tree.set_ef(50)
    tree.add_items(ds2)
    ind, distances = tree.knn_query(ds1, k=k)
    match = set()  # 集合
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b[0:]:  #
            match.add((names1[a], names2[b_i]))
    
    return match

### 2.1.3 Mutual Nearest Neighbors with Random Walk Expansion
def mnn(ds1, ds2, names1, names2, k, walk_steps, approx=True):
    if approx:
        match1 = nn_approx(ds1, ds2, names1, names2, k=k)
        match2 = nn_approx(ds2, ds1, names2, names1, k=k)
    else:
        match1 = nn(ds1, ds2, names1, names2, k=k)
        match2 = nn(ds2, ds1, names2, names1, k=k)
    mutual = match1 & set([(b, a) for a, b in match2])
    mutual = mutual | set([(b, a) for (a, b) in mutual])
    
    anchor_pairs_dict = {}
    query_pairs_dict = {}
    for pair in mutual:
        anchor, query = pair
        if anchor not in anchor_pairs_dict:
            anchor_pairs_dict[anchor] = []
        if query not in query_pairs_dict:
            query_pairs_dict[query] = []
        anchor_pairs_dict[anchor].append(query)
        query_pairs_dict[query].append(anchor)       
    pair_plus = []
    for x, y in mutual:
        start = (x, y)
        for i in range(walk_steps):
            pair_plus.append(start)
            start = (random.choice(anchor_pairs_dict[start[0]]), random.choice(query_pairs_dict[start[1]])) 
    mutual = set(pair_plus)
    
    return mutual

### 2.1.4 Batch-wise MNN Identification (Serial Version)
def get_mnns(data_matrix, batch_index, k, walk_steps, approx=True, verbose=True, log=None):
    cell_names = np.array(range(len(data_matrix)))
    batch_unique = np.unique(batch_index)
    cells_batch = []
    for i in batch_unique:
        cells_batch.append(cell_names[batch_index == i])
    mnns = set()
    mnns_num = 0
    if verbose:
        log.info("Calculate MNN pairs intra batch and inter batch...........")
        log.info("k={}".format(k))
    combinations = [(i, j) if i <= j else (j, i) for i, j in itertools.combinations_with_replacement(range(len(cells_batch)), 2)]
    for comb in combinations:
        i = comb[0]  
        j = comb[1]  
        if verbose:
            i_batch = batch_unique[i]
            j_batch = batch_unique[j]
            log.info("Processing datasets: {} = {}".format((i, j), (i_batch, j_batch)))
        target = list(cells_batch[j])
        ref = list(cells_batch[i])
        ds1 = data_matrix[target]
        ds2 = data_matrix[ref]
        names1 = target
        names2 = ref
        mutual = mnn(ds1, ds2, names1, names2, k=k, walk_steps=walk_steps, approx=approx)
        mnns = mnns | mutual
        if verbose:
            log.info("There are ({}) MNN pairs when processing {}={}".format(len(mutual), (i, j), (i_batch, j_batch)))
            mnns_num = mnns_num + len(mutual)
    if verbose:
        log.info("scBCN finds ({}) MNN pairs in dataset finally".format(mnns_num))
    
    return mnns

### 2.1.5 Batch-wise MNN Identification (Parallel Version)
def get_mnns_para(data_matrix, batch_index, k, walk_steps, approx=True, verbose=True, njob=8, log=None):
    cell_names = np.array(range(len(data_matrix)))
    batch_unique = np.unique(batch_index)
    cells_batch = []
    for i in batch_unique:
        cells_batch.append(cell_names[batch_index == i])
    if verbose:
        log.info("Calculate MNN pairs intra batch and inter batch...........")
        log.info("k={}".format(k))
    combinations = [(i, j) if i <= j else (j, i) for i, j in itertools.combinations_with_replacement(range(len(cells_batch)), 2)]
    res = Parallel(n_jobs=njob)(
        delayed(mnn)(data_matrix[list(cells_batch[comb[1]])], data_matrix[list(cells_batch[comb[0]])],
                     list(cells_batch[comb[1]]), list(cells_batch[comb[0]]), k=k, walk_steps=walk_steps, approx=approx) for comb in combinations)
    mnns = set(list(itertools.chain(*res)))
    if verbose:
        log.info("scBCN finds ({}) MNN pairs in dataset finally".format(len(mnns)))
    
    return mnns

###### 2.2 Construction of Similarity Matrix
def calculate_similarity_matrix(mnn_pairs, cluster_label, verbose, log):
    mnn_pairs_array = np.array(list(mnn_pairs))
    cluster_set = range(len(np.unique(cluster_label)))
    if len(mnn_pairs_array)==0:
        mnn_summary = pd.DataFrame(0, index=cluster_set, columns=cluster_set)
    else:
        mnn_pairs_df = pd.DataFrame({"pair1_clust": cluster_label[mnn_pairs_array[:, 0]].astype(int),
                                     "pair2_clust": cluster_label[mnn_pairs_array[:, 1]].astype(int),
                                     "pair1": mnn_pairs_array[:, 0],
                                     "pair2": mnn_pairs_array[:, 1]})
        mnn_summary = mnn_pairs_df.pivot_table(index="pair1_clust", columns="pair2_clust", aggfunc='size', fill_value=0)
        if not np.all(mnn_summary.values == mnn_summary.values.T):
            log.info("mnn_summary is not symmetric!")
        else:
            log.info("mnn_summary is symmetric!")
    
    mnn_table = pd.DataFrame(0, index=cluster_set, columns=cluster_set)
    for ind in list(mnn_summary.index):
        for col in list(mnn_summary.columns):
            mnn_table.loc[ind, col] = mnn_summary.loc[ind, col]
    np.fill_diagonal(mnn_table.values, 0)
    if verbose:
        log.info("{} MNN pairs link different clusters".format(np.sum(mnn_table.values)))
    sum_matrix = mnn_table.values
    mnn_cor = np.zeros_like(sum_matrix, dtype=float)
    clu_size = np.array(cluster_label.value_counts())
    for i in range(len(mnn_cor)):
        for j in range(len(mnn_cor)):
            mnn_cor[i, j] = sum_matrix[i, j].astype(float) / np.sqrt(clu_size[i] * clu_size[j])
    cor = pd.DataFrame(data=mnn_cor)
    cor_mat = cor.values
    np.fill_diagonal(cor_mat, 0)
    
    return cor, mnn_table

###### 2.3 Spectral Clustering Construction
### 2.3.1 Spectral Clustering
def spectral_cluster(G, n_cluster):
    adj_matrix = csr_matrix(nx.to_numpy_array(G))
    spectral = SpectralClustering(n_clusters=n_cluster, affinity='precomputed', random_state=42, n_init=50)
    labels = spectral.fit_predict(adj_matrix)
    
    return labels

### 2.3.2 Spectral Clustering Merging
def merge_spectral_cluster(cor, n_cluster=None, save_dir=None):
    normalized_cor = normalize(cor, norm='l2', axis=1)
    G = nx.Graph()
    num_nodes = cor.shape[0]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            weight = normalized_cor[i, j]
            G.add_edge(i, j, weight=weight)
    if n_cluster is None:
        from sklearn.cluster import KMeans
        eigenvalues = np.linalg.eigvalsh(nx.to_numpy_array(G))
        n_cluster = np.argmax(np.diff(np.sort(eigenvalues)[::-1])) + 1
    
    labels = spectral_cluster(G=G, n_cluster=n_cluster)
    
    return labels

### 2.3.3 Eigendecomposition of Laplacian Matrix of Graph
def eigenDecomposition(A, plot = True, topK =5, save_dir=None):
    L = csgraph.laplacian(A, normed=True)
    n_components = A.shape[0]
    eigenvalues, eigenvectors = LA.eig(L)  
    if plot:
        fig = plt.figure(figsize=(15, 12))
        plt.title('Largest eigen values of input matrix')
        plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
        plt.grid()
        plt.savefig(save_dir + "Estimate_number_of_cluster")
        plt.show()
        
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:topK]
    nb_clusters = index_largest_gap + 1
        
    return nb_clusters, eigenvalues, eigenvectors


######################## 3、Batch Correction Network Construction ########################
###### 3.1  Residual Block
import importlib
import torch.nn
nn = importlib.reload(torch.nn)
class ResidualBlock(nn.Module):
    def __init__(self, in_sz, out_sz):
        super(ResidualBlock, self).__init__()
        self.in_sz = in_sz
        self.out_sz = out_sz

        self.linear1 = nn.Linear(self.in_sz, self.out_sz)
        self.bn1 = nn.BatchNorm1d(self.out_sz)
        self.relu = nn.PReLU()
        self.linear2 = nn.Linear(self.out_sz, self.out_sz)
        self.bn2 = nn.BatchNorm1d(self.out_sz)

    def forward(self, x):
        identity = x 
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

###### 3.2 Residual Network
class ResidualNet(nn.Module):
    def __init__(self, in_sz, emb_szs, out_sz, n_blocks):
        super(ResidualNet, self).__init__()
        self.in_sz = in_sz
        self.emb_szs = emb_szs
        self.out_sz = out_sz
        self.n_blocks = n_blocks

        self.linear1 = nn.Linear(self.in_sz, self.emb_szs)
        self.relu = nn.PReLU()
        self.blocks = nn.ModuleList([ResidualBlock(self.emb_szs, self.emb_szs) for _ in range(self.n_blocks)])
        self.linear2 = nn.Linear(self.emb_szs, self.out_sz)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        for block in self.blocks:
            out = block(out)
        out = self.linear2(out)
        return out

###### 3.3 Training
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.badatahmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    

######################## scBCNMOdel！！！ ########################
class scBCNModel:
    def __init__(self, verbose=True, save_dir="./results/"):
        self.verbose = verbose
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir+"/")
        self.log = create_logger('', fh=self.save_dir+'log.txt')  
        if self.verbose:
            self.log.info("Create log file....")  
            self.log.info("Create scBCNModel Object Done....")


    ###  preprocess raw data to generate init cluster label
    def preprocess(self, adata, batch_key="BATCH", n_high_var=2000, hvg_list=None, normalize_samples=True, target_sum=1e4, log_normalize=True, normalize_features=True, pca_dim=100, scale_value=10.0, resolution=3.0):
        batch_key = checkInput(adata, batch_key, self.log)
        self.batch_key = batch_key
        self.reso = resolution
        self.nbatch = len(adata.obs[batch_key].value_counts())
        if self.verbose:
            self.log.info("Running preprocess() function...")
            self.log.info("Leiden (resolution={})".format(str(resolution)))
            self.log.info("BATCH_key={}".format(str(batch_key)))

        self.norm_args = (batch_key, n_high_var, hvg_list, normalize_samples, target_sum, log_normalize, normalize_features, scale_value, self.verbose, self.log)
        normalized_adata = Normalization(adata, *self.norm_args)
        emb = dimension_reduction(normalized_adata, pca_dim, self.verbose, self.log)
        init_clustering(emb, reso=self.reso, verbose=self.verbose, log=self.log)

        self.batch_index = normalized_adata.obs[batch_key].values
        normalized_adata.obs["init_cluster"] = emb.obs["init_cluster"].values.copy()
        self.num_init_cluster = len(emb.obs["init_cluster"].value_counts())
        if self.verbose:
            self.log.info("Preprocess Dataset Done...")
            
            return normalized_adata


    ### convert normalized adata to training data for scBCN
    def convertInput(self, adata, batch_key="BATCH", celltype_key=None):
        checkInput(adata, batch_key=batch_key, log=self.log)  # check batch
        if "X_pca" not in adata.obsm.keys():  # check pca
            sc.tl.pca(adata)
        if "init_cluster" not in adata.obs.columns:  # check init clustering
            sc.pp.neighbors(adata, random_state=0)
            sc.tl.leiden(adata, key_added="init_cluster", resolution=2.0)

        if issparse(adata.X):
            self.train_X = adata.X.toarray()
        else:
            self.train_X = adata.X.copy()
        self.nbatch = len(adata.obs[batch_key].value_counts())
        self.train_label = adata.obs["init_cluster"].values.copy()
        self.emb_matrix = adata.obsm["X_pca"].copy()
        self.batch_index = adata.obs[batch_key].values
        self.merge_df = pd.DataFrame(adata.obs["init_cluster"])
        if self.verbose:
            self.merge_df.value_counts().to_csv(self.save_dir+"cluster_distribution.csv")
        if celltype_key is not None:
            self.celltype = adata.obs[celltype_key].values
        else:
            self.celltype=None

       
    ### calculate connectivity
    def calculate_similarity(self, k=10, walk_steps=20, approx=True):
        self.k = k
        self.walk_steps = walk_steps
        self.approx = approx
        if self.verbose:
            self.log.info("k={}".format(k))
            self.log.info("Calculate similarity of cluster with MNN")
        if self.nbatch < 10:
            if self.verbose:
                self.log.info("Appoximate calculate MNN pairs intra batch and inter batch...")
            mnn_pairs = get_mnns(data_matrix=self.emb_matrix, batch_index=self.batch_index, k=k, walk_steps=walk_steps, approx=approx, verbose=self.verbose, log=self.log)
            if self.verbose:
                self.log.info("Find All Nearest Neighbours Done....")
        else:
            if self.verbose:
                self.log.info("Calculate MNN pairs in parallel mode to accelerate...")
                self.log.info("Appoximate calculate MNN Pairs intra batch and inter batch...")
            mnn_pairs = get_mnns_para(data_matrix=self.emb_matrix, batch_index=self.batch_index, k=k, walk_steps=walk_steps, approx=approx, verbose=self.verbose, log=self.log)
            if self.verbose:
                self.log.info("Find all nearest neighbours done!")

        if self.verbose:
            self.log.info("Calculate similarity matrix between clusters")
        self.cor_matrix, self.nn_matrix = calculate_similarity_matrix(mnn_pairs=mnn_pairs, cluster_label=self.train_label, verbose=self.verbose, log=self.log)
        if self.verbose:
            self.log.info("Save correlation matrix to file....")
            self.cor_matrix.to_csv(self.save_dir+"cor_matrix.csv")
            self.log.info("Save MNN pairs matrix to file")
            self.nn_matrix.to_csv(self.save_dir+"mnn_matrix.csv")
            self.log.info("Calculate similarity matrix done!")

        if self.celltype is not None:
            same_celltype = self.celltype[mnn_pairs[:, 0]]==self.celltype[mnn_pairs[:, 1]]
            equ_pair = sum(same_celltype)
            self.log.info("the number of mnn pairs which link same celltype is {}".format(equ_pair))
            df = pd.DataFrame({"celltype_pair1": self.celltype[mnn_pairs[:, 0]],
                               "celltype_pair2": self.celltype[mnn_pairs[:, 1]]})
            num_info = pd.crosstab(df["celltype_pair1"], df["celltype_pair2"], margins=True, margins_name="Total")
            ratio_info_row = pd.crosstab(df["celltype_pair1"], df["celltype_pair2"]).apply(lambda r: r/r.sum(), axis=1)
            ratio_info_col = pd.crosstab(df["celltype_pair1"], df["celltype_pair2"]).apply(lambda r: r/r.sum(), axis=0)
            num_info.to_csv(self.save_dir+"mnn_pair_num_info.csv")
            ratio_info_row.to_csv(self.save_dir+"mnn_pair_ratio_info_raw.csv")
            ratio_info_col.to_csv(self.save_dir+"mnn_pair_ratio_info_col.csv")
            self.log.info(num_info)
            self.log.info(ratio_info_row)
            self.log.info(ratio_info_col)

        return mnn_pairs, self.cor_matrix, self.nn_matrix


    ### merge small cluster to larger cluster and reassign cluster label
    def merge_cluster(self, n_cluster):
        self.nc_list = pd.DataFrame()
        df = self.merge_df.copy()
        df["value"] = np.ones(self.train_X.shape[0])
        map_list = merge_spectral_cluster(cor=self.cor_matrix.copy(), n_cluster=n_cluster)
        map_dict = {str(i): map_list[i] for i in range(len(map_list))}
        reverse_map = defaultdict(list)     
        for key, value in map_dict.items():
            reverse_map[value].append(key)
        map_set = list(reverse_map.values())
        self.merge_df["nc_"+str(n_cluster)] = self.merge_df["init_cluster"].map(map_dict)
        df[str(n_cluster)] = str(n_cluster) + "(" + self.merge_df["nc_"+str(n_cluster)].astype(str) + ")"
        if self.verbose:
            self.log.info("Merging cluster list:" + str(map_set))
        
        return df


    ### build network for scBCN training
    def build_net(self, in_dim=2000, emb_dim=256, out_dim=32, n_blocks=2, seed=1029):
        if in_dim != self.train_X.shape[1]:
            in_dim = self.train_X.shape[1]
        if(self.verbose):
            self.log.info("Build ResidualNet for scBCN training")
        seed_torch(seed)
        self.model = ResidualNet(in_sz=in_dim, emb_szs=emb_dim, out_sz=out_dim, n_blocks=n_blocks)
        if self.verbose:
            self.log.info(self.model)
            self.log.info("Build ResidualNet Net Done...")

    ### train scBCN to correct batch effect
    def train(self, expect_num_cluster=None, num_epochs=50, batch_size=64, metric="cosine", margin=5, scale=32, pos_eps=0.01, neg_eps=0.01, weights=[1, 0.5], device=None, save_model=False):
        if expect_num_cluster is None:
            if self.verbose:
                self.log.info("expect_num_cluster is None, use eigen value gap to estimate the number of celltype.")
            cor_matrix = self.cor_matrix.copy()
            for i in range(len(cor_matrix)):
                cor_matrix.loc[i, i] = 0.0
                A = cor_matrix.values / np.max(cor_matrix.values)  # normalize similarity matrix to [0,1]
                norm_A = A + matrix_power(A, 2)
                for i in range(len(A)):
                    norm_A[i, i] = 0.0

            k, _,  _ = eigenDecomposition(norm_A, save_dir=self.save_dir)
            self.log.info(f'Optimal number of clusters {k}')
            expect_num_cluster = k[0]
        if "nc_"+str(expect_num_cluster) not in self.merge_df:
            self.log.info("scBCN can't find the mering result of cluster={} ,you can run merge_cluster(ncluster_list=[{}]) function to get this".format(expect_num_cluster, expect_num_cluster))
            raise IOError
        self.train_label = self.merge_df["nc_"+str(expect_num_cluster)].values.astype(int)

        if os.path.isfile(os.path.join(self.save_dir, "scBCN_model.pkl")):
            self.log.info("Loading trained model...")
            self.model = torch.load(os.path.join(self.save_dir, "scBCN_model.pkl"))
        else:
            if self.verbose:
                self.log.info("train scBCN(expect_num_cluster={}) with ResidualNet".format(expect_num_cluster))
                self.log.info("expect_num_cluster={}".format(expect_num_cluster))
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if self.verbose:
                    if torch.cuda.is_available():
                        self.log.info("using GPU to train model")
                    else:
                        self.log.info("using CPU to train model")
            train_set = torch.utils.data.TensorDataset(torch.FloatTensor(self.train_X), torch.from_numpy(self.train_label).long())
            train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True, drop_last=True)
            self.model = self.model.to(device)
            optimizer = optim.Adam(self.model.parameters(), lr=0.01)
            if metric=="cosine":
                distance = distances.CosineSimilarity()  # use cosine_similarity()
            elif metric=="euclidean":
                distance = distances.LpDistance(p=2, normalize_embeddings=False)  # use euclidean distance
            else:
                self.log.info("Not implemented,to be updated")
                raise IOError

            reducer = reducers.MeanReducer()
            main_loss = losses.TupletMarginLoss(margin=margin, scale=scale, distance=distance, reducer=reducer)
            var_loss = losses.IntraPairVarianceLoss(pos_eps=pos_eps, neg_eps=neg_eps, distance=distance, reducer=reducer)
            loss_func = losses.MultipleLosses([main_loss, var_loss], weights=weights)
            mining_func = miners.MultiSimilarityMiner()
            if self.verbose:
                self.log.info("use {} distance to train model".format(metric))
            mined_epoch_loss = np.array([])
            for epoch in range(1, num_epochs+1):
                temp_num_loss = 0
                self.model.train() 
                for batch_idx, (train_data, training_labels) in enumerate(train_loader):
                    train_data, training_labels = train_data.to(device), training_labels.to(device)  
                    optimizer.zero_grad()  
                    embeddings = self.model(train_data)  
                    hard_tuple = mining_func(embeddings, training_labels)  
                    loss = loss_func(embeddings, training_labels, hard_tuple)
                    temp_num_loss = temp_num_loss + hard_tuple[0].size(0)
                    loss.backward()  
                    optimizer.step()  
                mined_epoch_loss = np.append(mined_epoch_loss, temp_num_loss)
                if self.verbose:
                    self.log.info("epoch={}, loss={}".format(epoch, mined_epoch_loss))

            if self.verbose:
                self.log.info("scBCN training done....")
            if save_model:
                if self.verbose:
                    self.log.info("save model....")
                torch.save(self.model.to(torch.device("cpu")), os.path.join(self.save_dir, "scBCN_model.pkl"))
            self.loss = mined_epoch_loss
        features = self.predict(self.train_X)
        
        return features


    ### predict data matrix (produce embedding)
    def predict(self, X, batch_size=128):
        if self.verbose:
            self.log.info("extract embedding for dataset with trained network")
        device = torch.device("cpu")
        dataloader = DataLoader(torch.FloatTensor(X), batch_size=batch_size, pin_memory=False, shuffle=False)
        data_iterator = tqdm(dataloader, leave=False, unit="batch")
        self.model = self.model.to(device)
        with torch.no_grad():
            self.model.eval()
            features = []
            for batch in data_iterator:
                batch = batch.to(device)
                output = self.model(batch)
                features.append(output.detach().cpu())  
            features = torch.cat(features).cpu().numpy()
        
        return features

    ##### integration for scBCN ！！！
    def integrate(self, adata, batch_key="BATCH", n_cluster=None, expect_num_cluster=None, k=10, walk_steps=20, num_epochs=50, batch_size=64, metric="cosine", margin=5, scale=32, pos_eps=0.01, neg_eps=0.01, weights=[1, 0.5], device=None, seed=1029, emb_dim=256, out_dim=32, n_blocks=2, save_model=False, celltype_key=None):
        
        self.convertInput(adata, batch_key=batch_key, celltype_key=celltype_key)
        self.calculate_similarity(k=k, walk_steps=walk_steps)
        self.merge_cluster(n_cluster=n_cluster)
        self.build_net(out_dim=out_dim, emb_dim=emb_dim, n_blocks=n_blocks, seed=seed)
        features = self.train(expect_num_cluster=expect_num_cluster, num_epochs=num_epochs, batch_size=batch_size, metric=metric, margin=margin, scale=scale, pos_eps=pos_eps, neg_eps=neg_eps, weights=weights, device=device, save_model=save_model)
           
        adata.obsm["X_emb"] = features
        adata.obs["reassign_cluster"] = self.train_label.astype(int).astype(str)
        adata.obs["reassign_cluster"] = adata.obs["reassign_cluster"].astype("category")