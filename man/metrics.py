import pandas as pd
import scanpy as sc
import numpy as np
import sklearn
import logging
import anndata2ri
import rpy2
import rpy2.rinterface_lib.callbacks
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
import warnings
warnings.filterwarnings('ignore')
ro.r.source('batchKL.R')
ro.r.source('calLISI.R')
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import pair_confusion_matrix

def calulate_ari_nmi(adata_integrated, n_clusters, random_state=0):
    adata_integrated = adata_integrated.copy()
    sc.pp.neighbors(adata_integrated, random_state=random_state)
    
    def auto_resolution_search(adata, target_clusters, max_iter=50):
        low, high = 0, 1000
        for _ in range(max_iter):
            mid = (low + high) / 2
            sc.tl.leiden(adata, resolution=mid, random_state=random_state)
            found_clusters = adata.obs["leiden"].nunique()
            if found_clusters == target_clusters:
                break
            elif found_clusters < target_clusters:
                low = mid
            else:
                high = mid
        return mid    
    resolution = auto_resolution_search(adata_integrated, n_clusters)
    
    sc.tl.leiden(adata_integrated, resolution=resolution, random_state=random_state)  
    sc.tl.umap(adata_integrated)
    if adata_integrated.X.shape[1]==2:
        adata_integrated.obsm["X_emb"] = adata_integrated.X

    def ari(true_labels, pred_labels):
        (tn, fp), (fn, tp) = pair_confusion_matrix(true_labels, pred_labels)
        tn = int(tn)
        tp = int(tp)
        fp = int(fp)
        fn = int(fn)
        if fn==0 and fp==0:
            return 1.0
        return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    true_labels = adata_integrated.obs["celltype"].astype(str)
    pred_labels = adata_integrated.obs["leiden"]
    ARI = ari(true_labels, pred_labels)
    NMI = normalized_mutual_info_score(true_labels, pred_labels)
    
    print("ARI:", ARI)
    print("NMI:", NMI)
    return ARI, NMI


def calculate_ASW_celltype(adata_integrated, embed="X_emb", metric="euclidean", scale=True):
    if embed not in adata_integrated.obsm.keys():
        raise KeyError(f'{embed} not in obsm')
    asw = sklearn.metrics.silhouette_score(X=adata_integrated.obsm[embed], labels=adata_integrated.obs["celltype"], metric=metric)
    scaled_asw = (asw + 1) / 2 if scale else asw
    print("ASW_celltype =", scaled_asw)
    return scaled_asw


def calulate_BatchKL(adata_integrated):
    with localconverter(ro.default_converter + pandas2ri.converter):
        meta_data = ro.conversion.py2rpy(adata_integrated.obs)

    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    embedding = adata_integrated.obsm["X_emb"]
    KL = ro.r.BatchKL(meta_data, embedding, n_cells=100, batch="BATCH")
    print("BatchKL=", KL)
    numpy2ri.deactivate()
    return KL


def calulate_LISI(adata_integrated):
    with localconverter(ro.default_converter + pandas2ri.converter):
        meta_data = ro.conversion.py2rpy(adata_integrated.obs)

    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    embedding = adata_integrated.obsm["X_emb"]
    lisi = ro.r.CalLISI(embedding, meta_data)
    numpy2ri.deactivate()
    print("ilisi=", lisi[1])
    return lisi


### 总：ARI、NMI、ASW_label、BatchKL、iLISI
def evaluate_dataset(adata_integrated, method="leiden2.0", n_cluster=3):
    print("...... method={} ......".format(method))
    ARI, NMI = calulate_ari_nmi(adata_integrated, n_clusters=n_cluster)
    ASW_celltype = calculate_ASW_celltype(adata_integrated)
    KL = calulate_BatchKL(adata_integrated)
    lisi = calulate_LISI(adata_integrated)

    results = {'ARI': np.round(ARI, 3),
               'NMI': np.round(NMI, 3),
               'ASW_celltype': np.round(ASW_celltype, 3),
               'BatchKL': np.round(KL[0], 3),
               'iLISI': np.round(lisi[1], 3)
              }

    result = pd.DataFrame.from_dict(results, orient='index')
    result.columns = [method]
    return adata_integrated, result