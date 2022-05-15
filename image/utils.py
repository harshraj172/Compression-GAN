import numpy as np
from sklearn.decomposition import PCA

import torch

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def reduce_dim(feat, method="PCA", dim=32):
    if method=="PCA":
        pca = PCA(n_components=dim)
        reduced_feat = pca.fit_transform(np.array(feat))
        return Tensor(reduced_feat)