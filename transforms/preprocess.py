# Copyright 2022 DSE lab.  All rights reserved.
import anndata
import cv2
import scipy.sparse as sp
from scipy.sparse import spmatrix
from scipy import stats
from scipy.sparse import csr_matrix, vstack, load_npz, save_npz
from sklearn.decomposition import PCA
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.preprocessing
import sklearn.neighbors
import sklearn.utils.extmath
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
import pandas as pd
import numpy as np
import scanpy as sc
import dgl
from dgl.sampling import random_walk, pack_traces
import dgl.function as fn
import torch
import torch.nn.functional as F
import collections
from typing import Optional, Union
from pathlib import Path
import time
import math
import os
from anndata import AnnData
import random
from sklearn.model_selection import train_test_split
import scipy

def normalize(adata,counts_per_cell_after=1e4,log_transformed=False):
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=counts_per_cell_after)

def log1p(adata):
    sc.pp.log1p(adata)

class lsiTransformer():
    def __init__(self,
                 n_components: int = 20,
                 drop_first=True,
                ):

        self.drop_first=drop_first
        self.n_components = n_components+drop_first
        self.tfidfTransformer = tfidfTransformer()
        self.normalizer = sklearn.preprocessing.Normalizer(norm="l1")
        self.pcaTransformer = sklearn.decomposition.TruncatedSVD(n_components = self.n_components, random_state=777)
        self.fitted = None

    def fit(self, adata: anndata.AnnData):
        X = self.tfidfTransformer.fit_transform(adata.layers['counts'])
        X_norm = self.normalizer.fit_transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        self.pcaTransformer.fit(X_norm)
        self.fitted = True

    def transform(self, adata):
        if not self.fitted:
            raise RuntimeError('Transformer was not fitted on any data')
        X = self.tfidfTransformer.transform(adata.layers['counts'])
        X_norm = self.normalizer.transform(X)
        X_norm = np.log1p(X_norm * 1e4)
        X_lsi = self.pcaTransformer.transform(X_norm)
        #         X_lsi -= X_lsi.mean(axis=1, keepdims=True)
        #         X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
        lsi_df = pd.DataFrame(X_lsi, index=adata.obs_names).iloc[:, int(self.drop_first):]
        return lsi_df

    def fit_transform(self, adata):
        self.fit(adata)
        return self.transform(adata)


class tfidfTransformer():
    def __init__(self):
        self.idf = None
        self.fitted = False

    def fit(self, X):
        self.idf = X.shape[0] / X.sum(axis=0)
        self.fitted = True

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError('Transformer was not fitted on any data')
        if scipy.sparse.issparse(X):
            tf = X.multiply(1 / X.sum(axis=1))
            return tf.multiply(self.idf)
        else:
            tf = X / X.sum(axis=1, keepdims=True)
            return tf * self.idf

    def fit_transform(self, X):
        self.fit(X)
