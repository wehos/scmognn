# Copyright 2022 DSE lab.  All rights reserved.

import time
import dgl
from dgl import nn as dglnn
from scipy.sparse import csc_matrix
import torch
import os
from collections import defaultdict
import pickle
from sklearn.decomposition import TruncatedSVD, PCA
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.preprocessing import normalize

import random
from torch.nn import functional as F
import preprocess


def csr_cosine_similarity(input_csr_matrix):
    similarity = input_csr_matrix * input_csr_matrix.T
    square_mag = similarity.diagonal()
    inv_square_mag = 1 / square_mag
    inv_square_mag[np.isinf(inv_square_mag)] = 0
    inv_mag = np.sqrt(inv_square_mag)
    res = similarity.multiply(inv_mag).T.multiply(inv_mag)
    return res.toarray()


def cosine_similarity_gene(input_matrix):
    from sklearn.metrics.pairwise import cosine_similarity
    res = cosine_similarity(input_matrix)
    res = np.abs(res)
    return res


def construct_graph(x, y, x_pixel=None, y_pixel=None, image=None, beta=49, alpha=1, histology=True):
    return calculate_adj_matrix(x, y, x_pixel=x_pixel, y_pixel=y_pixel, image=image, beta=beta, alpha=alpha,
                                histology=histology)


def construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph=None, test=False, **kwargs):
    """Generate a feature-cell graph, enhanced with domain-knowledge (e.g. pathway).

    Parameters
    ----------
    u: torch.Tensor
        1-dimensional tensor. Cell node id of each cell-feature edge.
    v: torch.Tensor
        1-dimensional tensor. Feature node id of each cell-feature edge.
    e: torch.Tensor
        1-dimensional tensor. Weight of each cell-feature edge.
    cell_node_features: torch.Tensor
        1-dimensional or 2-dimensional tensor.  Node features for each cell node.
    enhance_graph: list[torch.Tensor]
        Node ids and edge weights of enhancement graph.

    Returns
    --------
    graph: DGLGraph
        The generated graph.
    """

    TRAIN_SIZE = kwargs['TRAIN_SIZE']
    FEATURE_SIZE = kwargs['FEATURE_SIZE']

    if enhance_graph is None:
        print('WARNING: Enhance graph disabled.')

    if kwargs['only_pathway'] and enhance_graph is not None:
        assert (kwargs['subtask'].find('rna') != -1)
        uu, vv, ee = enhance_graph

        graph_data = {
            ('feature', 'feature2cell', 'cell'): (v, u),
            ('feature', 'pathway', 'feature'): (uu, vv),
        }
        graph = dgl.heterograph(graph_data)

        if kwargs['inductive'] != 'trans':
            graph.nodes['cell'].data['id'] = cell_node_features[:TRAIN_SIZE] if not test else cell_node_features
        else:
            graph.nodes['cell'].data['id'] = cell_node_features

        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['feature2cell'].data['weight'] = e
        graph.edges['pathway'].data['weight'] = torch.tensor(ee).float()

    elif kwargs['no_pathway'] or kwargs['subtask'].find('rna') == -1 or enhance_graph is None:


        if kwargs['inductive'] == 'opt':
            print('Not supported.')
            # graph_data = {
            #     ('cell', 'cell2feature', 'feature'): (u, v) if not test else (
            #         u[:g.edges(etype='cell2feature')[0].shape[0]], v[:g.edges(etype='cell2feature')[0].shape[0]]),
            #     ('feature', 'feature2cell', 'cell'): (v, u),
            # }

        else:
            graph_data = {
                ('cell', 'cell2feature', 'feature'): (u, v),
                ('feature', 'feature2cell', 'cell'): (v, u),
            }

        graph = dgl.heterograph(graph_data)

        if kwargs['inductive'] != 'trans':
            graph.nodes['cell'].data['id'] = cell_node_features[:TRAIN_SIZE] if not test else cell_node_features
        else:
            graph.nodes['cell'].data['id'] = cell_node_features
        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['feature2cell'].data['weight'] = e
        graph.edges['cell2feature'].data['weight'] = e[:graph.edges(etype='cell2feature')[0].shape[0]]

    else:
        assert (kwargs['subtask'].find('rna') != -1)
        uu, vv, ee = enhance_graph

        if kwargs['inductive'] == 'opt':
            print("Not supported.")
            # graph_data = {
            #     ('cell', 'cell2feature', 'feature'): (u, v) if not test else (
            #         u[:g.edges(etype='cell2feature')[0].shape[0]], v[:g.edges(etype='cell2feature')[0].shape[0]]),
            #     ('feature', 'feature2cell', 'cell'): (v, u),
            #     ('feature', 'pathway', 'feature'): (uu, vv),
            # }
        else:
            graph_data = {
                ('cell', 'cell2feature', 'feature'): (u, v),
                ('feature', 'feature2cell', 'cell'): (v, u),
                ('feature', 'pathway', 'feature'): (uu, vv),
            }
        graph = dgl.heterograph(graph_data)

        if kwargs['inductive'] != 'trans':
            graph.nodes['cell'].data['id'] = cell_node_features[:TRAIN_SIZE] if not test else cell_node_features
        else:
            graph.nodes['cell'].data['id'] = cell_node_features
        graph.nodes['feature'].data['id'] = torch.arange(FEATURE_SIZE).long()
        graph.edges['feature2cell'].data['weight'] = e
        graph.edges['cell2feature'].data['weight'] = e[:graph.edges(etype='cell2feature')[0].shape[0]]
        graph.edges['pathway'].data['weight'] = torch.tensor(ee).float()

    return graph


# TODO: haven't explained extra kwargs
def construct_pathway_graph(gex_data, **kwargs):
    """Generate nodes, edges and edge weights for pathway dataset.

    Parameters
    ----------
    gex_data: anndata.AnnData
        Gene data, contains feature matrix (.X) and feature names (.var['feature_types']).

    Returns
    --------
    uu: list[int]
        Predecessor node id of each edge.
    vv: list[int]
        Successor node id of each edge.
    ee: list[float]
        Edge weight of each edge.
    """

    pww = kwargs['pathway_weight']
    npw = kwargs['no_pathway']
    subtask = kwargs['subtask']
    pw_path = kwargs['pathway_path']
    uu = []
    vv = []
    ee = []

    assert (not npw)

    pk_path = f'pw_{subtask}_{pww}.pkl'
    #     pk_path = f'pw_{subtask}_{pww}.pkl'
    if os.path.exists(pk_path):
        print(
            'WARNING: Pathway file exist. Load pickle file by default. Auguments "--pathway_weight" and "--pathway_path" will not take effect.')
        uu, vv, ee = pickle.load(open(pk_path, 'rb'))
    else:
        # Load Original Pathway File
        with open(pw_path + '.entrez.gmt') as gmt:
            gene_list = gmt.read().split()

        gene_sets_entrez = defaultdict(list)
        indicator = 0
        for ele in gene_list:
            if not ele.isnumeric() and indicator == 1:
                indicator = 0
                continue
            if not ele.isnumeric() and indicator == 0:
                indicator = 1
                gene_set_name = ele
            else:
                gene_sets_entrez[gene_set_name].append(ele)

        with open(pw_path + '.symbols.gmt') as gmt:
            gene_list = gmt.read().split()

        gene_sets_symbols = defaultdict(list)

        for ele in gene_list:
            if ele in gene_sets_entrez:
                gene_set_name = ele
            elif not ele.startswith('http://'):
                gene_sets_symbols[gene_set_name].append(ele)

        pw = [i[1] for i in gene_sets_symbols.items()]

        # Generate New Pathway Data
        counter = 0
        total = 0
        feature_index = gex_data.var['feature_types'].index.tolist()
        gex_features = gex_data.X
        new_pw = []
        for i in pw:
            new_pw.append([])
            for j in i:
                if j in feature_index:
                    new_pw[-1].append(feature_index.index(j))

        if pww == 'cos':
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            sj = np.sqrt(
                                np.dot(gex_features[:, j].toarray().T, gex_features[:, j].toarray()).item())
                            sk = np.sqrt(
                                np.dot(gex_features[:, k].toarray().T, gex_features[:, k].toarray()).item())
                            jk = np.dot(gex_features[:, j].toarray().T, gex_features[:, k].toarray())
                            cossim = jk / sj / sk
                            ee.append(cossim.item())
        elif pww == 'one':
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            ee.append(1.)
        elif pww == 'pearson':
            corr = np.corrcoef(gex_features.toarray().T)
            for i in new_pw:
                for j in i:
                    for k in i:
                        if j != k:
                            uu.append(j)
                            vv.append(k)
                            ee.append(corr[j][k])

        pickle.dump([uu, vv, ee], open(pk_path, 'wb'))

    # Apply Threshold
    pwth = kwargs['pathway_threshold']
    nu = []
    nv = []
    ne = []

    for i in range(len(uu)):
        if abs(ee[i]) > pwth:
            ne.append(ee[i])
            nu.append(uu[i])
            nv.append(vv[i])
    uu, vv, ee = nu, nv, ne

    return uu, vv, ee


def construct_basic_feature_graph(feature_mod1, feature_mod1_test=None, bf_input=None, device='cuda'):
    input_train_mod1 = csc_matrix(feature_mod1)

    if feature_mod1_test is not None:
        input_test_mod1 = csc_matrix(feature_mod1_test)
        assert (input_test_mod1.shape[1] == input_train_mod1.shape[1])

        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)] + \
                                            [np.array(t.nonzero()[0] + i + input_train_mod1.shape[0]) for i, t in
                                             enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] + \
                                            [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        sample_size = input_train_mod1.shape[0] + input_test_mod1.shape[0]
        weights = torch.from_numpy(
            np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()

    else:
        u = torch.from_numpy(
            np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1], axis=0))
        sample_size = input_train_mod1.shape[0]
        weights = torch.from_numpy(np.concatenate([input_train_mod1.tocsr().data], axis=0)).float()

    graph_data = {
        ('cell', 'cell2feature', 'feature'): (u, v),
        ('feature', 'feature2cell', 'cell'): (v, u),
    }
    g = dgl.heterograph(graph_data)

    if bf_input:
        g.nodes['cell'].data['bf'] = gen_batch_features(bf_input)
    else:
        g.nodes['cell'].data['bf'] = torch.zeros(sample_size).float()

    g.nodes['cell'].data['id'] = torch.zeros(sample_size).long() #torch.arange(sample_size).long()
    #     g.nodes['cell'].data['source'] =
    g.nodes['feature'].data['id'] = torch.arange(input_train_mod1.shape[1]).long()
    g.edges['cell2feature'].data['weight'] = g.edges['feature2cell'].data['weight'] = weights

    g = g.to(device)
    return g


def gen_batch_features(ad_inputs):
    """Generate statistical features for each batch in the input data, and assign batch features to each cell.
    This function returns batch features for each cell in all the input sub-datasets.

    Parameters
    ----------
    ad_inputs: list[anndata.AnnData]
        A list of AnnData object, each contains a sub-dataset.

    Returns
    --------
    batch_features: torch.Tensor
        A batch_features matrix, each row refers to one cell from the datasets. The matrix can be directly used as the
        node features of cell nodes.
    """

    cells = []
    columns = ['cell_mean', 'cell_std', 'nonzero_25%', 'nonzero_50%', 'nonzero_75%', 'nonzero_max', 'nonzero_count',
               'nonzero_mean', 'nonzero_std', 'batch']

    assert len(ad_inputs) < 10, "WARNING: Input of gen_bf_features should be a list of AnnData objects."

    for ad_input in ad_inputs:
        bcl = list(ad_input.obs['batch'])
        print(set(bcl))
        for i, cell in enumerate(ad_input.X):
            cell = cell.toarray()
            nz = cell[np.nonzero(cell)]
            cells.append([cell.mean(), cell.std(), np.percentile(nz, 25), np.percentile(nz, 50), np.percentile(nz, 75),
                          cell.max(), len(nz) / 1000, nz.mean(), nz.std(), bcl[i]])

    cell_features = pd.DataFrame(cells, columns=columns)
    batch_source = cell_features.groupby('batch').mean().reset_index()
    batch_list = batch_source.batch.tolist()
    batch_source = batch_source.drop('batch', axis=1).to_numpy().tolist()
    b2i = dict(zip(batch_list, range(len(batch_list))))
    batch_features = []

    for ad_input in ad_inputs:
        for b in ad_input.obs['batch']:
            batch_features.append(batch_source[b2i[b]])

    batch_features = torch.tensor(batch_features).float()

    return batch_features


def construct_modality_prediction_graph(dataset, **kwargs):
    """Construct the cell-feature graph object for modality prediction task, based on the input dataset.

    Parameters
    ----------
    dataset: datasets.multimodality.ModalityPredictionDataset
        The input dataset, typically includes four input AnnData sub-datasets, which are train_mod1, train_mod2,
        test_mod1 and test_mod2 respectively.

    Returns
    --------
    g: DGLGraph
        The generated graph.
    """

    train_mod1 = dataset.modalities[0]
    input_train_mod1 = dataset.sparse_features()[0]
    if kwargs['inductive'] == 'trans':
        input_test_mod1 = dataset.sparse_features()[2]

    CELL_SIZE = kwargs['CELL_SIZE']
    TRAIN_SIZE = kwargs['TRAIN_SIZE']

    if kwargs['cell_init'] == 'none':
        cell_node_features = torch.ones(CELL_SIZE).long()
    elif kwargs['cell_init'] == 'pca':
        embedder_mod1 = TruncatedSVD(n_components=100)
        X_train_np = embedder_mod1.fit_transform(input_train_mod1.toarray())
        X_test_np = embedder_mod1.transform(input_test_mod1.toarray())
        cell_node_features = torch.cat([torch.from_numpy(X_train_np), torch.from_numpy(X_test_np)], 0).float()
    if (not kwargs['no_pathway']) and (kwargs['subtask'].find('rna') != -1):
        enhance_graph = construct_pathway_graph(train_mod1, **kwargs)
    else:
        enhance_graph = None

    if kwargs['inductive'] != 'trans':
        u = torch.from_numpy(
            np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1], axis=0))
        e = torch.from_numpy(input_train_mod1.tocsr().data).float()
        g = construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph, **kwargs)

        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)] + \
                                            [np.array(t.nonzero()[0] + i + TRAIN_SIZE) for i, t in
                                             enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] + \
                                            [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        e = torch.from_numpy(
            np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()
        gtest = construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph, test=True, **kwargs)
        return g, gtest

    else:
        u = torch.from_numpy(np.concatenate([np.array(t.nonzero()[0] + i) for i, t in enumerate(input_train_mod1)] + \
                                            [np.array(t.nonzero()[0] + i + TRAIN_SIZE) for i, t in
                                             enumerate(input_test_mod1)], axis=0))
        v = torch.from_numpy(np.concatenate([np.array(t.nonzero()[1]) for t in input_train_mod1] + \
                                            [np.array(t.nonzero()[1]) for t in input_test_mod1], axis=0))
        e = torch.from_numpy(
            np.concatenate([input_train_mod1.tocsr().data, input_test_mod1.tocsr().data], axis=0)).float()
        g = construct_enhanced_feature_graph(u, v, e, cell_node_features, enhance_graph, **kwargs)

        return g

