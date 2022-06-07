from dance.utils import set_seed
from dance.datasets.multimodality import JointEmbeddingNIPSDataset
from dance.modules.multi_modality.joint_embedding.scmogcn import ScMoGCNWrapper
import anndata as ad
import argparse
import random
import torch
import scanpy as sc
import numpy as np
import dgl
from dance.transforms.graph_construct import construct_basic_feature_graph, basic_feature_graph_propagation
import dance.utils.metrics as metrics

def scmogcn_test(adata):
    sc._settings.ScanpyConfig.n_jobs = 4
    adata_sol = dataset.test_sol
    adata.obs['batch'] = adata_sol.obs['batch'][adata.obs_names]
    adata.obs['cell_type'] = adata_sol.obs['cell_type'][adata.obs_names]
    print(adata.shape,adata_sol.shape)
    adata_bc = adata.obs_names
    adata_sol_bc = adata_sol.obs_names
    select = [item in adata_bc for item in adata_sol_bc]
    adata_sol = adata_sol[select, :]
    print(adata.shape, adata_sol.shape)

    adata.obsm['X_emb'] = adata.X
    nmi = metrics.get_nmi(adata)
    cell_type_asw = metrics.get_cell_type_ASW(adata)
    cc_con = metrics.get_cell_cycle_conservation(adata, adata_sol)
    traj_con = metrics.get_traj_conservation(adata, adata_sol)
    batch_asw = metrics.get_batch_ASW(adata)
    graph_score = metrics.get_graph_connectivity(adata)

    print('nmi %.4f, celltype_asw %.4f, cc_con %.4f, traj_con %.4f, batch_asw %.4f, graph_score %.4f\n' % (
    nmi, cell_type_asw, cc_con, traj_con, batch_asw, graph_score))

    print('average metric: %.4f' % np.mean(
        [round(i, 4) for i in [nmi, cell_type_asw, cc_con, traj_con, batch_asw, graph_score]]))

if __name__ == '__main__':
    rndseed = random.randint(0, 2147483647)
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--subtask', default = 'openproblems_bmmc_cite_phase2', choices = ['openproblems_bmmc_cite_phase2', 'openproblems_bmmc_multiome_phase2'])
    parser.add_argument('-d', '--data_folder', default = './data/joint_embedding')
    parser.add_argument('-pre', '--pretrained_folder', default = './data/joint_embedding/pretrained')
    parser.add_argument('-csv', '--csv_path', default = 'decoupled_lsi.csv')
    parser.add_argument('-l', '--layers', default = 3, type=int, choices = [3,4,5,6,7])
    parser.add_argument('-dis', '--disable_propagation', default = 0, type=int, choices = [0, 1, 2])
    parser.add_argument('-seed', '--rnd_seed', default = rndseed, type=int)
    parser.add_argument('-cpu', '--cpus', default = 1, type = int)
    parser.add_argument('-device', '--device', default = 'cuda:2')
    parser.add_argument('-bs', '--batch_size', default = 512, type=int)
    parser.add_argument('-nm', '--normalize', default = 1, type=int, choices = [0, 1])

    args = parser.parse_args()

    device = args.device
    pre_normalize = bool(args.normalize)
    torch.set_num_threads(args.cpus)
    rndseed = args.rnd_seed
    set_seed(rndseed)

    dataset = JointEmbeddingNIPSDataset(args.subtask, data_dir=args.data_folder).load_data()\
        .load_metadata().load_sol().preprocess(args.pretrained_folder).normalize()
    X_train, Y_train, X_test = dataset.preprocessed_data['X_train'], dataset.preprocessed_data['Y_train'], \
                                       dataset.preprocessed_data['X_test']

    g = construct_basic_feature_graph(X_train, X_test, device=device)
    X = basic_feature_graph_propagation(g, layers=args.layers, device=device)

    l = args.layers - 1

    model = ScMoGCNWrapper(args, dataset)
    model.fit(dataset, X, Y_train)
    model.load(f'models/model_joint_embedding_{rndseed}.pth')

    with torch.no_grad():
        embeds = model.predict(X, np.arange(X[0].shape[0])).cpu().numpy()
        print(embeds)

    mod1_obs = dataset.modalities[0].obs
    mod1_uns = dataset.modalities[0].uns
    adata = ad.AnnData(
        X=embeds,
        obs=mod1_obs,
        uns={
            'dataset_id': mod1_uns['dataset_id'],
            'method_id': 'scmogcn',
        },
    )
    adata.write_h5ad(f'./joint_embedding_{rndseed}.h5ad', compression="gzip")

    # old Neurips metrics
    # scmogcn_test(adata)
    # scmvae test
    metrics.dcca_evaluate(adata, dataset)