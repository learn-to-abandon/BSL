import networkx as nx
import scipy.sparse as sp
import scipy.io as scio
import torch
from dgl import load_graphs
from matplotlib import pyplot as plt
import numpy as np
import random
from torch_geometric.data import Data
from tqdm import tqdm
import dgl
import pickle
from dgl.data import FraudYelpDataset, FraudAmazonDataset


def norm(l):
    for j in range(l.shape[1]):
        min_col = torch.min(l[:, j])
        max_col = torch.max(l[:, j])
        l[:, j] = (l[:, j] - min_col) / (max_col - min_col)
    return l


def split_dataset(data, train_size=0.01, val_size=0.1, seed_num=72):
    random.seed(seed_num)
    train_mask = torch.zeros(data.y.shape).bool()
    val_mask = torch.zeros(data.y.shape).bool()
    test_mask = torch.zeros(data.y.shape).bool()
    train_mask_anm = torch.zeros(data.y.shape).bool()
    train_mask_norm = torch.zeros(data.y.shape).bool()

    anm_list = (data.y).nonzero(as_tuple=True)[0]
    norm_list = (data.y == 0).nonzero(as_tuple=True)[0]

    anm_id_list = torch.Tensor.tolist(anm_list)
    norm_id_list = torch.Tensor.tolist(norm_list)

    num_anm = len(anm_id_list)
    num_norm = int(len(norm_id_list) * 1)
    norm_id_list = norm_id_list[:num_norm]

    total = data.y.shape[0]

    train_anm_id = random.sample(anm_id_list, int(num_anm * train_size))
    train_norm_id = random.sample(norm_id_list, int(num_norm * train_size))

    anm_id_list = list(set(anm_id_list) - set(train_anm_id))
    norm_id_list = list(set(norm_id_list) - set(train_norm_id))

    val_anm_id = random.sample(anm_id_list, int(num_anm * val_size))
    val_norm_id = random.sample(norm_id_list, int(num_norm * val_size))

    test_anm_id = list(set(anm_id_list) - set(val_anm_id))
    test_norm_id = list(set(norm_id_list) - set(val_norm_id))

    test_anm_id = test_anm_id[:int(len(test_anm_id))]
    test_norm_id = test_norm_id[:int(len(test_norm_id))]

    train_mask[train_anm_id] = True
    train_mask[train_norm_id] = True

    val_mask[val_anm_id] = True
    val_mask[val_norm_id] = True

    test_mask[test_anm_id] = True
    test_mask[test_norm_id] = True

    return train_mask, val_mask, test_mask


def loadElliptic():
    data = pickle.load(open('/home/cloud/LZY/2/compare/antifraud-main/data/{}.dat'.format('elliptic'), 'rb'))
    train_mask, val_mask, test_mask = split_dataset(data)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def loadAmazon():
    dataset = FraudAmazonDataset()
    graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
    graph = dgl.add_self_loop(graph)
    label = torch.tensor(graph.ndata['label'], dtype=torch.int64)

    feature = graph.ndata['feature'].numpy()
    # feature = (feature - np.average(feature, 0)) / np.std(feature, 0)
    feature = torch.tensor(feature, dtype=torch.float32)

    adj = torch.tensor(graph.adj().coalesce().indices(), dtype=torch.int64)

    data = Data(x=feature, y=label, edge_index=adj)
    train_mask, val_mask, test_mask = split_dataset(data)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.train_mask[0:3305] = False
    data.test_mask[0:3305] = False

    return data


def loadYelp():
    dataset = FraudYelpDataset()
    graph = dgl.to_homogeneous(dataset[0], ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
    graph = dgl.add_self_loop(graph)
    label = torch.tensor(graph.ndata['label'], dtype=torch.int64)

    feature = graph.ndata['feature'].numpy()
    feature = (feature - np.average(feature, 0)) / np.std(feature, 0)
    feature = torch.tensor(feature, dtype=torch.float32)

    adj = torch.tensor(graph.adj().coalesce().indices(), dtype=torch.int64)

    data = Data(x=feature, y=label, edge_index=adj)
    train_mask, val_mask, test_mask = split_dataset(data)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def loadTFinance():
    print('deal with TFinance')
    graph, _ = load_graphs('/home/cloud/LZY/2/compare/antifraud-main/data/tfinance')
    # graph = dgl.node_subgraph(graph[0], range(0, 10000))
    graph = graph[0]
    label = torch.tensor(graph.ndata['label'].argmax(1), dtype=torch.int64)

    feature = graph.ndata['feature'].numpy()
    feature = (feature-np.average(feature, 0)) / np.std(feature, 0)
    feature = torch.tensor(feature, dtype=torch.float32)

    adj = torch.tensor(graph.adj().coalesce().indices(), dtype=torch.int64)

    data = Data(x=feature, y=label, edge_index=adj)
    train_mask, val_mask, test_mask = split_dataset(data)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

def loadTSocial():
    print('deal with TSocial')
    graph, _ = load_graphs('/home/cloud/LZY/2/compare/antifraud-main/data/tsocial')
    # graph = dgl.node_subgraph(graph[0], range(3000000, 5781065))
    graph = graph[0]
    label = torch.tensor(graph.ndata['label'], dtype=torch.int64)

    feature = graph.ndata['feature'].numpy()
    feature = (feature-np.average(feature, 0)) / np.std(feature, 0)
    feature = torch.tensor(feature, dtype=torch.float32)

    adj = torch.tensor(graph.adj().coalesce().indices(), dtype=torch.int64)

    data = Data(x=feature, y=label, edge_index=adj)
    train_mask, val_mask, test_mask = split_dataset(data)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


if __name__ == '__main__':
    a = loadAmazon()
    e = loadElliptic()
    t = loadTFinance()
    print('load over')
