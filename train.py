import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch import random
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
from model import Dmodel
from copy import deepcopy
from config import *
import pickle
import numpy as np
from data_split import loadAmazon, loadElliptic, loadTFinance, loadYelp, loadTSocial
from torch_geometric.loader.neighbor_sampler import NeighborSampler
import random as rd

import warnings

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def delete_edges(edge_index, labels, ratio=0.5):
    edge_label = labels[edge_index[0]] ^ labels[edge_index[1]]
    homo = edge_index.T[~edge_label.bool()]
    heter = edge_index.T[edge_label.bool()]
    print(heter.shape[0] / edge_index.shape[1])
    heter_random_index = torch.LongTensor(
        rd.sample(range(heter.shape[0]), int((1 - ratio) * heter.shape[0])))
    heter_random_selected = torch.index_select(heter, 0, heter_random_index).T
    filtered = torch.cat((heter_random_selected, homo.T), dim=1)
    return filtered


def train(batch_size, u_batch_size, model, criterion, optimizer, x, edge_index, y, u_x, u_edge_index):
    model.train()
    optimizer.zero_grad()

    y_hat, bias_loss = model.forward(True, batch_size, x, edge_index, y, u_batch_size, u_x, u_edge_index)
    # weak_y = model(batch_size, x, edge_index, unlabeled=False, weak_aug=True, labels=y)
    # u_weak_y = model(u_batch_size, u_x, u_edge_index, unlabeled=True, weak_aug=True)
    # u_strong_y = model(u_batch_size, u_x, u_edge_index, unlabeled=True, weak_aug=False)
    #
    # max_probs = torch.max(u_weak_y, dim=1)
    # pseudo_label = max_probs.indices.detach()
    # max_probs = max_probs.values
    #
    # mask = torch.greater_equal(
    #     max_probs,
    #     torch.ones_like(max_probs) * 0.99)

    CE_loss = criterion(y_hat, y[: batch_size])
    # sup_loss = criterion(weak_y, y_hat)
    # unsup_loss = (F.cross_entropy(u_strong_y, pseudo_label, reduction='none') * mask).mean()
    loss_train = CE_loss + bias_loss

    loss_train.backward()
    optimizer.step()
    return loss_train.detach().item()


def test(batch_size, model, x, edge_index, y):
    y_hat, br, rrbb = model.evaluating(batch_size, x, edge_index, y)

    y_test = y[:batch_size].cpu().numpy()
    y_pred = y_hat.cpu().numpy()
    br = br.cpu().numpy()
    rrbb = rrbb.cpu().numpy()
    return br, rrbb, y_test, y_pred


def main(params_config, dataset):
    batch_size = 512

    mu = 10

    data = None
    if dataset == 'Amazon':
        data = loadAmazon()
    elif dataset == 'Yelp':
        data = loadYelp()
    elif dataset == 'Elliptic':
        data = loadElliptic()
    elif dataset == 'TFinance':
        data = loadTFinance()
    elif dataset == 'TSocial':
        data = loadTSocial()
    else:
        print('No such dataset.')

    # data.edge_index = delete_edges(data.edge_index, data.y, 1)

    data = data.to(device)
    # print(data.is_cuda)
    net = Dmodel(in_channels=data.x.shape[1], hid_channels=params_config['hidden_channels'])
    net.to(device)

    # optimizer = optim.Adam([dict(params=net.linear, lr=1e-5)])
    optimizer = optim.Adam(net.parameters(), lr=params_config['lr'], weight_decay=params_config['weight_decay'])

    # weight = (1 - data.y[data.train_mask]).sum().item() / data.y[data.train_mask].sum().item()
    # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., weight]).to(device))
    criterion = torch.nn.CrossEntropyLoss()

    trainLoader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                                  sizes=[-1], batch_size=batch_size, shuffle=True)
    unlabeledLoader = NeighborSampler(data.edge_index, node_idx=data.val_mask,
                                      sizes=[-1], batch_size=batch_size * mu, shuffle=True)
    testLoader = NeighborSampler(data.edge_index, node_idx=data.test_mask,
                                 sizes=[-1], batch_size=batch_size, shuffle=True)

    for epoch in range(params_config['epochs']):
        loss = []
        unlabeled_iter = iter(unlabeledLoader)

        for idx, (batch_size, n_id, adjs) in enumerate(trainLoader):
            x = data.x[n_id]
            y = data.y[n_id]
            edge_index = adjs.edge_index

            try:
                u_batch_size, u_n_id, u_adjs = unlabeled_iter.__next__()
                u_x = data.x[u_n_id]
                u_edge_index = adjs.edge_index
                batch_loss = train(batch_size, u_batch_size, net, criterion, optimizer, x, edge_index, y, u_x,
                                   u_edge_index)
                loss.append(batch_loss)
            except:
                pass

        print('Epoch:{:04d}\tloss:{:.4f}'.format(epoch + 1, np.mean(loss)))

        # showEmbedding(br_t, rrbb_t, y_test)

        if epoch >= 20:
            y_test = None
            y_pred = None
            br_t = None
            rrbb_t = None
            precision = []
            recall = []
            F1 = []
            for batch_size, n_id, adjs in tqdm(testLoader):
                x = data.x[n_id]
                y = data.y[n_id]
                edge_index = adjs.edge_index
                br, rrbb, y, y_hat = test(batch_size, net, x, edge_index, y)
                # y_pred = np.argmax(y_hat, axis=1)
                # precision.append(metrics.precision_score(y, y_pred, average='macro'))
                # recall.append(metrics.recall_score(y, y_pred, average='macro'))
                # F1.append(metrics.f1_score(y, y_pred, average='macro'))

                if y_test is None:
                    y_test = y
                    y_pred = y_hat
                    br_t = br
                    rrbb_t = rrbb

                else:
                    y_test = np.append(y_test, y)
                    y_pred = np.concatenate((y_pred, y_hat), axis=0)
                    br_t = np.concatenate((br_t, br), axis=0)
                    rrbb_t = np.concatenate((rrbb_t, rrbb), axis=0)

            y_pred = np.argmax(y_pred, axis=1)
            report = metrics.classification_report(y_test, y_pred, digits=4)
            print(report)

            # # for large dataset
            # print(np.mean(precision), np.mean(recall), np.mean(F1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default='TFinance', help='Dataset [TSocial, TFinance, Elliptic, Amazon, Yelp]')

    random_seed = 2022
    torch.manual_seed(random_seed)

    args = parser.parse_args()

    params_config = dataset_config[args.dataset]
    auc_roc_list = []
    auc_pr_list = []
    main(params_config, args.dataset)
