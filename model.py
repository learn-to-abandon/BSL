import copy
import random

import numpy as np
import torch
import torch.nn.functional as F
from itertools import chain
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
import torch_geometric.nn as gnn
from tqdm import tqdm
import torch_geometric

from utils import aucPerformance


class Dmodel(nn.Module):
    def __init__(self, in_channels, hid_channels, dropout=0.0):
        super(Dmodel, self).__init__()
        self.relu = nn.ReLU()
        self.leakyRelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        # self.linear_transform = nn.Sequential(nn.Linear(in_channels, hid_channels),
        #                                       self.relu,
        #                                       nn.Linear(hid_channels, hid_channels),
        #                                       )
        self.linear_transform = nn.Sequential(nn.Linear(in_channels, hid_channels),
                                              self.leakyRelu,
                                              nn.Linear(hid_channels, hid_channels),
                                              )
        # SAGE for Amazon, Yelp, Elliptic, GCN for T-Finance, T-Social
        self.encoder = gnn.GCNConv(hid_channels, hid_channels, add_self_loops=True)
        # self.encoder = gnn.SAGEConv(hid_channels, hid_channels)

        self.hid_channels = hid_channels
        self.dim = int(hid_channels / 3)

        # self.GCN = gnn.GCNConv(in_channels, 2)
        # self.GAT = gnn.GAT(in_channels, 2, num_layers=5)
        # self.GraphSage = gnn.SAGEConv(in_channels, 2)

        self.edge_classifier = nn.Sequential(nn.Linear(self.dim, 1),
                                              self.sigmoid,
                                            )

        self.classifier = nn.Linear(int(hid_channels / 3), 2)
        self.W_rb = nn.Sequential(self.tanh,
                                  nn.Linear(self.dim, self.dim),
                                  )
        self.W_rr = nn.Sequential(self.tanh,
                                  nn.Linear(self.dim, self.dim),
                                  )
        self.W_bb = nn.Sequential(self.tanh,
                                  nn.Linear(self.dim, self.dim),
                                  )
        self.q = nn.Linear(self.dim, 1, bias=False)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.augmentation_loss = nn.MSELoss(reduction='mean')
        self.edge_loss = nn.CrossEntropyLoss()

        # self.linear = list(self.linear_transform.parameters())
        # self.linear.extend(list(self.classifier.parameters()))
        # self.linear.extend(list(self.encoder.parameters()))
        # self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, train, batch_size, x, edge_index, y, u_batch_size=None, u_x=None, u_edge_index=None):
        edge_index = edge_index.to(self.device)

        x = self.linear_transform(x)
        x = self.encoder(x, edge_index)
        class_x = self.attention(x[0: batch_size, :self.dim], x[0: batch_size, self.dim: self.dim*2], x[0: batch_size, self.dim*2:])
        y_hat = self.classifier(class_x)

        # GCN
        # y_hat = self.GCN(x, edge_index)[:batch_size, :]
        # GAT
        # y_hat = self.GAT(x, edge_index)[:batch_size, :]
        # GraphSage
        # y_hat = self.GraphSage(x, edge_index)[:batch_size, :]

        if train:
            bias_loss = 0
            br, rr, bb = self.split_feature(x, edge_index)
            edge_pred = torch.cat((br, rr, bb), 1)
            edge_y = self.edge_label(edge_index, y)

            edge_loss = self.edge_loss(edge_pred, edge_y)
            bias_loss += edge_loss

            u_x = self.linear_transform(u_x)
            u_edge_index = u_edge_index.to(self.device)
            u_x = self.encoder(u_x, u_edge_index)[:u_batch_size, :]
            normal_x = x[: batch_size, :][~(y.bool()[: batch_size])]
            abnormal_x = x[: batch_size, :][y.bool()[: batch_size]]
            bias_loss += self.attention_w(normal_x, abnormal_x) * 0.6

            normal_random = torch.randint(0, normal_x.shape[0], (u_batch_size,))
            normal_random = normal_x[normal_random]
            abnormal_random = torch.randint(0, abnormal_x.shape[0], (u_batch_size,))
            abnormal_random = abnormal_x[abnormal_random]

            WN = torch.cat((normal_random[:, :self.dim], u_x[:u_batch_size, self.dim:]), dim=1)
            WN = self.attention(WN[:, :self.dim], WN[:, self.dim: self.dim * 2], WN[:, self.dim * 2:])
            WA = torch.cat((abnormal_random[:, :self.dim], u_x[:u_batch_size, self.dim:]), dim=1)
            WA = self.attention(WA[:, :self.dim], WA[:, self.dim: self.dim * 2], WA[:, self.dim * 2:])
            SN = torch.cat((u_x[:u_batch_size, :self.dim], normal_random[:, self.dim:]), dim=1)
            SN = self.attention(SN[:, :self.dim], SN[:, self.dim: self.dim * 2], SN[:, self.dim * 2:])
            SA = torch.cat((u_x[:u_batch_size, :self.dim], abnormal_random[:, self.dim:]), dim=1)
            SA = self.attention(SA[:, :self.dim], SA[:, self.dim: self.dim * 2], SA[:, self.dim * 2:])

            a = self.classifier(WN).softmax(-1)[:, 1]
            b = self.classifier(WA).softmax(-1)[:, 1]
            c = self.classifier(SN).softmax(-1)[:, 1]
            d = self.classifier(SA).softmax(-1)[:, 1]
            weak_loss = self.augmentation_loss(a, b)
            strong_loss = self.augmentation_loss(c, d)
            augmentation_loss = weak_loss - strong_loss
            # print(augmentation_loss)

            bias_loss += augmentation_loss * 0.6

            return y_hat, augmentation_loss

        else:
            return y_hat, x[0: batch_size, :self.dim], x[0: batch_size, self.dim:]

    def attention_w(self, normal_x, abnormal_x):
        w_br = self.q(self.W_rb(normal_x[:, :self.dim]))
        w_rr = self.q(self.W_rr(normal_x[:, self.dim:self.dim*2]))
        w_bb = self.q(self.W_bb(normal_x[:, self.dim*2:]))
        W = torch.cat((w_br, w_rr, w_bb), 1)
        W = F.softmax(W, dim=1)
        l = torch.mean(W[:, 1]) - torch.mean(W[:, 2])
        # print(l)
        # print(torch.mean(W[:, 0]).data, torch.mean(W[:, 1]).data, torch.mean(W[:, 2]).data)
        w_br = self.q(self.W_rb(abnormal_x[:, :self.dim]))
        w_rr = self.q(self.W_rr(abnormal_x[:, self.dim:self.dim*2]))
        w_bb = self.q(self.W_bb(abnormal_x[:, self.dim*2:]))
        W = torch.cat((w_br, w_rr, w_bb), 1)
        W = F.softmax(W, dim=1)
        l = l - torch.mean(W[:, 1]) + torch.mean(W[:, 2])
        # print(- torch.mean(W[:, 1]) + torch.mean(W[:, 2]))
        # print(torch.mean(W[:, 0]).data, torch.mean(W[:, 1]).data, torch.mean(W[:, 2]).data)

        return l

    def attention(self, br, rr, bb):
        w_br = self.q(self.W_rb(br))
        w_rr = self.q(self.W_rr(rr))
        w_bb = self.q(self.W_bb(bb))
        W = torch.cat((w_br, w_rr, w_bb), 1)
        W = F.softmax(W, dim=1)
        # print(br.shape, W[:, 0].shape)
        x_br = torch.bmm(W[:, 0].unsqueeze(-1).unsqueeze(-1), br.unsqueeze(1))
        x_rr = torch.bmm(W[:, 1].unsqueeze(-1).unsqueeze(-1), rr.unsqueeze(1))
        x_bb = torch.bmm(W[:, 2].unsqueeze(-1).unsqueeze(-1), bb.unsqueeze(1))
        return (x_br + x_rr + x_bb).squeeze(1)

    def split_feature(self, x, edge_index):
        s = int(self.hid_channels / 3)
        br = x[:, :s]
        rr = x[:, s: s * 2]
        bb = x[:, s * 2:]

        br_edge = br[edge_index[0, :], :] - br[edge_index[1, :], :]
        rr_edge = rr[edge_index[0, :], :] - rr[edge_index[1, :], :]
        bb_edge = bb[edge_index[0, :], :] - bb[edge_index[1, :], :]

        br_edge = self.edge_classifier(br_edge)
        rr_edge = self.edge_classifier(rr_edge)
        bb_edge = self.edge_classifier(bb_edge)

        return br_edge, rr_edge, bb_edge

    def edge_label(self, edge_index, labels):
        edge_start = labels[edge_index[0, :]]
        edge_end = labels[edge_index[1, :]]
        br = edge_start ^ edge_end
        rr = edge_start & edge_end
        bb = (~edge_start + 2) & (~edge_end + 2)
        edge_y = torch.argmax(torch.stack((br, rr, bb), 1), -1)

        return edge_y

    def deal_inf(self, g):
        g = torch.where(g == 0, torch.full_like(g, 1), g)
        g = torch.where(torch.isinf(-g), torch.full_like(g, 0), g)
        return g

    @torch.no_grad()
    def evaluating(self, batch_size, x, edge_index, labels):
        self.eval()
        y_hat = self.forward(False, batch_size, x, edge_index, labels)
        self.train()
        return y_hat
