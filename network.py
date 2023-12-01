import torch.nn as nn
from torch.nn.functional import normalize
from torch.nn import Linear
import torch
import numpy as np


def get_kNNgraph2(data, K_num):
    # each row of data is a sample
    x_norm = np.reshape(torch.sum(torch.square(data), 1), [-1, 1])  # column vector
    x_norm2 = np.reshape(torch.sum(torch.square(data), 1), [1, -1])  # column vector
    dists = x_norm - 2 * torch.matmul(data, torch.transpose(data, 0, 1)) + x_norm2
    num_sample = data.shape[0]
    graph = np.zeros((num_sample, num_sample), dtype=np.int_)
    for i in range(num_sample):
        distance = dists[i, :]
        small_index = np.argsort(distance)
        graph[i, small_index[0:K_num]] = 1
    graph = graph - np.diag(np.diag(graph))
    resultgraph = np.maximum(graph, np.transpose(graph))
    return resultgraph


def comp(g):
    g = g + np.identity(g.shape[0])
    g = torch.tensor(g)
    d = np.diag(g.sum(axis=1))
    d = torch.tensor(d)
    s = pow(d, -0.5)
    where_are_inf = np.isinf(s)
    s[where_are_inf] = 0
    s = torch.matmul(torch.matmul(s, g), s).to(torch.float32)
    return s


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.enc1_1 = Linear(input_dim, 1800)
        self.enc1_2 = Linear(1800, feature_dim)
        self.weight = torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(feature_dim, feature_dim)))

    def forward(self, x, s):
        output1_1 = torch.tanh(self.enc1_1(torch.matmul(s, x)))
        output1_2 = torch.tanh(self.enc1_2(torch.matmul(s, output1_1)))
        z = output1_2
        a = torch.sigmoid(torch.matmul(torch.matmul(z, self.weight), z.T))
        return z, a


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.dec1_1 = Linear(feature_dim, 1800)
        self.dec1_2 = Linear(1800, input_dim)
        self.weight_1 = torch.nn.init.xavier_uniform_(nn.Parameter(torch.FloatTensor(feature_dim, feature_dim)))

    def forward(self, z, s):
        h1 = torch.tanh(torch.matmul(z, self.weight_1))
        h1_1 = torch.tanh(self.dec1_1(torch.matmul(s, h1)))
        h1_2 = torch.tanh(self.dec1_2(torch.matmul(s, h1_1)))
        x = h1_2
        return x


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view

    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        ar = []
        ars = []
        for v in range(self.view):
            x = xs[v]
            g = get_kNNgraph2(x, 10)
            s = comp(g)
            z, a = self.encoders[v](x, s)
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z, s)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
            ar.append(g)
            ars.append(a)
        return hs, qs, xrs, zs, ar, ars

    def forward_plot(self, xs):
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            g = get_kNNgraph2(x, 10)
            s = comp(g)
            z, a = self.encoders[v](x, s)
            zs.append(z)
            h = self.feature_contrastive_module(z)
            hs.append(h)
        return zs, hs

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            g = get_kNNgraph2(x, 10)
            s = comp(g)
            z, a = self.encoders[v](x, s)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds