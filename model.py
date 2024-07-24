import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'


class GraphConvolution(Module):
    """
    simple GCN layer
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias:
            self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight) - self.bias
        output = F.relu(torch.matmul(adj, output))
        return output


class PowerLayer(nn.Module):
    """
    The power layer: calculates the log-transformed power of the data
    """

    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))

    def forward(self, x):
        return torch.log(self.pooling(x.pow(2)))


class LGGNet(nn.Module):
    @staticmethod
    def temporal_learner(in_chan, out_chan, kernel, pool, pool_step_rate):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1)),
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate * pool))
        )

    def __init__(self, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph
                 ):
        # input_size: EEG frequency x channel x datapoint
        super(LGGNet, self).__init__()
        self.input_size = input_size  # Added
        self.idx = idx_graph
        self.window = [0.5, 0.25, 0.125]
        self.pool = pool
        self.channel = input_size[1]
        self.brain_area = len(self.idx)

        # by setting the convolutional kernel being (1,length) and the strids being 1 we can use conv2d to
        # achieve the 1d convolution operation
        self.Tception1 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[0] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[1] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(input_size[0], num_T,
                                               (1, int(self.window[2] * sampling_rate)),
                                               self.pool, pool_step_rate)
        self.BN_t = nn.BatchNorm2d(num_T)
        self.BN_t_ = nn.BatchNorm2d(num_T)
        self.OneXOneConv = nn.Sequential(
            nn.Conv2d(num_T, num_T, kernel_size=(1, 1), stride=(1, 1)),
            nn.LeakyReLU(),
            nn.AvgPool2d((1, 2)))
        # diag(W) to assign a weight to each local areas
        size = self.get_size_temporal(input_size)
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, size[-1]),
                                                requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32),
                                              requires_grad=True)

        # aggregate function
        self.aggregate = Aggregator(self.idx)

        # trainable adj weight for global network
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        # to be used after local graph embedding
        self.bn = nn.BatchNorm1d(self.brain_area)
        self.bn_ = nn.BatchNorm1d(self.brain_area)
        # learn the global network of networks
        self.GCN = GraphConvolution(size[-1], out_graph)

        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), 5),
            nn.Dropout(p=dropout_rate),
            nn.Linear(5, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def temporal_learning_block(self, x):
        out = self.Tception1(x)
        y = self.Tception2(x)
        out = torch.cat((out, y), dim=-1)
        y = self.Tception3(x)
        out = torch.cat((out, y), dim=-1)
        out = self.BN_t(out)
        out = self.OneXOneConv(out)
        out = self.BN_t_(out)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, (out.size(0), out.size(1), -1))
        return out

    def common_forward(self, x):
        out = self.temporal_learning_block(x)
        out = self.local_filter_fun(out, self.local_filter_weight)
        out = self.aggregate.forward(out)
        adj = self.get_adj(out)
        out = self.bn(out)
        out = self.GCN(out, adj)
        out = self.bn_(out)
        out = out.view(out.size()[0], -1)
        return out

    def forward(self, x):
        out = self.common_forward(x)
        out = self.fc(out)

        out = self.sigmoid(out)  # Added
        return out

    def get_size_temporal(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        return self.temporal_learning_block(data).size()

    def local_filter_fun(self, x, w):
        w = w.unsqueeze(0).repeat(x.size()[0], 1, 1)
        x = F.relu(torch.mul(x, w) - self.local_filter_bias)
        return x

    def get_adj(self, x, self_loop=True):
        # x: b, node, feature
        adj = self.self_similarity(x)  # b, n, n
        num_nodes = adj.shape[-1]
        adj = F.relu(adj * (self.global_adj + self.global_adj.transpose(1, 0)))
        if self_loop:
            adj = adj + torch.eye(num_nodes).to(DEVICE)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum)
        mask[rowsum == 0] = 1
        rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj

    @staticmethod
    def self_similarity(x):
        # x: b, node, feature
        x_ = x.permute(0, 2, 1)
        s = torch.bmm(x, x_)
        return s


class Aggregator:

    def __init__(self, idx_area):
        # chan_in_area: a list of the number of channels within each area
        self.chan_in_area = idx_area
        self.idx = self.get_idx(idx_area)
        self.area = len(idx_area)

    def forward(self, x):
        # x: batch x channel x data
        data = []
        for i, area in enumerate(range(self.area)):
            if i < self.area - 1:
                data.append(self.aggr_fun(x[:, self.idx[i]:self.idx[i + 1], :], dim=1))
            else:
                data.append(self.aggr_fun(x[:, self.idx[i]:, :], dim=1))
        return torch.stack(data, dim=1)

    @staticmethod
    def get_idx(chan_in_area):
        idx = [0] + chan_in_area
        idx_ = [0]
        for i in idx:
            idx_.append(idx_[-1] + i)
        return idx_[1:]

    @staticmethod
    def aggr_fun(x, dim):
        # return torch.max(x, dim=dim).values
        return torch.mean(x, dim=dim)


class RNNLGGNet(LGGNet, nn.Module):
    def __init__(self, LGG_model, hidden_size, num_layers, dropout_rate, phase: int = 2):
        super(RNNLGGNet, self).__init__()
        self.lgg = LGG_model
        self.input_size = LGG_model.input_size
        self.idx = LGG_model.idx
        self.window = LGG_model.window
        self.pool = LGG_model.pool
        self.channel = LGG_model.channel
        self.brain_area = LGG_model.brain_area

        self.Tception1 = LGG_model.Tception1
        self.Tception2 = LGG_model.Tception2
        self.Tception3 = LGG_model.Tception3
        self.BN_t = LGG_model.BN_t
        self.BN_t_ = LGG_model.BN_t_
        self.OneXOneConv = LGG_model.OneXOneConv
        # diag(W) to assign a weight to each local areas
        self.local_filter_weight = LGG_model.local_filter_weight
        self.local_filter_bias = LGG_model.local_filter_bias

        # aggregate function
        self.aggregate = LGG_model.aggregate

        # trainable adj weight for global network
        self.global_adj = LGG_model.global_adj
        # to be used after local graph embedding
        self.bn = LGG_model.bn
        self.bn_ = LGG_model.bn
        # learn the global network of networks
        self.GCN = LGG_model.GCN

        if phase == 2:
            for param in self.Tception1.parameters():
                param.requires_grad = False
            for param in self.Tception2.parameters():
                param.requires_grad = False
            for param in self.Tception3.parameters():
                param.requires_grad = False
            for param in self.BN_t.parameters():
                param.requires_grad = False
            for param in self.OneXOneConv.parameters():
                param.requires_grad = False
            for param in self.BN_t_.parameters():
                param.requires_grad = False
            for param in self.local_filter_weight.parameters():
                param.requires_grad = False
            for param in self.local_filter_bias.parameters():
                param.requires_grad = False
            for param in self.global_adj.parameters():
                param.requires_grad = False
            for param in self.bn.parameters():
                param.requires_grad = False
            for param in self.GCN.parameters():
                param.requires_grad = False
            for param in self.bn_.parameters():
                param.requires_grad = False

        self.rnn = nn.GRU(self.get_size_common_forward(self.input_size), hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, 5),
            nn.Dropout(p=dropout_rate),
            nn.Linear(5, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.common_forward(x)
        out = self.rnn(out.unsqueeze(1))[0]
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out

    def get_size_common_forward(self, input_size):
        # input_size: frequency x channel x data point
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])))
        return self.common_forward(data).size()
