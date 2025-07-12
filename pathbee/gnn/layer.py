import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from .utils import sparse_mm_chunked
import torch.nn as nn

class GNN_Layer(Module):
    """
    Layer defined for GNN-Bet
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GNN_Layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GNN_Layer_Init(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, adj):
        out = sparse_mm_chunked(adj, self.weight, chunk_size=2048)
        if self.bias is not None:
            out += self.bias
        return out


class MLP(nn.Module):
    def __init__(self, nhid, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.linear1 = nn.Linear(nhid, 2 * nhid)
        self.linear2 = nn.Linear(2 * nhid, 2 * nhid)
        self.linear3 = nn.Linear(2 * nhid, 1)

    def forward(self, input_vec, dropout, chunk_size=100000):
        """
        input_vec: [N, nhid]
        return: [N, 1]
        """
        outputs = []
        N = input_vec.size(0)

        for i in range(0, N, chunk_size):
            chunk = input_vec[i:i + chunk_size]  # [chunk, nhid]

            score_temp = F.relu(self.linear1(chunk))
            score_temp = F.dropout(score_temp, dropout, self.training)

            score_temp = F.relu(self.linear2(score_temp))
            score_temp = F.dropout(score_temp, dropout, self.training)

            score_temp = self.linear3(score_temp)
            outputs.append(score_temp)

        return torch.cat(outputs, dim=0)
