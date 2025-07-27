import math
import torch
import os
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from .utils import chunked_sparse_matmul, validate_sparse_dimensions
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast

# Configuration for chunking
USE_CHUNKED_MATMUL = os.getenv('PATHBEE_USE_CHUNKED_MATMUL', 'true').lower() == 'true'
CHUNK_THRESHOLD = int(os.getenv('PATHBEE_CHUNK_THRESHOLD', '50000'))


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
        try:
            # Validate inputs for large tensor safety
            if hasattr(adj, 'sparse_sizes') and any(dim > 2**31-1 for dim in adj.sparse_sizes()):
                raise RuntimeError(f"Adjacency matrix dimensions {adj.sparse_sizes()} exceed INT32_MAX")
            
            support = torch.mm(input, self.weight)
            
            # Ensure device compatibility
            adj_device = adj.device() if hasattr(adj, 'device') and callable(adj.device) else getattr(adj, 'device', None)
            if adj_device is not None and adj_device != support.device:
                raise RuntimeError(f"Device mismatch: adj on {adj_device}, support on {support.device}")
            
            print(f"Layer in: chunked_sparse_matmul")
            
            # Safe sparse matrix multiplication - SparseTensor must be on left side
            # output = chunked_sparse_matmul(adj, support)

            with autocast('cuda', enabled=False):
                output = adj.matmul(support)
                
            if self.bias is not None:
                # Ensure bias is on same device
                if self.bias.device != output.device:
                    self.bias = self.bias.to(output.device)
                return output + self.bias
            else:
                return output
        except Exception as e:
            print(f"Error in GNN_Layer forward pass: {e}")
            print(f"Input shape: {input.shape if hasattr(input, 'shape') else 'unknown'}")
            print(f"Adj type: {type(adj)}, shape: {getattr(adj, 'sparse_sizes', getattr(adj, 'shape', 'unknown'))()}")
            raise e

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GNN_Layer_Init(Module):
    """
    First layer of GNN_Init, for embedding lookup
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GNN_Layer_Init, self).__init__()
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

    def forward(self, adj):
        try:
            # Validate inputs for large tensor safety
            if hasattr(adj, 'sparse_sizes') and any(dim > 2**31-1 for dim in adj.sparse_sizes()):
                raise RuntimeError(f"Adjacency matrix dimensions {adj.sparse_sizes()} exceed INT32_MAX")
            
            support = self.weight
            
            # Ensure device compatibility
            adj_device = adj.device() if hasattr(adj, 'device') and callable(adj.device) else getattr(adj, 'device', None)
            if adj_device is not None and adj_device != support.device:
                raise RuntimeError(f"Device mismatch: adj on {adj_device}, support on {support.device}")
            
            print(f"Layer Init: chunked_sparse_matmul")
            
            # Safe sparse matrix multiplication - SparseTensor must be on left side
            # output = chunked_sparse_matmul(adj, support)

            with autocast('cuda', enabled=False):
                output = adj.matmul(support)
            if self.bias is not None:
                # Ensure bias is on same device
                if self.bias.device != output.device:
                    self.bias = self.bias.to(output.device)
                return output + self.bias
            else:
                return output
        except Exception as e:
            print(f"Error in GNN_Layer_Init forward pass: {e}")
            print(f"Adj type: {type(adj)}, shape: {getattr(adj, 'sparse_sizes', getattr(adj, 'shape', 'unknown'))()}")
            raise e
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class MLP(Module):
    def __init__(self, nhid, dropout):
        super(MLP, self).__init__()
        self.dropout = dropout
        self.linear1 = torch.nn.Linear(nhid, 2*nhid)
        self.linear2 = torch.nn.Linear(2*nhid, 2*nhid)
        self.linear3 = torch.nn.Linear(2*nhid, 1)

    def _forward_body(self, x):
        x = F.relu(self.linear1(x))
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.linear2(x))
        x = F.dropout(x, self.dropout, self.training)
        score = self.linear3(x)
        return score

    def forward(self, input_vec, dropout):
        # 对前两层整体做 checkpoint
        score = checkpoint(self._forward_body, input_vec)
        print(f"MLP: checkpoint")
        # 输出层
        # score = self.linear3(x)
        return score
