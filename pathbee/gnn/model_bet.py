import torch.nn as nn
import torch.nn.functional as F
from .layer import GNN_Layer
from .layer import GNN_Layer_Init
from .layer import MLP
import torch


class GNN_Bet(nn.Module):
    def __init__(self, ninput, nhid, dropout):
        super(GNN_Bet, self).__init__()

        self.gc1 = GNN_Layer_Init(ninput, nhid)
        self.gc2 = GNN_Layer(nhid, nhid)
        self.gc3 = GNN_Layer(nhid, nhid)
        self.gc4 = GNN_Layer(nhid, nhid)
        self.gc5 = GNN_Layer(nhid, nhid)
        self.gc6 = GNN_Layer(nhid, nhid)
        self.dropout = dropout

        self.score_layer = MLP(nhid, self.dropout)

    def forward(self, adj1, adj2):
        # Layer 1
        x_1 = F.normalize(F.relu(self.gc1(adj1)), p=2, dim=1)
        x2_1 = F.normalize(F.relu(self.gc1(adj2)), p=2, dim=1)

        # Layer 2
        x_2 = F.normalize(F.relu(self.gc2(x_1, adj1)), p=2, dim=1)
        x2_2 = F.normalize(F.relu(self.gc2(x2_1, adj2)), p=2, dim=1)

        # Layer 3
        x_3 = F.normalize(F.relu(self.gc3(x_2, adj1)), p=2, dim=1)
        x2_3 = F.normalize(F.relu(self.gc3(x2_2, adj2)), p=2, dim=1)

        # Layer 4
        x_4 = F.normalize(F.relu(self.gc4(x_3, adj1)), p=2, dim=1)
        x2_4 = F.normalize(F.relu(self.gc4(x2_3, adj2)), p=2, dim=1)

        # Layer 5
        x_5 = F.normalize(F.relu(self.gc5(x_4, adj1)), p=2, dim=1)
        x2_5 = F.normalize(F.relu(self.gc5(x2_4, adj2)), p=2, dim=1)

        # Layer 6 (last, no need to detach here)
        x_6 = F.relu(self.gc6(x_5, adj1))
        x2_6 = F.relu(self.gc6(x2_5, adj2))

        # Aggregate
        x_7 = (x_1 + x_2 + x_3 + x_4 + x_5 + x_6).detach()
        x2_7 = (x2_1 + x2_2 + x2_3 + x2_4 + x2_5 + x2_6).detach()
        # Score MLP (use accumulation instead of stack)
        score1_sum = 0
        for x in [x_1, x_2, x_3, x_4, x_5]:
            with torch.no_grad():
                score1_sum += self.score_layer(x, self.dropout, chunk_size=1000)
        # Keep last layers trainable
        score1_sum += self.score_layer(x_6, self.dropout, chunk_size=1000)
        score1_sum += self.score_layer(x_7, self.dropout, chunk_size=1000)
        score1 = score1_sum / 7

        score2_sum = 0
        for x2 in [x2_1, x2_2, x2_3, x2_4, x2_5]:
            with torch.no_grad():
                score2_sum += self.score_layer(x2, self.dropout, chunk_size=1000)
        score2_sum += self.score_layer(x2_6, self.dropout, chunk_size=1000)
        score2_sum += self.score_layer(x2_7, self.dropout, chunk_size=1000)
        score2 = score2_sum / 7 

        # Final fusion
        x = score1 * score2

        return x

