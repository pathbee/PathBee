import torch.nn as nn
import torch.nn.functional as F
from .layer import GNN_Layer
from .layer import GNN_Layer_Init
from .layer import MLP
import torch
from torch.utils import checkpoint

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
        print(f"adj1.dtype: {adj1.dtype}")
        print(f"adj2.dtype: {adj2.dtype}")
        
        # 第一层 - 保持梯度
        x1 = F.normalize(F.relu(self.gc1(adj1)), p=2, dim=1)
        x2 = F.normalize(F.relu(self.gc1(adj2)), p=2, dim=1)
        print(f"x1.dtype: {x1.dtype}")
        print(f"x2.dtype: {x2.dtype}")
        
        # 初始化累积分数和残差
        score1_acc = self.score_layer(x1, self.dropout)  # ✅ 第一层保持梯度
        score2_acc = self.score_layer(x2, self.dropout)  # ✅ 第一层保持梯度
        residual1_acc = x1.clone()
        residual2_acc = x2.clone()
        
        # 当前层的输入
        current_x1 = x1
        current_x2 = x2
        
        # 中间层 (gc2, gc3, gc4, gc5) - 断开梯度
        layers = [self.gc2, self.gc3, self.gc4, self.gc5]
        # layers = [self.gc2]
        for i, layer in enumerate(layers):
            # 使用checkpoint
            x1_new = F.normalize(F.relu(checkpoint.checkpoint(layer, current_x1, adj1)), p=2, dim=1)
            x2_new = F.normalize(F.relu(checkpoint.checkpoint(layer, current_x2, adj2)), p=2, dim=1)
            
            # ❌ 中间层断开梯度（gc2, gc3, gc4, gc5无法更新）
            score1_acc = score1_acc + self.score_layer(x1_new.detach(), self.dropout)
            score2_acc = score2_acc + self.score_layer(x2_new.detach(), self.dropout)
            residual1_acc = residual1_acc + x1_new.detach()
            residual2_acc = residual2_acc + x2_new.detach()
            
            # 释放前一层输出，更新当前层输出
            if current_x1 is not x1:
                del current_x1, current_x2
            current_x1 = x1_new
            current_x2 = x2_new
            
            if i % 2 == 1:
                torch.cuda.empty_cache()
        
        # 最后一层 (gc6) - 保持梯度
        x1_final = F.relu(checkpoint.checkpoint(self.gc6, current_x1, adj1))
        x2_final = F.relu(checkpoint.checkpoint(self.gc6, current_x2, adj2))
        del current_x1, current_x2
        
        # ✅ 最后一层保持梯度（gc6可以更新）
        score1_acc = score1_acc + self.score_layer(x1_final, self.dropout)
        score2_acc = score2_acc + self.score_layer(x2_final, self.dropout)
        residual1_acc = residual1_acc + x1_final
        residual2_acc = residual2_acc + x2_final
        
        # 残差连接的分数 - 这里包含了所有层，所以第一层和最后一层都能通过这里收到梯度
        # ✅ 残差连接保持梯度（第一层和最后一层都能通过residual收到额外梯度）
        score1_acc = score1_acc + self.score_layer(residual1_acc, self.dropout)
        score2_acc = score2_acc + self.score_layer(residual2_acc, self.dropout)
        
        del x1, x2, x1_final, x2_final, residual1_acc, residual2_acc
        
        # 计算平均分数（7个分数的平均）
        score1_acc = score1_acc / 7.0
        score2_acc = score2_acc / 7.0
        
        result = torch.mul(score1_acc, score2_acc)
        
        del score1_acc, score2_acc
        torch.cuda.empty_cache()
        
        return result