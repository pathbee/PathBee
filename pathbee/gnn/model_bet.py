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
        
        # 第一层
        x1 = F.normalize(F.relu(self.gc1(adj1)), p=2, dim=1)
        x2 = F.normalize(F.relu(self.gc1(adj2)), p=2, dim=1)
        print(f"x1.dtype: {x1.dtype}")
        print(f"x2.dtype: {x2.dtype}")
        
        # 初始化累积分数和残差
        score1_acc = self.score_layer(x1, self.dropout)
        score2_acc = self.score_layer(x2, self.dropout)
        residual1_acc = x1.clone()  # 使用clone避免就地操作问题
        residual2_acc = x2.clone()
        
        # 当前层的输入（用于下一层）
        current_x1 = x1
        current_x2 = x2
        
        layers = [self.gc2, self.gc3, self.gc4, self.gc5]
        for i, layer in enumerate(layers):
            # 使用checkpoint，传入前一层输出作为特征
            x1_new = F.normalize(F.relu(checkpoint.checkpoint(layer, current_x1, adj1)), p=2, dim=1)
            x2_new = F.normalize(F.relu(checkpoint.checkpoint(layer, current_x2, adj2)), p=2, dim=1)
            
            # 累积分数和残差
            score1_acc = score1_acc + self.score_layer(x1_new.detach(), self.dropout)
            score2_acc = score2_acc + self.score_layer(x2_new.detach(), self.dropout)
            residual1_acc = residual1_acc + x1_new.detach()
            residual2_acc = residual2_acc + x2_new.detach()
            
            # 释放前一层输出，更新当前层输出
            if current_x1 is not x1:  # 避免删除第一层输出（可能还在使用）
                del current_x1, current_x2
            current_x1 = x1_new
            current_x2 = x2_new
            
            if i % 2 == 1:
                torch.cuda.empty_cache()
        
        # 最后一层（不做normalize）
        x1_final = F.relu(checkpoint.checkpoint(self.gc6, current_x1, adj1))
        x2_final = F.relu(checkpoint.checkpoint(self.gc6, current_x2, adj2))
        del current_x1, current_x2
        
        # 最后一层的分数和残差
        score1_acc = score1_acc + self.score_layer(x1_final.detach(), self.dropout)
        score2_acc = score2_acc + self.score_layer(x2_final.detach(), self.dropout)
        residual1_acc = residual1_acc + x1_final.detach()
        residual2_acc = residual2_acc + x2_final.detach()
        
        # 残差连接的分数
        score1_acc = score1_acc + self.score_layer(residual1_acc.detach(), self.dropout)
        score2_acc = score2_acc + self.score_layer(residual2_acc.detach(), self.dropout)
        
        del x1, x2, x1_final, x2_final, residual1_acc, residual2_acc
        
        # 计算平均分数（7个分数的平均）
        score1_acc = score1_acc / 7.0
        score2_acc = score2_acc / 7.0
        
        result = torch.mul(score1_acc, score2_acc)
        
        del score1_acc, score2_acc
        torch.cuda.empty_cache()
        
        return result