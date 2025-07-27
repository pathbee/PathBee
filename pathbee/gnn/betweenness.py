import numpy as np
import pickle
import networkx as nx
import torch
from .utils import *
import random
import torch.nn as nn
from .model_bet import GNN_Bet
torch.manual_seed(20)
import argparse


from torch.amp import autocast, GradScaler

def train(device, adj_size, list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train, model, optimizer, ltype="pb"):
    model.train()
    scaler = GradScaler('cuda')
    loss_train = 0
    num_samples_train = len(list_adj_train)
    # num_samples_train = 2   

    # print adj size
    print(f"adj size: {adj_size}")
    
    for i in range(num_samples_train):
        adj = list_adj_train[i].to(device)
        adj_t = list_adj_t_train[i].to(device)
        num_nodes = list_num_node_train[i]
        
        optimizer.zero_grad()
        
        with autocast('cuda'):
            y_out = model(adj, adj_t)
            print(f"y_out.requires_grad: {y_out.requires_grad}")
            true_arr = torch.from_numpy(bc_mat_train[:, i]).float().to(device)
            
            if ltype == "pb":
                loss_rank = mrl_loss_sampling(y_out, true_arr, num_nodes, device, adj_size)
            elif ltype == "mrl":
                loss_rank = mrl_loss(y_out, true_arr, num_nodes, device, adj_size)
            elif ltype == "mse":
                loss_rank = mse_loss(y_out, true_arr, num_nodes, device, adj_size)
            else:
                raise ValueError(f"Invalid loss type: {ltype}")
        
        loss_train += loss_rank.item()
        print(f"loss_rank: {loss_rank}")
        # print loss_train
        print(f"loss_train: {loss_train}")
        loss_rank.backward()
        optimizer.step()
        del adj, adj_t, num_nodes, y_out, true_arr, loss_rank
        torch.cuda.empty_cache()
    
    print(f"loss_train: {loss_train}", flush=True)
    return loss_train / num_samples_train

# def train(device, adj_size, list_adj_train, list_adj_t_train, list_num_node_train, bc_mat_train, model, optimizer, ltype="pb"):
#     model.train()
#     loss_train = 0
#     num_samples_train = len(list_adj_train)

#     print(f"adj size: {adj_size}")
    
#     for i in range(num_samples_train):
#         adj = list_adj_train[i].to(device)
#         adj_t = list_adj_t_train[i].to(device)
#         num_nodes = list_num_node_train[i]

#         optimizer.zero_grad()

#         y_out = model(adj, adj_t)  # model 应该已在内部控制精度
#         print(f"y_out.requires_grad: {y_out.requires_grad}")

#         true_arr = torch.from_numpy(bc_mat_train[:, i]).to(dtype=y_out.dtype, device=device)

#         if ltype == "pb":
#             loss_rank = mrl_loss_sampling(y_out, true_arr, num_nodes, device, adj_size)
#         elif ltype == "mrl":
#             loss_rank = mrl_loss(y_out, true_arr, num_nodes, device, adj_size)
#         elif ltype == "mse":
#             loss_rank = mse_loss(y_out, true_arr, num_nodes, device, adj_size)
#         else:
#             raise ValueError(f"Invalid loss type: {ltype}")

#         loss_train += loss_rank.item()
#         loss_rank.backward()
#         optimizer.step()

#     print(f"loss_train: {loss_train}", flush=True)
#     return loss_train / num_samples_train




def test(device, adj_size, list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test, model, optimizer, ltype="pb") -> int:
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)
    # num_samples_test = 1
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        with autocast('cuda'):
            y_out = model(adj,adj_t)

            true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
            true_val = true_arr.to(device)
            if ltype == "pb":
                loss_rank = mrl_loss_sampling(y_out, true_val, num_nodes, device, adj_size)
            elif ltype == "mrl":
                loss_rank = mrl_loss(y_out, true_val, num_nodes, device, adj_size)
            elif ltype == "mse":
                loss_rank =  mse_loss(y_out, true_val, num_nodes, device, adj_size)
            else:
                raise ValueError(f"Invalid loss type: {ltype}")
            loss_val = loss_val + float(loss_rank)
            kt = ranking_correlation(y_out,true_val,num_nodes,adj_size)
            list_kt.append(kt)
            #g_tmp = list_graph_test[j]
            #print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")
    print(f' loss_test: {loss_val}', flush=True)
    print(f"  Average KT score on test graphs is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}", flush=True)
    return loss_val



