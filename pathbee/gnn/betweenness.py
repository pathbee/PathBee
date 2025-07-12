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


def train(device, adj_size, list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train, model, optimizer ):
    model.train()
    ltype = "MRL"
    total_count_train = list()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_t = list_adj_t_train[i]
        adj = adj.to(device, non_blocking=True)
        adj_t = adj_t.to(device, non_blocking=True)

        optimizer.zero_grad()
            
        y_out = model(adj,adj_t)
        true_arr = torch.from_numpy(bc_mat_train[:,i]).float()
        true_val = true_arr.to(device)
        if ltype == "MRL":
            # print("MRL has been selected")
            loss_rank = loss_cal(y_out,true_val,num_nodes,device, adj_size)
        else:
            # print("MSE has been selected")
            loss_rank = mse_loss(y_out,true_val,num_nodes,device, adj_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()
    print(f"loss_train: {loss_train}", flush=True)

def test(device, adj_size, list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test, model, optimizer) -> int:
    model.eval()
    loss_val = 0
    list_kt = list()
    num_samples_test = len(list_adj_test)
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]
        
        y_out = model(adj,adj_t)

        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)
        loss_rank = loss_cal(y_out,true_val,num_nodes,device,adj_size)
        loss_val = loss_val + float(loss_rank)
        kt = ranking_correlation(y_out,true_val,num_nodes,adj_size)
        list_kt.append(kt)
        #g_tmp = list_graph_test[j]
        #print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")
    print(f' loss_test: {loss_val}', flush=True)
    print(f"  Average KT score on test graphs is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}", flush=True)
    return loss_val



