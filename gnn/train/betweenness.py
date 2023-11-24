import numpy as np
import pickle
import networkx as nx
import torch
from utils import *
import random
import torch.nn as nn
from model_bet import GNN_Bet
torch.manual_seed(20)
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, flush=True)
#Loading graph data
parser = argparse.ArgumentParser()
parser.add_argument("--g",default="SF")
parser.add_argument("--l",default="MRL")
args = parser.parse_args()
gtype = args.g
ltype = args.l
print(gtype, flush=True)
print(ltype, flush=True)

if gtype == "SF":
    data_path = "graphs/synthetic/data_splits/"
    print("Scale-free graphs selected.", flush=True)

#Load training data
print(f"Loading data...", flush=True)
with open(data_path+"training.pickle","rb") as fopen:
    list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)
    # print(f"{list_num_node_train}")

with open(data_path+"test.pickle","rb") as fopen:
    list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)

model_size = 4500000
#Get adjacency matrices from graphs
print(f"Graphs to adjacency conversion.", flush=True)

list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train,list_n_seq_train,list_num_node_train,model_size)
list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test,list_n_seq_test,list_num_node_test,model_size)

def train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train):
    model.train()
    total_count_train = list()
    loss_train = 0
    num_samples_train = len(list_adj_train)
    for i in range(num_samples_train):
        adj = list_adj_train[i]
        num_nodes = list_num_node_train[i]
        adj_t = list_adj_t_train[i]
        adj = adj.to(device)
        adj_t = adj_t.to(device)

        optimizer.zero_grad()
            
        y_out = model(adj,adj_t)
        true_arr = torch.from_numpy(bc_mat_train[:,i]).float()
        true_val = true_arr.to(device)
        if ltype == "MRL":
            # print("MRL has been selected")
            loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
        else:
            # print("MSE has been selected")
            loss_rank = mse_loss(y_out,true_val,num_nodes,device,model_size)
        loss_train = loss_train + float(loss_rank)
        loss_rank.backward()
        optimizer.step()
    print(f"loss_train: {loss_train}", flush=True)

def test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test):
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
        if ltype == "MRL":
            loss_rank = loss_cal(y_out,true_val,num_nodes,device,model_size)
        else:
            loss_rank = mse_loss(y_out,true_val,num_nodes,device,model_size)
        loss_val = loss_val + float(loss_rank)
        kt = ranking_correlation(y_out,true_val,num_nodes,model_size)
        list_kt.append(kt)
        #g_tmp = list_graph_test[j]
        #print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")
    print(f' loss_test: {loss_val}', flush=True)
    print(f"  Average KT score on test graphs is: {np.mean(np.array(list_kt))} and std: {np.std(np.array(list_kt))}", flush=True)

# Model parameters
hidden = 12

model = GNN_Bet(ninput=model_size, nhid=hidden, dropout=0.6)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
num_epoch = 10

print("Training", flush=True)
print(f"Total Number of epoches: {num_epoch}", flush=True)
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}", flush=True)
    train(list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train)

    #to check test loss while training
    with torch.no_grad():
        test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)

    # save the model
    torch.save(model, f"gnn/models/MRL_6layer_sampling_{e+1}.pt")

#test on 10 test graphs and print average KT Score and its stanard deviation
with torch.no_grad():
    test(list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test)