import torch
from networkit import centrality, Graph
import networkx as nx
import pickle
import numpy as np
from scipy.stats import kendalltau
from .utils import *
import time
import os

# def create_dataset(graphs, num_copies):

#     adj_size = 4500000
#     num_data = len(graphs)
#     total_num = num_data * num_copies
#     cent_mat = np.zeros((adj_size,total_num), dtype=float)
#     list_graph = list()
#     list_node_num = list()
#     list_n_sequence = list()
#     mat_index = 0
#     for g_data in graphs:

#         graph, cent_dict = g_data
#         nodelist = [i for i in graph.nodes()]
#         assert len(nodelist)==len(cent_dict), "Number of nodes are not equal"
#         node_num = len(nodelist)

#         for i in range(num_copies):
#             tmp_nodelist = list(nodelist)
#             list_graph.append(graph)
#             list_node_num.append(node_num)
#             list_n_sequence.append(tmp_nodelist)

#             for ind,node in enumerate(tmp_nodelist):
#                 cent_mat[ind,mat_index] = cent_dict[node]
#             mat_index +=  1

#     serial_list = [i for i in range(total_num)]
#     cent_mat_tmp = cent_mat[:, np.array(serial_list)]
#     cent_mat = cent_mat_tmp

#     return list_graph, list_n_sequence, list_node_num, cent_mat

def ranking_correlation(y_out, true_val, node_num, model_size, map_name):
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))

    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()
    predict_arr = predict_arr[:node_num]
    true_arr = true_arr[:node_num]
    order_result = np.argsort(predict_arr)[::-1]

    kt, _ = kendalltau(predict_arr[:node_num], true_arr[:node_num])

    return kt, order_result, predict_arr[:node_num]

def inference(model, list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test, model_size, device, map_names):
    model.eval()
    num_samples_test = len(list_adj_test)
    orders = []
    pre_arrs = []
    for j in range(num_samples_test):
        adj = list_adj_test[j]
        adj_t = list_adj_t_test[j]
        adj=adj.to(device)
        adj_t = adj_t.to(device)
        num_nodes = list_num_node_test[j]

        y_out = model(adj, adj_t)

        true_arr = torch.from_numpy(bc_mat_test[:,j]).float()
        true_val = true_arr.to(device)

        kt, order, pred_arr = ranking_correlation(y_out, true_val, num_nodes, model_size, map_names[j])

        orders.append(order)
        pre_arrs.append(pred_arr)


    return orders, pre_arrs

def preprocess(graph_path): 
    g_nx = read_graph(map_path=graph_path)
    if nx.number_of_isolates(g_nx) > 0:
        g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
        g_nx = nx.convert_node_labels_to_integers(g_nx) # 10647
    
    bet_dict = dict([(index,1) for index in range(nx.number_of_nodes(g_nx))])
    return (g_nx, bet_dict)



           




