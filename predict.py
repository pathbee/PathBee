import torch
from networkit import centrality, Graph
import networkx as nx
import pickle
import numpy as np
from scipy.stats import kendalltau
#from utils_gnn import *
from utils import *
import time
import getopt
import sys
import os

import argparse


nets = sys.argv[1:-2]
centrality_folder = sys.argv[-2]
model_path = sys.argv[-1]
# Allows the user to specify the type of centrality calculation ('bc', 'dc', etc.) from the command line.
def read_graph(map_path):
    import networkx as nx
    G = nx.DiGraph()
    f = open(map_path, 'r')
    data = f.readlines()
    f.close()
    for idx, lines in enumerate(data):
        src, dest = lines.split(" ")
        G.add_edge(int(src), int(dest))

    print(f"map {map_path} has {len(G.nodes())} nodes and {len(G.edges())} edges.")
    # print(G.number_of_nodes)
    return G

def cal_exact_bet(g_nkit):

    exact_bet = centrality.Betweenness(g_nkit, normalized=True).run().ranking()
    exact_bet_dict = dict()
    for j in exact_bet:
        exact_bet_dict[j[0]] = j[1]
    # print(exact_bet)
    return exact_bet_dict

def read_bet(mapPath):
    # 打开文件并读取所有行
#    with open('./gnnMeetPll/centrality/' + map_name + "/BC.txt", 'r') as f:
    with open('centralities/bc/' + os.path.basename(mapPath), 'r') as f:
#    with open('./gnnMeetPll/' + map_name + "/BC.txt", 'r') as f:
        lines = f.readlines()

    # 创建一个空字典
    d = {}

    # 遍历所有行并将其解析为key-value对
    for line in lines:
        nums = line.split()  # 使用空格分隔每行的两个数字
        key = int(nums[1])
        value = float(nums[0])
        d[key] = value

    # 按照value排序
    bc_dict = dict(sorted(d.items(), key=lambda item: item[1],reverse=True))
    print(bc_dict)
    return bc_dict

def create_dataset(graphs, num_copies):

    adj_size = 4500000
    num_data = len(graphs)
    total_num = num_data * num_copies
    cent_mat = np.zeros((adj_size,total_num), dtype=float)
    list_graph = list()
    list_node_num = list()
    list_n_sequence = list()
    mat_index = 0
    for g_data in graphs:

        graph, cent_dict = g_data
        nodelist = [i for i in graph.nodes()]
        assert len(nodelist)==len(cent_dict), "Number of nodes are not equal"
        node_num = len(nodelist)

        for i in range(num_copies):
            tmp_nodelist = list(nodelist)
            list_graph.append(graph)
            list_node_num.append(node_num)
            list_n_sequence.append(tmp_nodelist)

            for ind,node in enumerate(tmp_nodelist):
                cent_mat[ind,mat_index] = cent_dict[node]
            mat_index +=  1

    serial_list = [i for i in range(total_num)]
    cent_mat_tmp = cent_mat[:, np.array(serial_list)]
    cent_mat = cent_mat_tmp

    return list_graph, list_n_sequence, list_node_num, cent_mat

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

def test(model, list_adj_test, list_adj_t_test, list_num_node_test, bc_mat_test, model_size, device, map_names):
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

print(os.path.basename(nets[0]))
dest_value_path = os.path.join(centrality_folder, os.path.basename(nets[0])[:-4] + "_value.txt")
dest_ranking_path = os.path.join(centrality_folder, os.path.basename(nets[0])[:-4] + "_ranking.txt")
dest_dir = os.path.dirname(dest_value_path)
print(model_path)
graphs = []

if os.path.exists(dest_value_path) and os.path.exists(dest_ranking_path):
    print(f"{nets[0]} has been calculated before.")
else:
    for net in nets:
        g_nx = read_graph(map_path=net)
        if nx.number_of_isolates(g_nx) > 0:
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
            g_nx = nx.convert_node_labels_to_integers(g_nx) # 10647
        # g_nkit = nx2nkit(g_nx)
        # bet_dict = cal_exact_bet(g_nkit)
        
        ## TODO: generate fake bc here.
        # bet_dict = read_bet(net)
        bet_dict = dict([(index,1) for index in range(nx.number_of_nodes(g_nx))])
        graphs.append([g_nx, bet_dict])


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #model_name = ["1.pt","2.pt","3.pt","4.pt", "5.pt", "6.pt", "7.pt", "8.pt", "9.pt", "10.pt"]


        
    start = time.perf_counter()
    model = torch.load(model_path)
    print(model, flush=True)
    # use the ori-19-model to get the order
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(graphs, num_copies = 1)
    list_adj_test, list_adj_t_test = graph_to_adj_bet(list_graph, list_n_sequence, list_node_num, model_size=4500000)
    end = time.perf_counter()
    print(f'The time of load model is {end-start}s')


    start = time.perf_counter()
    orders, pre_arrs = test(model, list_adj_test, list_adj_t_test, list_node_num, cent_mat, model_size=4500000, device=device, map_names=nets)

    end = time.perf_counter()
    print(f'The time of cal centrality is {end-start}s\n\n')


    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    centrality_values = []
    centrality_rankings = []
    for index, item  in enumerate(pre_arrs[0]):
        centrality_values.append((index,item))
    centrality_rankings = (sorted(dict(centrality_values).items(), key=lambda item: item[1],reverse=True))
    # print(centrality_values)
    # print(centrality_rankings)
    # print(pre_arrs[0])
    with open(dest_value_path, 'w') as f:
    #     # 循环遍历源文件的所有行
        for index_centrality_pair in centrality_values:
            f.write(f"{index_centrality_pair[1]} {index_centrality_pair[0]}\n")

    with open(dest_ranking_path, 'w') as f:
    #     # 循环遍历源文件的所有行
        for index_centrality_pair in centrality_rankings:
            f.write(f"{index_centrality_pair[1]} {index_centrality_pair[0]}\n")
           




