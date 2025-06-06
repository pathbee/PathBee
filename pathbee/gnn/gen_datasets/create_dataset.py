import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
import os
random.seed(10)

def reorder_list(input_list,serial_list):
    new_list_tmp = [input_list[j] for j in serial_list]
    return new_list_tmp

def create_dataset(list_data,num_copies, adj_size):
    num_data = len(list_data)
    total_num = num_data*num_copies
    cent_mat = np.zeros((adj_size,total_num), dtype=np.float64)
    list_graph = list()
    list_node_num = list()
    list_n_sequence = list()
    mat_index = 0
    for g_data in list_data:
        graph, cent_dict = g_data
        nodelist = [i for i in graph.nodes()]
        assert len(nodelist)==len(cent_dict),"Number of nodes are not equal"
        node_num = len(nodelist)

        for i in range(num_copies):
            tmp_nodelist = list(nodelist)
            random.shuffle(tmp_nodelist)
            list_graph.append(graph)
            list_node_num.append(node_num)
            list_n_sequence.append(tmp_nodelist)

            for ind,node in enumerate(tmp_nodelist):
                cent_mat[ind,mat_index] = cent_dict[node]
            mat_index +=  1


    serial_list = [i for i in range(total_num)]
    random.shuffle(serial_list)

    list_graph = reorder_list(list_graph,serial_list)
    list_n_sequence = reorder_list(list_n_sequence,serial_list)
    list_node_num = reorder_list(list_node_num,serial_list)
    cent_mat_tmp = cent_mat[:,np.array(serial_list)]
    cent_mat = cent_mat_tmp

    return list_graph, list_n_sequence, list_node_num, cent_mat


def get_split(source_file,num_train,num_test,num_copies,adj_size,save_path):

    with open(source_file,"rb") as fopen:
        list_data = pickle.load(fopen)

    num_graph = len(list_data)
    print(f"{num_train}, {num_test}, {num_graph}")
    assert num_train+num_test == num_graph,"Required split size doesn't match number of graphs in pickle file."
    
    #For training split
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[:num_train],num_copies, adj_size)
    # print(save_path+"_test.pickle","wb")
    with open(os.path.join(save_path,"training.pickle"),"wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen, protocol=4)

    #For test split
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset(list_data[num_train:num_train+num_test], num_copies, adj_size)

    with open(os.path.join(save_path,"test.pickle"),"wb") as fopen:
        pickle.dump([list_graph,list_n_sequence,list_node_num,cent_mat],fopen, protocol=4)
