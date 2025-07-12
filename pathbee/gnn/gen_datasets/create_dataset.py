import networkx as nx
import pickle
import numpy as np
import time
import glob
import random
import os
from multiprocessing import Pool
from functools import partial

random.seed(10)
np.random.seed(10)

def reorder_list(input_list, serial_list):
    return [input_list[j] for j in serial_list]

def create_dataset(list_data, num_copies, adj_size):
    """Optimized version, keeping the exact same interface and output"""
    num_data = len(list_data)
    total_num = num_data * num_copies
    
    # Pre-allocate memory
    cent_mat = np.zeros((adj_size, total_num), dtype=np.float64)
    list_graph = [None] * total_num
    list_node_num = [0] * total_num
    list_n_sequence = [None] * total_num
    
    mat_index = 0
    
    for g_data in list_data:
        graph, cent_dict = g_data
        nodelist = [i for i in graph.nodes()]
        assert len(nodelist) == len(cent_dict), "Number of nodes are not equal"
        node_num = len(nodelist)
        
        # Pre-compute centrality values array to accelerate subsequent access
        cent_values_dict = {node: cent_dict[node] for node in nodelist}
        
        for i in range(num_copies):
            tmp_nodelist = list(nodelist)
            random.shuffle(tmp_nodelist)
            
            list_graph[mat_index] = graph
            list_node_num[mat_index] = node_num
            list_n_sequence[mat_index] = tmp_nodelist
            
            # Vectorized assignment - this is the main performance improvement point
            cent_values = np.array([cent_values_dict[node] for node in tmp_nodelist])
            cent_mat[:node_num, mat_index] = cent_values
            
            mat_index += 1
    
    # Use numpy random permutation, faster than Python's random.shuffle
    serial_list = list(range(total_num))
    random.shuffle(serial_list)  # Keep the same random behavior as the original version
    
    # Reorder
    list_graph = reorder_list(list_graph, serial_list)
    list_n_sequence = reorder_list(list_n_sequence, serial_list)
    list_node_num = reorder_list(list_node_num, serial_list)
    cent_mat_tmp = cent_mat[:, np.array(serial_list)]
    cent_mat = cent_mat_tmp
    
    return list_graph, list_n_sequence, list_node_num, cent_mat

def _process_graph_batch(args):
    """Helper function for parallel processing"""
    graph_data, num_copies, adj_size, start_seed = args
    
    # Set random seed for current process
    random.seed(start_seed)
    np.random.seed(start_seed)
    
    graph, cent_dict = graph_data
    nodelist = [i for i in graph.nodes()]
    node_num = len(nodelist)
    
    # Pre-compute centrality values
    cent_values_dict = {node: cent_dict[node] for node in nodelist}
    
    # Generate all copies
    graphs = []
    sequences = []
    node_nums = []
    cent_columns = []
    
    for i in range(num_copies):
        tmp_nodelist = list(nodelist)
        random.shuffle(tmp_nodelist)
        
        graphs.append(graph)
        sequences.append(tmp_nodelist)
        node_nums.append(node_num)
        
        # Create centrality column
        cent_column = np.zeros(adj_size)
        cent_values = np.array([cent_values_dict[node] for node in tmp_nodelist])
        cent_column[:node_num] = cent_values
        cent_columns.append(cent_column)
    
    return graphs, sequences, node_nums, cent_columns

def create_dataset_parallel(list_data, num_copies, adj_size, n_processes=None):
    """Parallel version, keeping the same interface"""
    if n_processes is None:
        n_processes = min(os.cpu_count(), len(list_data))
    
    # Prepare parameters, ensuring random seed consistency
    args_list = []
    for i, graph_data in enumerate(list_data):
        args_list.append((graph_data, num_copies, adj_size, 10 + i))
    
    # Parallel processing
    with Pool(n_processes) as pool:
        results = pool.map(_process_graph_batch, args_list)
    
    # Merge results
    list_graph = []
    list_n_sequence = []
    list_node_num = []
    cent_columns = []
    
    for graphs, sequences, node_nums, columns in results:
        list_graph.extend(graphs)
        list_n_sequence.extend(sequences)
        list_node_num.extend(node_nums)
        cent_columns.extend(columns)
    
    # Create centrality matrix
    cent_mat = np.column_stack(cent_columns)
    
    # Random reordering
    total_num = len(list_graph)
    serial_list = list(range(total_num))
    random.shuffle(serial_list)
    
    list_graph = reorder_list(list_graph, serial_list)
    list_n_sequence = reorder_list(list_n_sequence, serial_list)
    list_node_num = reorder_list(list_node_num, serial_list)
    cent_mat_tmp = cent_mat[:, np.array(serial_list)]
    cent_mat = cent_mat_tmp
    
    return list_graph, list_n_sequence, list_node_num, cent_mat

def get_split(source_file, num_train, num_test, num_copies, adj_size, save_path, use_parallel=True):
    """Optimized version, keeping the exact same interface"""
    with open(source_file, "rb") as fopen:
        list_data = pickle.load(fopen)
    
    num_graph = len(list_data)
    print(f"{num_train}, {num_test}, {num_graph}")
    assert num_train + num_test == num_graph, "Required split size doesn't match number of graphs in pickle file."
    
    # Choose which version of create_dataset to use
    if use_parallel and len(list_data) > 4:  # Only use parallel processing when data is large enough
        create_func = create_dataset_parallel
        print("Using parallel processing...")
    else:
        create_func = create_dataset
    
    # Training set
    print("Processing training set...")
    start_time = time.time()
    list_graph, list_n_sequence, list_node_num, cent_mat = create_func(
        list_data[:num_train], num_copies, adj_size
    )
    print(f"Training set processed in {time.time() - start_time:.2f} seconds")
    
    with open(os.path.join(save_path, "training.pickle"), "wb") as fopen:
        pickle.dump([list_graph, list_n_sequence, list_node_num, cent_mat], fopen, protocol=4)
    
    # Test set
    print("Processing test set...")
    start_time = time.time()
    list_graph, list_n_sequence, list_node_num, cent_mat = create_func(
        list_data[num_train:num_train + num_test], num_copies, adj_size
    )
    print(f"Test set processed in {time.time() - start_time:.2f} seconds")
    
    with open(os.path.join(save_path, "test.pickle"), "wb") as fopen:
        pickle.dump([list_graph, list_n_sequence, list_node_num, cent_mat], fopen, protocol=4)