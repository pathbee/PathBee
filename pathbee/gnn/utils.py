from networkit import *
import networkx as nx
from scipy.linalg import block_diag
from scipy.sparse import csr_matrix
from scipy.stats import kendalltau
import scipy.sparse as sp
import numpy as np
import torch
import os
from multiprocessing import Process, Queue  
import os  
from typing import List
import logging

def get_out_edges(g_nkit,node_sequence):
    global all_out_dict
    all_out_dict = dict()
    for all_n in node_sequence:
        all_out_dict[all_n]=set()
        
    for all_n in node_sequence:
            _ = g_nkit.forEdgesOf(all_n,nkit_outedges)
            
    return all_out_dict

def get_in_edges(g_nkit,node_sequence):
    global all_in_dict
    all_in_dict = dict()
    for all_n in node_sequence:
        all_in_dict[all_n]=set()
        
    for all_n in node_sequence:
            _ = g_nkit.forInEdgesOf(all_n,nkit_inedges)
            
    return all_in_dict


def nkit_inedges(u,v,weight,edgeid):
    all_in_dict[u].add(v)


def nkit_outedges(u,v,weight,edgeid):
    all_out_dict[u].add(v)

  
def concat_path(*args):  
    """  
    Concatenates arbitrary number of path segments and returns the full path.  
      
    Parameters:  
    *args: Arbitrary number of path segments  
    """  
    # Concatenate the path segments  
    return os.path.join(*args)  

def get_file_without_extension_name(full_path):

    base_name = os.path.basename(full_path)  
    # Remove the extension from the file name  
    filename_without_ext = os.path.splitext(base_name)[0]

    return filename_without_ext
  
def nx2nkit(g_nx):
    
    node_num = g_nx.number_of_nodes()
    g_nkit = Graph(directed=True)
    
    for i in range(node_num):
        g_nkit.addNode()
    
    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)
        
    assert g_nx.number_of_nodes()==g_nkit.numberOfNodes(),"Number of nodes not matching"
    assert g_nx.number_of_edges()==g_nkit.numberOfEdges(),"Number of edges not matching"
        
    return g_nkit


def execute_command(cmd: str) -> None:  
    """  
    Execute a shell command.  
    """
    import subprocess   
    # If the command is None, just return  
    logger = get_logger()
    if cmd is None:  
        return 
        # raise ValueError("the command must not be None")
    logger.info(f"start running {cmd}...")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)   
    logger.info(result.stdout) 
    logger.info(f"End.")  

  
def parallel_process(commands: List[str], num_processes: int) -> None:  
    """  
    Execute shell commands in parallel.  
  
    commands: List of shell commands.  
    num_processes: Number of processes to start.  
    """  
    # Create a queue to hold the commands  
    command_queue = Queue()  
    for cmd in commands:  
        command_queue.put(cmd)  
  
    # Add "stop" signals to the queue  
    for _ in range(num_processes):  
        command_queue.put(None)  
  
    # Create and start initial processes  
    processes = [Process(target=execute_command, args=(command_queue.get(),)) for _ in range(num_processes)]  
    for p in processes:  
        p.start()  
  
    # Wait for all processes to finish  
    for p in processes:  
        p.join()  


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
    return G

def cal_exact_bet(g_nkit):
    #exact_bet = nx.betweenness_centrality(g_nx,normalized=True)

    exact_bet = centrality.Betweenness(g_nkit,normalized=True).run().ranking()
    exact_bet_dict = dict()
    for j in exact_bet:
        exact_bet_dict[j[0]] = j[1]
    return exact_bet_dict

def cal_exact_close(g_nx):
    pass

    
def clique_check(index,node_sequence,all_out_dict,all_in_dict):
    node = node_sequence[index]
    in_nodes = all_in_dict[node]
    out_nodes = all_out_dict[node]

    for in_n in in_nodes:
        tmp_out_nodes = set(out_nodes)
        tmp_out_nodes.discard(in_n)
        if tmp_out_nodes.issubset(all_out_dict[in_n]) == False:
            return False
    
    return True

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




def graph_to_adj_bet(list_graph,list_n_sequence,list_node_num,model_size):
    

    list_adjacency = list()
    list_adjacency_t = list()
    list_degree = list()
    max_nodes = model_size
    zero_list = list()
    list_rand_pos = list()
    list_sparse_diag = list()
    
    for i in range(len(list_graph)):
        print(f"Processing graphs: {i+1}/{len(list_graph)}",end='\r', flush=True)
        graph = list_graph[i]
        edges = list(graph.edges())
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        #self_loops = [i for i in graph.selfloop_edges()]
        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
        node_sequence = list_n_sequence[i]

        adj_temp = nx.adjacency_matrix(graph,nodelist=node_sequence)

        node_num = list_node_num[i]
        
        adj_temp_t = adj_temp.transpose()
        
        arr_temp1 = np.sum(adj_temp,axis=1)
        arr_temp1 = np.reshape(arr_temp1, (-1, 1))

        arr_temp2 = np.sum(adj_temp_t,axis=1)
        arr_temp2 = np.reshape(arr_temp2, (-1, 1))
        

        arr_multi = np.multiply(arr_temp1,arr_temp2)
        
        arr_multi = np.where(arr_multi>0,1.0,0.0)
        
        degree_arr = arr_multi
        
        non_zero_ind = np.nonzero(degree_arr.flatten())
        non_zero_ind = non_zero_ind[0]
        
        g_nkit = nx2nkit(graph)
        

        in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
        all_out_dict = get_out_edges(g_nkit,node_sequence)
        all_in_dict = get_in_edges(g_nkit,in_n_seq) # 非0 


        
        for index in non_zero_ind:
           
            is_zero = clique_check(index,node_sequence,all_out_dict,all_in_dict)
            if is_zero == True:
              
                degree_arr[index,0]=0.0
                    
        adj_temp = adj_temp.multiply(csr_matrix(degree_arr))
        adj_temp_t = adj_temp_t.multiply(csr_matrix(degree_arr))
                

        rand_pos = 0
        top_mat = csr_matrix((rand_pos,rand_pos))
        remain_ind = max_nodes - rand_pos - node_num
        bottom_mat = csr_matrix((remain_ind,remain_ind))
        
        list_rand_pos.append(rand_pos)
        #remain_ind = max_nodes - node_num
        #small_arr = csr_matrix((remain_ind,remain_ind))
        
        #adding extra padding to adj mat,normalise and save as torch tensor

        adj_temp = csr_matrix(adj_temp)
        adj_mat = sp.block_diag((top_mat,adj_temp,bottom_mat))
        
        adj_temp_t = csr_matrix(adj_temp_t)
        adj_mat_t = sp.block_diag((top_mat,adj_temp_t,bottom_mat))
        
        adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
        list_adjacency.append(adj_mat)
        
        adj_mat_t = sparse_mx_to_torch_sparse_tensor(adj_mat_t)
        list_adjacency_t.append(adj_mat_t)
    print("", flush=True)          
    return list_adjacency,list_adjacency_t


def get_logger(name: str = "pathbee", level: str = 'INFO', filename: str = 'pathbee.log') -> logging.Logger:  
    """   
    Create and return a logger with the specified name and level.  
  
    Args:  
        name (str): Name of the logger.  
        level (str): Logging level, default is 'INFO'.  
        filename (str): Name of the log file, default is 'app.log'.  
  
    Returns:  
        logging.Logger: A logger instance.  
    """  
    logger = logging.getLogger(name)  
    logger.setLevel(level)  
  
    # Only add handlers if the logger doesn't have any  
    if not logger.handlers:  
        # Create a file handler and a console handler  
        fh = logging.FileHandler(filename)  
        ch = logging.StreamHandler()  
  
        # Set the level for both handlers  
        fh.setLevel(level)  
        ch.setLevel(level)  
  
        # Create a formatter and add it to the handlers  
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')  
        fh.setFormatter(formatter)  
        ch.setFormatter(formatter)  
  
        # Add the handlers to the logger  
        logger.addHandler(fh)  
        logger.addHandler(ch)  
  
    return logger  


def graph_to_adj_close(list_graph,list_n_sequence,list_node_num,model_size,print_time=False):
    

    list_adjacency = list()
    list_adjacency_mod = list()
    list_degree = list()
    max_nodes = model_size
    zero_list = list()
    list_rand_pos = list()
    list_sparse_diag = list()
    
    for i in range(len(list_graph)):
        print(f"Processing graphs: {i+1}/{len(list_graph)}",end='\r')
        graph = list_graph[i]
        edges = list(graph.edges())
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)

        self_loops = list(nx.selfloop_edges(graph))
        graph.remove_edges_from(self_loops)
        node_sequence = list_n_sequence[i]

        adj_temp = nx.adjacency_matrix(graph,nodelist=node_sequence)

        node_num = list_node_num[i]
        
        adj_temp_t = adj_temp.transpose()
        
        arr_temp1 = np.sum(adj_temp,axis=1)
        arr_temp2 = np.sum(adj_temp_t,axis=1)
        

        arr_multi = np.multiply(arr_temp1,arr_temp2)
        
        arr_multi = np.where(arr_multi>0,1.0,0.0)

        
        degree_arr = arr_multi
        
        non_zero_ind = np.nonzero(degree_arr.flatten())
        non_zero_ind = non_zero_ind[0]
        
        g_nkit = nx2nkit(graph)
        

        in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
        all_out_dict = get_out_edges(g_nkit,node_sequence)
        all_in_dict = get_in_edges(g_nkit,in_n_seq)

        
        for index in non_zero_ind:
           
            is_zero = clique_check(index,node_sequence,all_out_dict,all_in_dict)
            if is_zero == True:
              
                degree_arr[index,0]=0.0

        #modify the in-degree matrix for different layers
        degree_arr = degree_arr.reshape(1,node_num)
 
        #for out_degree
        adj_temp_mod = adj_temp.multiply(csr_matrix(degree_arr))


        rand_pos = 0
        top_mat = csr_matrix((rand_pos,rand_pos))
        remain_ind = max_nodes - rand_pos - node_num
        bottom_mat = csr_matrix((remain_ind,remain_ind))
        
        list_rand_pos.append(rand_pos)
        #remain_ind = max_nodes - node_num
        #small_arr = csr_matrix((remain_ind,remain_ind))
        
        #adding extra padding to adj mat,normalise and save as torch tensor
        
        adj_temp = csr_matrix(adj_temp)
        adj_mat = sp.block_diag((top_mat,adj_temp,bottom_mat))
        
        adj_temp_mod = csr_matrix(adj_temp_mod)
        adj_mat_mod = sp.block_diag((top_mat,adj_temp_mod,bottom_mat))

        
        adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)
        list_adjacency.append(adj_mat)
        
        adj_mat_mod = sparse_mx_to_torch_sparse_tensor(adj_mat_mod)
        list_adjacency_mod.append(adj_mat_mod)

    print("")        
    return list_adjacency,list_adjacency_mod

def ranking_correlation(y_out, true_val, node_num, model_size=None):
    y_out = y_out.view(-1)
    true_val = true_val.view(-1)

    # 截断
    y_out = y_out[:node_num]
    true_val = true_val[:node_num]

    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()

    kt, _ = kendalltau(predict_arr, true_arr)
    return kt



def loss_cal(y_out, true_val, num_nodes, device, model_size):
    top_num = int(0.2 * num_nodes)

    # Safely flatten (don't reshape to model_size)
    y_out = y_out.view(-1)
    true_val = true_val.view(-1)

    # Check consistency
    assert y_out.shape[0] >= num_nodes, f"y_out too short: {y_out.shape[0]} < {num_nodes}"
    assert true_val.shape[0] >= num_nodes, f"true_val too short: {true_val.shape[0]} < {num_nodes}"

    # Sort by true_val
    _, order_y_true = torch.sort(-true_val[:num_nodes])

    sample_num1 = int(4 * num_nodes)
    sample_num2 = int(8 * num_nodes)
    sample_num3 = int(0 * num_nodes)  # Optional

    # Indices
    ind_1 = torch.randint(0, top_num, (sample_num1,), device=device)
    ind_2 = torch.randint(0, top_num, (sample_num1,), device=device)

    ind_3 = torch.randint(0, top_num, (sample_num2,), device=device)
    ind_4 = torch.randint(top_num, num_nodes, (sample_num2,), device=device)

    ind_5 = torch.randint(top_num, num_nodes, (sample_num2,), device=device)
    ind_6 = torch.randint(0, top_num, (sample_num2,), device=device)

    ind_7 = torch.randint(top_num, num_nodes, (sample_num3,), device=device)
    ind_8 = torch.randint(top_num, num_nodes, (sample_num3,), device=device)

    ind_a = torch.cat((ind_1, ind_3, ind_5, ind_7))
    ind_b = torch.cat((ind_2, ind_4, ind_6, ind_8))

    rank_measure = torch.sign(-1 * (ind_a - ind_b)).float()

    input_arr1 = y_out[:num_nodes][order_y_true[ind_a]]
    input_arr2 = y_out[:num_nodes][order_y_true[ind_b]]

    loss_rank = torch.nn.MarginRankingLoss(margin=1.0)(input_arr1, input_arr2, rank_measure)

    return loss_rank


def mse_loss(y_out, true_val, num_nodes, device, model_size):
    y_out = y_out.view(-1)
    true_val = true_val.view(-1)

    assert y_out.shape[0] >= num_nodes, f"y_out too short: {y_out.shape[0]} < {num_nodes}"
    assert true_val.shape[0] >= num_nodes, f"true_val too short: {true_val.shape[0]} < {num_nodes}"

    input_arr1 = y_out[:num_nodes]
    input_arr2 = true_val[:num_nodes]

    loss = torch.nn.MSELoss(reduction='sum')(input_arr1, input_arr2)

    return loss



def sparse_mm_chunked(adj: torch.Tensor, weight: torch.Tensor, chunk_size: int = 128):
    """
    Perform sparse × dense matmul in chunks along columns of weight.
    adj: sparse [N, D]
    weight: dense [D, H]
    return: dense [N, H]
    """
    outputs = []
    D, H = weight.shape
    for j in range(0, H, chunk_size):
        end = min(j + chunk_size, H)
        weight_chunk = weight[:, j:end]             # [D, chunk]
        out_chunk = torch.spmm(adj, weight_chunk)   # [N, chunk]
        outputs.append(out_chunk)
    return torch.cat(outputs, dim=1)  # [N, H]
