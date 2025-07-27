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
from torch_sparse import SparseTensor
from torch.utils import checkpoint
import gc
import time
from concurrent.futures import ThreadPoolExecutor

def ensure_sparse_tensor_device(sparse_tensor, target_device):
    """Safely move SparseTensor to target device with validation."""
    # Get current device properly
    current_device = None
    if hasattr(sparse_tensor, 'device'):
        if callable(sparse_tensor.device):
            current_device = sparse_tensor.device()
        else:
            current_device = sparse_tensor.device
    
    if current_device is None:
        return sparse_tensor
        
    if current_device == target_device:
        return sparse_tensor
        
    logger = get_logger()
    logger.debug(f"Moving SparseTensor from {current_device} to {target_device}")
    
    try:
        return sparse_tensor.to(target_device)
    except Exception as e:
        logger.error(f"Failed to move SparseTensor to {target_device}: {e}")
        raise RuntimeError(f"Device transfer failed: {e}")

def validate_sparse_dimensions(sparse_tensor, max_dim=2**31-1):
    """Validate SparseTensor dimensions don't exceed limits."""
    if hasattr(sparse_tensor, 'sparse_sizes'):
        dims = sparse_tensor.sparse_sizes()
        if any(dim > max_dim for dim in dims):
            raise RuntimeError(f"SparseTensor dimensions {dims} exceed maximum {max_dim}")
    return True

def _sparse_chunk_matmul(chunk_sparse, dense_matrix):
    return chunk_sparse.matmul(dense_matrix)

def chunked_sparse_matmul(sparse_adj, dense_matrix, chunk_size=20000):
    logger = get_logger()
    
    if not hasattr(sparse_adj, 'sparse_sizes'):
        return sparse_adj.matmul(dense_matrix)
    
    n_rows, n_cols = sparse_adj.sparse_sizes()
    *_, out_dim = dense_matrix.shape
    
    if n_rows <= chunk_size:
        return sparse_adj.matmul(dense_matrix)
    
    sparse_device = sparse_adj.device() if callable(sparse_adj.device) else sparse_adj.device
    if sparse_device != dense_matrix.device:
        dense_matrix = dense_matrix.to(sparse_device)
    
    # 关键改进：使用列表收集结果，最后一次性拼接
    chunk_results = []
    
    for start_row in range(0, n_rows, chunk_size):
        end_row = min(start_row + chunk_size, n_rows)
        
        try:
            # 创建chunk
            storage = sparse_adj.storage
            row_mask = (storage.row() >= start_row) & (storage.row() < end_row)
            if row_mask.sum() == 0:
                # 空chunk，添加零张量
                chunk_results.append(torch.zeros(
                    end_row - start_row, out_dim, 
                    device=sparse_device, dtype=dense_matrix.dtype,
                    requires_grad=dense_matrix.requires_grad
                ))
                continue
            
            # print chunk
            print(f"chunk_rows: {chunk_rows}")
            
            chunk_rows = storage.row()[row_mask] - start_row
            chunk_cols = storage.col()[row_mask]
            chunk_vals = storage.value()[row_mask]
            
            # 在创建SparseTensor时就指定正确的设备
            chunk_sparse = SparseTensor(
                row=chunk_rows,
                col=chunk_cols, 
                value=chunk_vals,
                sparse_sizes=(end_row - start_row, n_cols)
            ).to(sparse_device)
            
            # 使用checkpoint
            chunk_result = checkpoint.checkpoint(
                _sparse_chunk_matmul, 
                chunk_sparse, 
                dense_matrix,
                use_reentrant=False  # 使用新的checkpoint实现
            )
            
            chunk_results.append(chunk_result)
            
            # 立即删除中间变量
            del chunk_sparse, chunk_rows, chunk_cols, chunk_vals, row_mask
            
        except Exception as e:
            logger.error(f"Error in chunk {start_row}-{end_row}: {e}")
            return sparse_adj.matmul(dense_matrix)
        
        # 每几个chunk清理一下缓存

        if (start_row // chunk_size) % 4 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    
    # 一次性拼接所有结果
    output = torch.cat(chunk_results, dim=0)
    
    # 清理chunk_results列表
    del chunk_results
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output

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

def scipy_to_sparse_tensor(coo_matrix) -> SparseTensor:
    """Convert a scipy sparse matrix to a torch_sparse SparseTensor with safety checks."""
    logger = get_logger()
    
    if hasattr(coo_matrix, 'tocoo'):
        coo = coo_matrix.tocoo()
    else:
        coo = coo_matrix
     
    coo = coo.astype(np.float32)
    
    # Ensure indices are int64 and values are float32
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    val = torch.from_numpy(coo.data.astype(np.float32))

    sparse_sizes = (int(coo.shape[0]), int(coo.shape[1]))
    
    sparse_tensor = SparseTensor(row=row, col=col, value=val, sparse_sizes=sparse_sizes)
    return sparse_tensor.coalesce()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def graph_to_adj_bet(list_graph, list_n_sequence, list_node_num, model_size):
    """
    向量化优化版本 - 直接替换原函数
    保持相同的输入输出接口，但大幅提升性能
    """
    list_adjacency = []
    list_adjacency_t = []
    list_rand_pos = []
    
    max_nodes = model_size
    total_graphs = len(list_graph)
    
    # 动态调整打印频率
    print_interval = max(1, total_graphs // 50)
    
    for i in range(total_graphs):
        if i % print_interval == 0 or i == total_graphs - 1:
            print(f"Processing graphs: {i+1}/{total_graphs}", end='\r', flush=True)
        
        # 预处理图 - 优化版本
        graph = list_graph[i]
        if hasattr(graph, 'edges') and hasattr(graph, 'nodes'):
            # 已经是NetworkX图对象，避免重建
            working_graph = graph.copy() if nx.number_of_selfloops(graph) > 0 else graph
            if nx.number_of_selfloops(working_graph) > 0:
                working_graph.remove_edges_from(nx.selfloop_edges(working_graph))
        else:
            # 从边列表创建图
            edges = list(graph.edges()) if hasattr(graph, 'edges') else graph
            working_graph = nx.MultiDiGraph()
            working_graph.add_edges_from(edges)
            # 移除自环
            self_loops = list(nx.selfloop_edges(working_graph))
            if self_loops:
                working_graph.remove_edges_from(self_loops)
        
        node_sequence = list_n_sequence[i]
        node_num = list_node_num[i]
        
        # 创建邻接矩阵
        adj_temp = nx.adjacency_matrix(working_graph, nodelist=node_sequence)
        
        # 高效的度计算 - 向量化版本
        out_degrees = np.asarray(adj_temp.sum(axis=1)).ravel()
        in_degrees = np.asarray(adj_temp.sum(axis=0)).ravel()
        
        # 布尔运算优化
        degree_mask = (out_degrees > 0) & (in_degrees > 0)
        degree_arr = degree_mask.astype(float).reshape(-1, 1)
        
        # 向量化的Clique检查
        non_zero_ind = np.where(degree_mask)[0]
        
        if len(non_zero_ind) > 0:
            g_nkit = nx2nkit(working_graph)
            in_n_seq = [node_sequence[nz_ind] for nz_ind in non_zero_ind]
            all_out_dict = get_out_edges(g_nkit, node_sequence)
            all_in_dict = get_in_edges(g_nkit, in_n_seq)
            
            # 批量处理clique检查
            clique_results = []
            for index in non_zero_ind:
                is_zero = clique_check(index, node_sequence, all_out_dict, all_in_dict)
                clique_results.append(is_zero)
            
            # 向量化更新degree_arr
            clique_indices = non_zero_ind[np.array(clique_results)]
            if len(clique_indices) > 0:
                degree_arr[clique_indices, 0] = 0
        
        # 稀疏矩阵运算
        degree_sparse = csr_matrix(degree_arr)
        adj_temp = adj_temp.multiply(degree_sparse)
        adj_temp_t = adj_temp.T.multiply(degree_sparse)
        
        # 优化的padding策略
        rand_pos = 0
        remain_ind = max_nodes - rand_pos - node_num
        
        # 水平padding - 优化版本
        current_cols = adj_temp.shape[1]
        if current_cols < max_nodes:
            right_pad_size = max_nodes - current_cols
            right_pad_shape = (adj_temp.shape[0], right_pad_size)
            right_mat = csr_matrix(right_pad_shape)
            adj_temp_padded = sp.hstack([adj_temp, right_mat], format='csr')
            adj_temp_t_padded = sp.hstack([adj_temp_t, right_mat], format='csr')
        else:
            adj_temp_padded = adj_temp
            adj_temp_t_padded = adj_temp_t
        
        # 垂直padding - 使用列表收集然后一次性vstack
        matrices_to_stack = []
        matrices_to_stack_t = []
        
        if rand_pos > 0:
            top_mat = csr_matrix((rand_pos, max_nodes))
            matrices_to_stack.append(top_mat)
            matrices_to_stack_t.append(top_mat)
        
        matrices_to_stack.append(adj_temp_padded)
        matrices_to_stack_t.append(adj_temp_t_padded)
        
        if remain_ind > 0:
            bottom_mat = csr_matrix((remain_ind, max_nodes))
            matrices_to_stack.append(bottom_mat)
            matrices_to_stack_t.append(bottom_mat)
        
        # 高效的vstack操作
        if len(matrices_to_stack) == 1:
            adj_mat_csr = matrices_to_stack[0]
            adj_mat_t_csr = matrices_to_stack_t[0]
        else:
            adj_mat_csr = sp.vstack(matrices_to_stack, format='csr')
            adj_mat_t_csr = sp.vstack(matrices_to_stack_t, format='csr')
        
        # 转换为稀疏张量
        adj_mat = scipy_to_sparse_tensor(adj_mat_csr.tocoo())
        adj_mat_t = scipy_to_sparse_tensor(adj_mat_t_csr.tocoo())
        
        list_adjacency.append(adj_mat)
        list_adjacency_t.append(adj_mat_t)
        list_rand_pos.append(rand_pos)
        
        # 内存清理 - 显式删除大对象
        del adj_temp, adj_temp_t, degree_arr, degree_sparse
        if 'adj_temp_padded' in locals():
            del adj_temp_padded, adj_temp_t_padded
        del adj_mat_csr, adj_mat_t_csr
    
    print("", flush=True)
    return list_adjacency, list_adjacency_t


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
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')  
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
        adj_mat_csr = sp.block_diag((top_mat,adj_temp,bottom_mat))
        
        adj_temp_mod = csr_matrix(adj_temp_mod)
        adj_mat_mod_csr = sp.block_diag((top_mat,adj_temp_mod,bottom_mat))

        
        # Add memory and dimension checks before conversion
        logger = get_logger()
        coo_mat = adj_mat_csr.tocoo()
        coo_mat_mod = adj_mat_mod_csr.tocoo()
        
        # Memory usage estimation (rough)
        est_memory_mb = (coo_mat.nnz * (8 + 8 + 4)) / (1024 * 1024)  # int64 + int64 + float32
        logger.info(f"Estimated memory for sparse tensor: {est_memory_mb:.2f} MB")
        
        if est_memory_mb > 10000:  # > 10GB warning
            logger.warning(f"Large sparse tensor detected: {est_memory_mb:.2f} MB. Consider chunking.")
        
        adj_mat = scipy_to_sparse_tensor(coo_mat)
        adj_mat_mod = scipy_to_sparse_tensor(coo_mat_mod)
        
        list_adjacency.append(adj_mat)
        list_adjacency_mod.append(adj_mat_mod)

    print("")        
    return list_adjacency,list_adjacency_mod

def ranking_correlation(y_out,true_val,node_num,model_size):
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))

    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()


    kt,_ = kendalltau(predict_arr[:node_num],true_arr[:node_num])

    return kt


def mrl_loss_sampling(y_out,true_val,num_nodes,device,model_size):
   top_num = int(0.2 * num_nodes)

   y_out = y_out.reshape((model_size))
   true_val = true_val.reshape((model_size))

   _, order_y_true = torch.sort(-true_val[:num_nodes])

   sample_num1 = int(4*num_nodes)
   sample_num2 = int(8*num_nodes)
   sample_num3 = int(0*num_nodes)  
  
  # 15-15
   ind_1 = torch.randint(0, top_num, (sample_num1, )).long().to(device)
   ind_2 = torch.randint(0, top_num, (sample_num1, )).long().to(device)
  # 15-85 
   ind_3 = torch.randint(0, top_num, (sample_num2, )).long().to(device)
   ind_4 = torch.randint(top_num, num_nodes, (sample_num2, )).long().to(device)
  # 85-15
   ind_5 = torch.randint(top_num, num_nodes, (sample_num2, )).long().to(device)
   ind_6 = torch.randint(0, top_num, (sample_num2, )).long().to(device)
  # 85-85
   ind_7 = torch.randint(top_num, num_nodes, (sample_num3, )).long().to(device)
   ind_8 = torch.randint(top_num, num_nodes, (sample_num3, )).long().to(device)

   ind_a = torch.cat((ind_1, ind_3, ind_5, ind_7))
   ind_b = torch.cat((ind_2, ind_4, ind_6, ind_8))

   rank_measure = torch.sign(-1 * (ind_a - ind_b)).float()

   input_arr1 = y_out[:num_nodes][order_y_true[ind_a]].to(device)
   input_arr2 = y_out[:num_nodes][order_y_true[ind_b]].to(device)

   loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1, input_arr2, rank_measure)

   return loss_rank

def mrl_loss(y_out,true_val,num_nodes,device,model_size):
   top_num = int(0.2 * num_nodes)

   y_out = y_out.reshape((model_size))
   true_val = true_val.reshape((model_size))

   _, order_y_true = torch.sort(-true_val[:num_nodes])

   sample_num1 = int(4*num_nodes)
   sample_num2 = int(8*num_nodes)
   sample_num3 = int(0*num_nodes)  
  
  # 15-15
   ind_1 = torch.randint(0, num_nodes, (sample_num1, )).long().to(device)
   ind_2 = torch.randint(0, num_nodes, (sample_num1, )).long().to(device)
  # 15-85 
   ind_3 = torch.randint(0, num_nodes, (sample_num2, )).long().to(device)
   ind_4 = torch.randint(0, num_nodes, (sample_num2, )).long().to(device)
  # 85-15
   ind_5 = torch.randint(0, num_nodes, (sample_num2, )).long().to(device) 
   ind_6 = torch.randint(0, num_nodes, (sample_num2, )).long().to(device)
  # 85-85
   ind_7 = torch.randint(0, num_nodes, (sample_num3, )).long().to(device)
   ind_8 = torch.randint(0, num_nodes, (sample_num3, )).long().to(device)

   ind_a = torch.cat((ind_1, ind_3, ind_5, ind_7))
   ind_b = torch.cat((ind_2, ind_4, ind_6, ind_8))

   rank_measure = torch.sign(-1 * (ind_a - ind_b)).float()

   input_arr1 = y_out[:num_nodes][order_y_true[ind_a]].to(device)
   input_arr2 = y_out[:num_nodes][order_y_true[ind_b]].to(device)

   loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1, input_arr2, rank_measure)

   return loss_rank

def mse_loss(y_out, true_val, num_nodes, device, model_size):
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))
    input_arr1 = y_out[:num_nodes].to(device)
    input_arr2 = true_val[:num_nodes].to(device)
    # print(f"predict: {y_out[:10]}")
    # print(f"true: {true_val[:10]}")

    loss = torch.nn.MSELoss(reduction='sum').forward(input_arr1, input_arr2)

    return loss

