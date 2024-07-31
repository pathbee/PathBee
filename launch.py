from pathbee.gnn.gen_datasets.create_dataset import *
from pathbee.gnn.gen_datasets.generate_graph import *
from pathbee.gnn.betweenness import *
from pathbee.gnn.predict import *
import fire
from typing import List, Union

def gen_dataset(
        num_of_graphs: int = 5,
        num_train: int = 4,
        num_test: int = 1,
        num_copies: int = 50,
        graph_type: str = "SF",
        dataset_folder: str = "datasets/graphs/synthetic/",
        num_nodes: int = 100000,
        adj_size: int = 100000,

):
    list_bet_data = list()
    print("Generating graphs and calculating centralities...")
    for i in range(num_of_graphs):
        print(f"Graph index:{i+1}/{num_of_graphs}",end='\r')
        g_nx = create_graph(graph_type, num_nodes)
        if nx.number_of_isolates(g_nx)>0:
            #print("Graph has isolates.")
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
            g_nx = nx.convert_node_labels_to_integers(g_nx)
        g_nkit = nx2nkit(g_nx)
        bet_dict = cal_exact_bet(g_nkit)
        list_bet_data.append([g_nx,bet_dict])
    raw_data_path = os.path.join(dataset_folder, "raw_data.pickle")

    print(len(list_bet_data))
    with open(raw_data_path,"wb") as fopen:
        pickle.dump(list_bet_data,fopen)
    print(f"\nRaw Graphs saved to {raw_data_path}, Now permutate graphs...")

    #save betweenness split
    get_split(raw_data_path,num_train,num_test,num_copies,adj_size,dataset_folder)
    print(f" Datasets saved to {dataset_folder}/training.pickle and {dataset_folder}/test.pickle.")


def train_gnn(
        dataset_folder: str = "datasets/graphs/synthetic/",
        adj_size: int = 100000,
        num_graph: int = 5,
        hidden: int = 12,
        num_epoch: int = 10,
        model_path: str = "models/model.pt"
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    print(f"Loading data...", flush=True)
    pth_data_path = [os.path.join(dataset_folder, "data_train.pth"), os.path.join(dataset_folder, "data_test.pth") ]
    with open(os.path.join(dataset_folder, "training.pickle"), "rb") as fopen:
        list_graph_train,list_n_seq_train,list_num_node_train,bc_mat_train = pickle.load(fopen)
    with open(os.path.join(dataset_folder, "test.pickle"), "rb") as fopen:
        list_graph_test,list_n_seq_test,list_num_node_test,bc_mat_test = pickle.load(fopen)
    
    if os.path.exists(pth_data_path[0]) and os.path.exists(pth_data_path[1]): 
        # 使用torch.load()加载数据  
        data_train = torch.load(pth_data_path[0])  
        data_test = torch.load(pth_data_path[1])  
        
        # 从字典中提取你的列表  
        list_adj_train = data_train['list_adj_train']  
        list_adj_t_train = data_train['list_adj_t_train']  
        
        list_adj_test = data_test['list_adj_test']  
        list_adj_t_test = data_test['list_adj_t_test']  
    else:
        #Get adjacency matrices from graphs
        print(f"Graphs to adjacency conversion.", flush=True)

        list_adj_train,list_adj_t_train = graph_to_adj_bet(list_graph_train, list_n_seq_train, list_num_node_train, adj_size)
        list_adj_test,list_adj_t_test = graph_to_adj_bet(list_graph_test, list_n_seq_test, list_num_node_test, adj_size)
        data_train = {  
            'list_adj_train': list_adj_train,  
            'list_adj_t_train': list_adj_t_train  
        }  
        data_test = {  
            'list_adj_test': list_adj_test,  
            'list_adj_t_test': list_adj_t_test  
        }  
    
        # save adjacency matrices  
        torch.save(data_train, os.path.join(dataset_folder, 'data_train.pth'))  
        torch.save(data_test, os.path.join(dataset_folder, 'data_test.pth'))
        print(f"Adajacency matricies saved to {os.path.join(dataset_folder, 'data_train.pth')} and {os.path.join(dataset_folder, 'data_test.pth')}")


    # train
    print("Training", flush=True)
    print(f"Total Number of epoches: {num_epoch}", flush=True)
    model = GNN_Bet(ninput=adj_size, nhid=hidden, dropout=0.6)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss = 9999999
    for e in range(num_epoch):
        print(f"Epoch number: {e+1}/{num_epoch}", flush=True)
        train(device, adj_size, list_adj_train,list_adj_t_train,list_num_node_train,bc_mat_train, model, optimizer)

        #to check test loss while training
        with torch.no_grad():
            loss_epoch = test(device, adj_size, list_adj_test,list_adj_t_test,list_num_node_test,bc_mat_test, model, optimizer)
            if loss_epoch < loss:
                loss = loss_epoch 
                torch.save(model, model_path)
                print(f"save the {e+1} epoch model to {model_path}...")



def inference_gnn_based_centrality(
        graph_folder: str = "datasets/graphs/real_world",
        graph_names: List[str] = ["GN.txt"],
        centrality_folder: str = "datasets/centralities/gnn",
        model_path: str = "models/MRL_6layer_1.pt",
        adj_size: int = 100000
):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)    
    for graph_name in graph_names:
        graph_path = concat_path(graph_folder, graph_name)
        graph = preprocess(graph_path)

        
        start = time.perf_counter()
        model = torch.load(model_path)
        list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset([graph], num_copies = 1, adj_size= adj_size)
        list_adj_test, list_adj_t_test = graph_to_adj_bet(list_graph, list_n_sequence, list_node_num, adj_size)
        end = time.perf_counter()
        print(f'The time of load model is {end-start}s')


        start = time.perf_counter()
        _, pre_arrs = inference(model, list_adj_test, list_adj_t_test, list_node_num, cent_mat, adj_size, device)
        end = time.perf_counter()
        print(f'The time of cal centrality is {end-start}s\n\n')

        dest_value_path = concat_path(centrality_folder, get_file_without_extension_name(model_path), get_file_without_extension_name(graph_name) + "_value.txt")
        dest_ranking_path = concat_path(centrality_folder, get_file_without_extension_name(model_path), get_file_without_extension_name(graph_name) + "_ranking.txt")

        dest_dir = os.path.dirname(dest_value_path)
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        centrality_values = []
        centrality_rankings = []
        for index, item  in enumerate(pre_arrs[0]):
            centrality_values.append((index,item))
        centrality_rankings = (sorted(dict(centrality_values).items(), key=lambda item: item[1],reverse=True))

        with open(dest_value_path, 'w') as f:
            for index_centrality_pair in centrality_values:
                f.write(f"{index_centrality_pair[1]} {index_centrality_pair[0]}\n")

        with open(dest_ranking_path, 'w') as f:
            for index_centrality_pair in centrality_rankings:
                f.write(f"{index_centrality_pair[1]} {index_centrality_pair[0]}\n")

def run_2_hop_labeling(
        graph_folder: str = "datasets/graphs/real_world",
        graph_names: Union[str, List[str]] = "GN.txt",  
        centrality_folder: str = "datasets/centralities/gnn/sum_aggre",  
        algorithm_path: str = "pathbee/algorithms/2_hop_labeling.cpp",  
        num_processes: int = 1  
) -> None:  
    """  
    Run 2-hop labeling algorithm.  
  
    graph_paths: Path(s) to graph file(s).  
    centrality_path: Path to centrality file.  
    algorithm_path: Path to 2-hop labeling algorithm.  
    num_processes: (Optional) Number of processes to start. Default is 1.  
    """  
    # Compile the algorithm  
    os.system(f"g++ {algorithm_path} -o 2_hop_labeling")  
    
    if isinstance(graph_names, str):  
        graph_name = graph_names
        cmd = f"./2_hop_labeling {concat_path(graph_folder, graph_name)} {concat_path(centrality_folder, get_file_without_extension_name(graph_name)+ '_ranking.txt')}"  
        execute_command(cmd)
    elif isinstance(graph_names, list):  
        commands = [f"./2_hop_labeling {concat_path(graph_folder, graph_name)} {concat_path(centrality_folder, get_file_without_extension_name(graph_name)+'_ranking.txt')}" for graph_name in graph_names]  
        parallel_process(commands, num_processes)  

command_map = {
    "gen": gen_dataset,
    "train": train_gnn,
    "infer": inference_gnn_based_centrality,
    "indexing": run_2_hop_labeling,
}

def main(command: str = "pll", *args, **kwargs):
    # print(command)
    command_map[command](*args, **kwargs)

if __name__ == "__main__":
    fire.Fire(main)