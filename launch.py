from pathbee.gnn.gen_datasets.create_dataset import *
from pathbee.gnn.gen_datasets.generate_graph import *
from pathbee.gnn.betweenness import *
from pathbee.gnn.predict import preprocess, inference, create_dataset_for_predict
from multiprocessing import Pool, cpu_count

import argparse
import json
from typing import List, Union, Literal
from pathlib import Path
import subprocess

logger = get_logger()

def get_device():
    """Get the appropriate device for the current system."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

def _generate_single_graph(args):
    """Helper function for generating a single graph, used for parallel processing"""
    graph_index, graph_type, num_nodes, seed_offset = args
    
    # Set random seed for current process, ensuring reproducibility
    random.seed(10 + seed_offset + graph_index)
    np.random.seed(10 + seed_offset + graph_index)
    
    try:
        # Generate graph
        g_nx = create_graph(graph_type, num_nodes)
        
        # Handle isolated nodes
        if nx.number_of_isolates(g_nx) > 0:
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
            g_nx = nx.convert_node_labels_to_integers(g_nx)
        
        # Calculate centrality
        g_nkit = nx2nkit(g_nx)
        bet_dict = cal_exact_bet(g_nkit)
        
        return graph_index, [g_nx, bet_dict], None
    except Exception as e:
        return graph_index, None, str(e)

def gen_dataset(
        num_of_graphs: int,
        num_train: int,
        num_test: int,
        num_copies: int,
        graph_type: str,
        dataset_folder: str,
        num_nodes: int,
        adj_size: int,
):
    """Generate dataset for training and testing. Default is to enable parallel optimization"""
    
    # Automatically select the optimal number of processes
    n_processes = min(cpu_count(), max(1, num_of_graphs))
    
    list_bet_data = [None] * num_of_graphs  # Pre-allocate list
    
    if num_of_graphs > 1:
        logger.info(f"Generating graphs and calculating centralities using {n_processes} processes...")
        
        # Prepare parameters
        args_list = [(i, graph_type, num_nodes, 0) for i in range(num_of_graphs)]
        
        # Parallel processing
        with Pool(n_processes) as pool:
            # Use imap_unordered to get real-time progress feedback
            results = pool.imap_unordered(_generate_single_graph, args_list)
            
            completed = 0
            for graph_index, graph_data, error in results:
                completed += 1
                print(f"Graph index:{completed}/{num_of_graphs}", end='\r')
                
                if error:
                    logger.error(f"Error generating graph {graph_index}: {error}")
                    raise RuntimeError(f"Failed to generate graph {graph_index}: {error}")
                
                list_bet_data[graph_index] = graph_data
    else:
        # When there is only one graph, process directly
        logger.info("Generating graphs and calculating centralities...")
        print(f"Graph index:1/{num_of_graphs}", end='\r')
        
        args = (0, graph_type, num_nodes, 0)
        graph_index, graph_data, error = _generate_single_graph(args)
        
        if error:
            logger.error(f"Error generating graph {graph_index}: {error}")
            raise RuntimeError(f"Failed to generate graph {graph_index}: {error}")
        
        list_bet_data[0] = graph_data
    
    # Save raw data
    raw_data_path = os.path.join(dataset_folder, "raw_data.pickle")
    Path(dataset_folder).mkdir(parents=True, exist_ok=True)
    with open(raw_data_path, "wb") as fopen:
        pickle.dump(list_bet_data, fopen)
    
    logger.info(f"\nRaw Graphs saved to {raw_data_path}, Now permutate graphs...")
    
    # Use optimized version of get_split
    get_split(
        raw_data_path,
        num_train,
        num_test,
        num_copies,
        adj_size,
        dataset_folder
    )
    
    logger.info(f"Datasets saved to {dataset_folder}/training.pickle and {dataset_folder}/test.pickle")

def train_gnn(
        dataset_folder: str,
        adj_size: int,
        hidden: int,
        num_epoch: int,
        model_path: str,
):
    """Train GNN model."""
    device = get_device()
    logger.info(f"Using device: {device}")

    
    logger.info("Loading data...")
    pth_data_path = [
        os.path.join(dataset_folder, "data_train.pth"),
        os.path.join(dataset_folder, "data_test.pth")
    ]
    
    # load training.pickle and test.pickle
    with open(os.path.join(dataset_folder, "training.pickle"), "rb") as fopen:
        list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train = pickle.load(fopen)
    with open(os.path.join(dataset_folder, "test.pickle"), "rb") as fopen:
        list_graph_test, list_n_seq_test, list_num_node_test, bc_mat_test = pickle.load(fopen)
        
    # if data_train.pth and data_test.pth exist, skip the following steps
    if os.path.exists(pth_data_path[0]) and os.path.exists(pth_data_path[1]):
        logger.info("Data already exists, skipping conversion to adjacency matrices")
        # load data_train.pth and data_test.pth
        data_train = torch.load(pth_data_path[0])
        data_test = torch.load(pth_data_path[1])
        list_adj_train = data_train['list_adj_train']
        list_adj_t_train = data_train['list_adj_t_train']
        list_adj_test = data_test['list_adj_test']
        list_adj_t_test = data_test['list_adj_t_test']
    
    # check the dim of list_adj_train and list_adj_t_train, if it match the adj_size, skip the following steps
    if list_adj_train[0].shape[0] == adj_size and list_adj_t_train[0].shape[0] == adj_size:
        logger.info("Data already exists, skipping conversion to adjacency matrices")
        pass
    else:
        logger.info("Data does not exist, converting to adjacency matrices")

        # Get adjacency matrices from graphs and save to data_train.pth and data_test.pth
        logger.info("Converting graphs to adjacency matrices")
        list_adj_train, list_adj_t_train = graph_to_adj_bet(
            list_graph_train, list_n_seq_train, list_num_node_train, adj_size
        )
        list_adj_test, list_adj_t_test = graph_to_adj_bet(
            list_graph_test, list_n_seq_test, list_num_node_test, adj_size
        )

        # Save adjacency matrices
        data_train = {
            'list_adj_train': list_adj_train,
            'list_adj_t_train': list_adj_t_train
        }
        data_test = {
            'list_adj_test': list_adj_test,
            'list_adj_t_test': list_adj_t_test
        }

        torch.save(data_train, pth_data_path[0])
        torch.save(data_test, pth_data_path[1])
        logger.info(f"Adjacency matrices saved to {pth_data_path[0]} and {pth_data_path[1]}")

    # Training
    logger.info("Starting training")
    print(f"Total Number of epochs: {num_epoch}")
    model = GNN_Bet(ninput=adj_size, nhid=hidden, dropout=0.6)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss = float('inf')

    for e in range(num_epoch):
        print(f"Epoch number: {e+1}/{num_epoch}")
        train(device, adj_size, list_adj_train, list_adj_t_train,
              list_num_node_train, bc_mat_train, model, optimizer)

        with torch.no_grad():
            loss_epoch = test(device, adj_size, list_adj_test,
                            list_adj_t_test, list_num_node_test, bc_mat_test, model, optimizer)
            if loss_epoch < loss:
                loss = loss_epoch
                torch.save(model, model_path)
                print(f"Saved model from epoch {e+1} to {model_path}")

def inference_gnn(
        graph_path: str,
        save_dir: str,
        model_path: str,
        adj_size: int,
):
    """Run inference using trained GNN model."""
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Model used for inference: {model_path}")


    graph = preprocess(graph_path)

    model = torch.load(model_path)
    list_graph, list_n_sequence, list_node_num, cent_mat = create_dataset_for_predict(
        [graph], num_copies=1, adj_size=adj_size
    )
    list_adj_test, list_adj_t_test = graph_to_adj_bet(
        list_graph, list_n_sequence, list_node_num, adj_size
    )
    _, pre_arrs = inference(
        model, list_adj_test, list_adj_t_test, list_node_num, cent_mat,
        adj_size, device
    )

    dest_value_path = concat_path(save_dir, f"gnn_value.txt")
    dest_ranking_path = concat_path(save_dir, f"gnn_ranking.txt")

    dest_dir = os.path.dirname(dest_value_path)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    centrality_values = [(index, item) for index, item in enumerate(pre_arrs[0])]
    centrality_rankings = sorted(dict(centrality_values).items(), key=lambda item: item[1], reverse=True)

    with open(dest_value_path, 'w') as f:
        for index_centrality_pair in centrality_values:
            f.write(f"{index_centrality_pair[1]} {index_centrality_pair[0]}\n")

    with open(dest_ranking_path, 'w') as f:
        for index_centrality_pair in centrality_rankings:
            f.write(f"{index_centrality_pair[1]} {index_centrality_pair[0]}\n")

def run_2_hop_labeling(
        graph_path: str,
        centrality_path: str,
        algorithm_path: str,
        num_processes: int,
        index_path: str,
):
    """Run 2-hop labeling algorithm."""
    # Compile and run
    execute_command(f"g++ {algorithm_path} -o 2_hop_labeling")
    execute_command("chmod +x 2_hop_labeling")  
    cmd = f"./2_hop_labeling construct {graph_path} {centrality_path} {index_path}"
    parallel_process([cmd], num_processes)
    logger.info("2-hop labeling completed")

def add_query_args(parser):
    parser.add_argument('--index-path', type=str, nargs='+', required=True, help='Path(s) to the index file(s)')
    parser.add_argument('--graph-path', type=str, required=True, help='Path to the graph file')
    parser.add_argument('--num-queries', type=int, default=100000, help='Number of random queries to perform')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--result-dir', type=str, default=None, help='Path to save the CSV files')
    parser.add_argument('--stratified', action='store_true', help='Use stratified sampling (default: random sampling)')

def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='PathBee: Graph Neural Network for Path Queries')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Gen dataset command
    gen_parser = subparsers.add_parser('gen', help='Generate dataset for training and testing')
    gen_parser.add_argument('--num-of-graphs', type=int, default=2,
                          help='Number of graphs to generate')
    gen_parser.add_argument('--num-train', type=int, default=1,
                          help='Number of training graphs')
    gen_parser.add_argument('--num-test', type=int, default=1,
                          help='Number of test graphs')
    gen_parser.add_argument('--num-copies', type=int, default=50,
                          help='Number of copies for each graph')
    gen_parser.add_argument('--graph-type', type=str, default='SF',
                          help='Type of graph to generate')
    gen_parser.add_argument('--dataset-folder', type=str, default='datasets/graphs/synthetic/',
                          help='Folder to save the dataset')
    gen_parser.add_argument('--num-nodes', type=int, default=100000,
                          help='Number of nodes in each graph')
    gen_parser.add_argument('--adj-size', type=int, default=10000000,
                          help='Size of adjacency matrix')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train GNN model')
    train_parser.add_argument('--dataset-folder', type=str, default='datasets/graphs/synthetic/',
                            help='Folder containing the dataset')
    train_parser.add_argument('--adj-size', type=int, default=10000000,
                            help='Size of adjacency matrix')
    train_parser.add_argument('--hidden', type=int, default=12,
                            help='Hidden layer size')
    train_parser.add_argument('--num-epoch', type=int, default=2,
                            help='Number of training epochs')
    train_parser.add_argument('--model-path', type=str, default='models/model.pt',
                            help='Path to save the trained model')

    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference using trained GNN model')
    infer_parser.add_argument('--graph-path', type=str, required=True,
                            help='Path to the input graph file')
    infer_parser.add_argument('--save-dir', type=str, required=True,
                            help='Path to save centrality results')
    infer_parser.add_argument('--model-path', type=str, default='models/MRL_6layer_1.pt',
                            help='Path to the trained model')
    infer_parser.add_argument('--adj-size', type=int, default=10000000,
                            help='Size of adjacency matrix')

    # Indexing command
    indexing_parser = subparsers.add_parser('index', help='Run 2-hop labeling algorithm')
    indexing_parser.add_argument('--graph-path', type=str, required=True,
                               help='Path to the input graph file')
    indexing_parser.add_argument('--centrality-path', type=str, required=True,
                               help='Path to the centrality ranking file')
    indexing_parser.add_argument('--algorithm-path', type=str, default='pathbee/algorithms/2_hop_labeling.cpp',
                               help='Path to the 2-hop labeling algorithm source code')
    indexing_parser.add_argument('--num-processes', type=int, default=1,
                               help='Number of parallel processes to use')
    indexing_parser.add_argument('--index-path', type=str, required=True,
                               help='Path to the index file')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query distance using constructed index')
    add_query_args(query_parser)

    # Centrality command
    cen_parser = subparsers.add_parser('cen', help='Calculate centrality of a graph')
    cen_parser.add_argument('--graph-path', type=str, required=True, help='Path to the input graph file')
    cen_parser.add_argument('--save-dir', type=str, required=True, help='Directory to save centrality results')
    cen_parser.add_argument('--centrality', type=str, nargs='+', default=['bc'],
                           help='Centrality type(s) to calculate (dc, bc, gs, kadabra, close, eigen)')
    cen_parser.add_argument('--force', action='store_true', help='Force recalculation even if results exist')
    cen_parser.add_argument('--python-path', type=str, default='python3', help='Python interpreter to use')
    cen_parser.add_argument('--script-path', type=str, default='datasets/cal_centrality.py', help='Path to cal_centrality.py')

    return parser

def main():
    parser = setup_parser()
    args = parser.parse_args()

    if args.command == 'gen':
        gen_dataset(
            num_of_graphs=args.num_of_graphs,
            num_train=args.num_train,
            num_test=args.num_test,
            num_copies=args.num_copies,
            graph_type=args.graph_type,
            dataset_folder=args.dataset_folder,
            num_nodes=args.num_nodes,
            adj_size=args.adj_size
        )
    elif args.command == 'train':
        train_gnn(
            dataset_folder=args.dataset_folder,
            adj_size=args.adj_size,
            hidden=args.hidden,
            num_epoch=args.num_epoch,
            model_path=args.model_path
        )
    elif args.command == 'infer':
        inference_gnn(
            graph_path=args.graph_path,
            save_dir=args.save_dir,
            model_path=args.model_path,
            adj_size=args.adj_size
        )
    elif args.command == 'index':
        run_2_hop_labeling(
            graph_path=args.graph_path,
            centrality_path=args.centrality_path,
            algorithm_path=args.algorithm_path,
            num_processes=args.num_processes,
            index_path=args.index_path
        )
    elif args.command == 'query':
        from scripts.query_distribution import generate_query_csv
        generate_query_csv(
            args.index_path,
            args.graph_path,
            args.num_queries,
            args.seed,
            args.result_dir,
            args.stratified
        )
        from scripts.query_distribution import plot_query_time_distribution
        plot_query_time_distribution(
            result_dir=args.result_dir,
            stratified=args.stratified
        )

    elif args.command == 'cen':
        # Call cal_centrality.py with the provided arguments
        cen_types = ' '.join([f'--centrality {c}' for c in args.centrality])
        force_flag = '--force' if args.force else ''
        cmd = (f"{args.python_path} {args.script_path} "
               f"--graph-path {args.graph_path} "
               f"--save-dir {args.save_dir} "
               f"{cen_types} {force_flag}")
        print(f"Running: {cmd}")
        os.system(cmd)  
    else:
        parser.print_help()

if __name__ == "__main__":
    logger.info("START EXECUTE CODE************************")
    main()
    logger.info("FINISH EXECUTE CODE************************\n\n\n")
