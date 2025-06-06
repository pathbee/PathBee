from pathbee.gnn.gen_datasets.create_dataset import *
from pathbee.gnn.gen_datasets.generate_graph import *
from pathbee.gnn.betweenness import *
from pathbee.gnn.predict import *

import fire
import json
from typing import List, Union, Literal
from pathlib import Path

logger = get_logger()

def load_config(config_path: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

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
    """Generate dataset for training and testing."""
    list_bet_data = list()
    logger.info("Generating graphs and calculating centralities...")
    
    for i in range(num_of_graphs):
        print(f"Graph index:{i+1}/{num_of_graphs}", end='\r')
        g_nx = create_graph(graph_type, num_nodes)
        
        if nx.number_of_isolates(g_nx) > 0:
            g_nx.remove_nodes_from(list(nx.isolates(g_nx)))
            g_nx = nx.convert_node_labels_to_integers(g_nx)
            
        g_nkit = nx2nkit(g_nx)
        bet_dict = cal_exact_bet(g_nkit)
        list_bet_data.append([g_nx, bet_dict])

    raw_data_path = os.path.join(dataset_folder, "raw_data.pickle")
    Path(dataset_folder).mkdir(parents=True, exist_ok=True)

    with open(raw_data_path, "wb") as fopen:
        pickle.dump(list_bet_data, fopen)
    logger.info(f"\nRaw Graphs saved to {raw_data_path}, Now permutate graphs...")

    get_split(
        raw_data_path,
        num_train,
        num_test,
        num_copies,
        adj_size,
        dataset_folder
    )
    logger.info(f"Datasets saved to {dataset_folder}/training.pickle and {dataset_folder}/test.pickle.")

def get_device():
    """Get the appropriate device for the current system."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")

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
    
    with open(os.path.join(dataset_folder, "training.pickle"), "rb") as fopen:
        list_graph_train, list_n_seq_train, list_num_node_train, bc_mat_train = pickle.load(fopen)
    with open(os.path.join(dataset_folder, "test.pickle"), "rb") as fopen:
        list_graph_test, list_n_seq_test, list_num_node_test, bc_mat_test = pickle.load(fopen)

    # Get adjacency matrices from graphs
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

def inference_gnn_based_centrality(
        graph_folder: str,
        graph_names: List[str],
        centrality_folder: str,
        model_path: str,
        adj_size: int,
):
    """Run inference using trained GNN model."""
    device = get_device()
    logger.info(f"Using device: {device}")
    logger.info(f"Model used for inference: {model_path}")

    for graph_name in graph_names:
        graph_path = concat_path(graph_folder, graph_name)
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

        base_name = get_file_without_extension_name(graph_name)
        dest_value_path = concat_path(centrality_folder, f"{base_name}_value.txt")
        dest_ranking_path = concat_path(centrality_folder, f"{base_name}_ranking.txt")

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
        graph_folder: str,
        graph_names: Union[str, List[str]],
        centrality_folder: str,
        centrality_type: Literal['dc', 'bc', 'gnn'],
        algorithm_path: str,
        num_processes: int,
):
    """Run 2-hop labeling algorithm."""
    # Compile the algorithm
    execute_command(f"g++ {algorithm_path} -o 2_hop_labeling")

    if isinstance(graph_names, str):
        graph_names = [graph_names]

    commands = []
    for graph_name in graph_names:
        base_name = get_file_without_extension_name(graph_name)
        cmd = f"./2_hop_labeling {concat_path(graph_folder, graph_name)} {concat_path(centrality_folder, centrality_type, f'{base_name}_ranking.txt')}"
        commands.append(cmd)

    parallel_process(commands, num_processes)
    logger.info("2-hop labeling completed")

command_map = {
    "gen": gen_dataset,
    "train": train_gnn,
    "infer": inference_gnn_based_centrality,
    "indexing": run_2_hop_labeling,
}

def main(command: str = "pll", config_path: str = "config.json"):
    """Main entry point for the application."""
    if command not in command_map:
        raise ValueError(f"Unknown command: {command}. Available commands: {list(command_map.keys())}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Call the appropriate function with its parameters
    if command == "gen":
        gen_dataset(**config['gen'])
    elif command == "train":
        train_gnn(**config['train'])
    elif command == "infer":
        inference_gnn_based_centrality(**config['infer'])
    elif command == "indexing":
        run_2_hop_labeling(**config['indexing'])

if __name__ == "__main__":
    logger.info("START EXECUTE CODE************************")
    fire.Fire(main)
    logger.info("FINISH EXECUTE CODE************************\n\n\n")
