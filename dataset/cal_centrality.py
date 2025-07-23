import networkit as nk
import networkx as nx
import sys
import os 
import time
import argparse
from typing import Tuple, List, Dict, Any, Callable

def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Calculate various centrality measures for graphs')
    parser.add_argument('--centrality', '-c', type=str, nargs='+', default=['bc'],
                      choices=['dc', 'bc', 'gs', 'rk', 'kadabra', 'close', 'eigen'],
                      help='Type of centrality to calculate (dc=degree, bc=betweenness, etc.). Can specify multiple.')
    parser.add_argument('--graph-path', '-g', type=str, required=True,
                      help='Path to the graph file')
    parser.add_argument('--save-dir', '-s', type=str, required=True,
                      help='Directory to save centrality results')
    parser.add_argument('--force', '-f', action='store_true',
                      help='Force recalculation even if results exist')
    return parser

# Loads a graph from a text file.
def read_graph(map_path: str) -> Tuple[nx.DiGraph, nk.Graph]:
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"Graph file not found: {map_path}")
        
    g_nx = nx.DiGraph()
    try:
        with open(map_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    src, dest = int(parts[0]), int(parts[1])
                    # Handle weighted graphs
                    if len(parts) >= 3:
                        weight = float(parts[2])
                        g_nx.add_edge(src, dest, weight=weight)
                    else:
                        g_nx.add_edge(src, dest)
    except Exception as e:
        raise RuntimeError(f"Error reading graph file {map_path}: {str(e)}")
        
    g_nk = nx2nkit(g_nx)
    print(f"Map {map_path} has {g_nx.number_of_nodes()} nodes and {g_nx.number_of_edges()} edges.")
    return g_nx, g_nk

# Converts a NetworkX graph to a NetworKit graph
def nx2nkit(g_nx):
    node_num = g_nx.number_of_nodes()
    g_nkit = nk.Graph(directed=True, weighted=True)

    for i in range(node_num):
        g_nkit.addNode()

    for e1, e2, data in g_nx.edges(data=True):
        weight = data.get('weight', 1.0)  # Default weight is 1.0 for unweighted edges
        g_nkit.addEdge(e1, e2, weight)   
    return g_nkit

# Returns all file names in the given directory.
def get_all_file_names(folder_path):
    file_names = os.listdir(folder_path)
    return file_names

# This function calculates the centrality of a graph using the provided function
def calculate_centrality(g_nx, g_nk, centrality_func):
    if centrality_func in (degree_centrality, eigenvector_centrality):
        return centrality_func(g_nx)
    else:
        return centrality_func(g_nk)

# Calculates the degree centrality of a graph
def degree_centrality(g_nx):
    temp = nx.degree_centrality(g_nx)  # Calculate degree centrality
    degree_value = [(node, node_centrality) for node, node_centrality in temp.items()]
    degree_ranking = sorted(temp.items(), key=lambda x: x[1], reverse=True)  
    return degree_value, degree_ranking

# Calculates the eigenvector centrality of a graph
def eigenvector_centrality(g_nx):
    temp = nx.eigenvector_centrality(g_nx)  # Calculate eigenvector centrality
    Eigen_value = [(node, node_centrality) for node, node_centrality in temp.items()]
    Eigen_ranking = sorted(temp.items(), key=lambda x: x[1], reverse=True)  # Sort the results
    return Eigen_value, Eigen_ranking

# Calculates the betweenness centrality of a graph
def betweenness_centrality(g_nk):
    temp = nk.centrality.Betweenness(g_nk, normalized=True).run()  # Calculate betweenness centrality
    BC_value = list(enumerate(temp.scores()))
    BC_ranking = temp.ranking() 
    return BC_value, BC_ranking

# Calculates the GS betweenness centrality of a graph
def gs_betweenness_centrality(g_nk):
    temp = nk.centrality.EstimateBetweenness(g_nk, nSamples=8192).run()  # Calculate GS betweenness centrality
    GS_BC_value = list(enumerate(temp.scores()))
    GS_BC_ranking = temp.ranking()  
    return GS_BC_value, GS_BC_ranking

def rk_betweenness_centrality(g_nk):
    temp = nk.centrality.ApproxBetweenness(g_nk).run()  # Calculate Rk betweenness centrality
    RK_BC_value = list(enumerate(temp.scores()))
    RK_BC_ranking = temp.ranking()  
    return RK_BC_value, RK_BC_ranking

# Calculates the Kadabra betweenness centrality of a graph
def kadabra_betweenness_centrality(g_nk):
    temp = nk.centrality.KadabraBetweenness(g_nk, err=0.01, delta=0.1).run()  # Calculate Kadabra betweenness centrality
    Kadabra_BC_value = list(enumerate(temp.scores()))
    Kadabra_BC_ranking = temp.ranking()  
    return Kadabra_BC_value, Kadabra_BC_ranking

# Calculates the closeness centrality of a graph
def closeness_centrality(g_nk):
    temp = nk.centrality.Closeness(g_nk, False, nk.centrality.ClosenessVariant.Generalized).run()  # Calculate closeness centrality
    Close_value = list(enumerate(temp.scores()))
    Close_ranking = temp.ranking()  
    return Close_value, Close_ranking

def save_centrality(centrality_value: List[Tuple], centrality_ranking: List[Tuple], save_path: str) -> None:
    try:
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok=True)
        
        value_path = save_path[:-4] + "_value.txt"
        ranking_path = save_path[:-4] + "_ranking.txt"
        
        with open(value_path, 'w') as f:
            for item in centrality_value:
                f.write(f"{item[1]} {item[0]}\n")
                
        with open(ranking_path, 'w') as f:
            for item in centrality_ranking:
                f.write(f"{item[1]} {item[0]}\n")
    except Exception as e:
        raise RuntimeError(f"Error saving centrality results: {str(e)}")

# Create a dictionary mapping strings to functions
centrality_dict = {
    'dc': degree_centrality,  # Degree centrality
    'bc': betweenness_centrality,  # Betweenness centrality
    'gs': gs_betweenness_centrality,  # GS Betweenness centrality
    'rk': rk_betweenness_centrality,  # Rk Betweenness centrality
    'kadabra': kadabra_betweenness_centrality,  # Kadabra Betweenness centrality
    'close': closeness_centrality,  # Closeness centrality
    'eigen': eigenvector_centrality  # Eigenvector centrality
}

def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.exists(args.graph_path):
        raise FileNotFoundError(f"Graph file not found: {args.graph_path}")
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    try:
        print(f"Processing graph: {args.graph_path}", flush=True)
        g_nx, g_nk = read_graph(args.graph_path)
        
        # Process each centrality type
        for centrality_type in args.centrality:
            # Generate save paths - only save ranking file with centrality name
            ranking_path = os.path.join(args.save_dir, f"{centrality_type}.txt")
            
            # Skip if results exist and force flag is not set
            if not args.force and os.path.exists(ranking_path):
                print(f"Results for {centrality_type} already exist. Use --force to recalculate.")
                continue
                
            print(f"Calculating {centrality_type} centrality...", flush=True)
            centrality_value, centrality_ranking = calculate_centrality(
                g_nx=g_nx,
                g_nk=g_nk,
                centrality_func=centrality_dict[centrality_type]
            )
            
            # Save only ranking file
            with open(ranking_path, 'w') as f:
                for item in centrality_ranking:
                    f.write(f"{item[1]} {item[0]}\n")
                    
            print(f"Completed {centrality_type} centrality calculation")
            
        print("All calculations completed successfully")
    except Exception as e:
        print(f"Error processing graph: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
    