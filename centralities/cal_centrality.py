import networkit as nk
import networkx as nx
import sys
import os 
import time
import argparse

# Allows the user to specify the type of centrality calculation ('bc', 'dc', etc.) from the command line.
parser = argparse.ArgumentParser()
parser.add_argument('-centrality', type=str, default='bc')
args = parser.parse_args()

# Loads a graph from a text file.
def read_graph(map_path):
    g_nx = nx.DiGraph()
    f = open(map_path, 'r')
    data = f.readlines()
    f.close()
    for idx, lines in enumerate(data):
        src, dest = lines.split()
        g_nx.add_edge(int(src), int(dest))
    g_nk = nx2nkit(g_nx)
    print(f"map {map_path} has {g_nx.number_of_nodes()} nodes and {g_nx.number_of_edges()} edges.")
    return g_nx, g_nk

# Converts a NetworkX graph to a NetworKit graph
def nx2nkit(g_nx):
    node_num = g_nx.number_of_nodes()
    g_nkit = nk.Graph(directed=True)

    for i in range(node_num):
        g_nkit.addNode()

    for e1,e2 in g_nx.edges():
        g_nkit.addEdge(e1,e2)   
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


    
def save_centrality(centrality_value, centrality_ranking, save_path):
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    file_path = os.path.join(save_path)
    with open(save_path[:-4] + "_value.txt", 'w') as f:
        for item in centrality_value:
            f.write(str(item[1]) + " " + str(item[0]) + "\n")
    with open(save_path[:-4] + "_ranking.txt", 'w') as f:
        for item in centrality_ranking:
            f.write(str(item[1]) + " " + str(item[0]) + "\n")

# Create a dictionary mapping strings to functions
centrality_dict = {
    'dc': degree_centrality,  # Degree centrality
    'bc': betweenness_centrality,  # Betweenness centrality
    'gs': gs_betweenness_centrality,  # GS Betweenness centrality
    'kadabra': kadabra_betweenness_centrality,  # Kadabra Betweenness centrality
    'close': closeness_centrality,  # Closeness centrality
    'eigen': eigenvector_centrality  # Eigenvector centrality
}

        
graph_folder = "graphs/real_world"
centrality_folder = 'centralities'
centrality_type = args.centrality

if __name__ == "__main__":  
    graphs = get_all_file_names(graph_folder)
    for graph in graphs:
        
        save_path = os.path.join(centrality_folder, centrality_type, graph)
        if os.path.exists(save_path[:-4] + "_value.txt") and os.path.exists(save_path[:-4] + "_ranking.txt"):
            print(f"{graph} has been calculated before.")
        else:
            print(f"run {graph}", flush= True)
            g_nx, g_nk = read_graph(f"{graph_folder}/{graph}")
            centrality_value, centrality_ranking = calculate_centrality(g_nx= g_nx, g_nk= g_nk, centrality_func= centrality_dict[centrality_type])
            save_centrality(centrality_value = centrality_value, centrality_ranking= centrality_ranking, save_path= save_path)
    