import networkx as nx
from networkit import *
import numpy as np
np.random.seed(1)


def create_graph(graph_type, num_nodes):

    # num_nodes = np.random.randint(100000,100001)

    if graph_type == " ER":
        #Erdos-Renyi random graphs
        pass

    if graph_type == "SF":
        #Scalefree graphs
        alpha = np.random.randint(40,60)*0.01
        gamma = 0.05
        beta = 1 - alpha - gamma
        g_nx = nx.scale_free_graph(num_nodes,alpha = alpha,beta = beta,gamma = gamma)
        return g_nx

    if graph_type == "GRP":
        #Gaussian-Random Partition Graphs
        pass

        


