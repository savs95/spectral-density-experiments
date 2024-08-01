import cmath
import numpy as np
import networkx as nx
from tqdm import tqdm
import scipy as sp
from multiprocessing import Pool
from scipy.stats import wasserstein_distance as w1_dist


def eig_normAdj(G):
    #Returns normalized adjacency eigenvalues
    return np.ones(nx.number_of_nodes(G)) - nx.normalized_laplacian_spectrum(G)

def find_vtx_index(vtx_u, vtx_v, adj_list_G):
    # Finds the index of vtx_v in the adjacency list of vtx_u
    for idx in range(len(adj_list_G[vtx_u])):
        if adj_list_G[vtx_u][idx] == vtx_v:
            return idx
def replacement_prod(G,l):
    adj_list_G = {}
    # Create a dictionary of neighbour lists
    for vtx in G.nodes():
        neighbour_list = list(G.neighbors(vtx))
        adj_list_G[vtx] = neighbour_list
    # G_rep: Initialize the replacement product graph
    G_rep = nx.Graph()
    # Create the nodes of the replacement product graph 
    # Indexed by the original vertex and the index of the neighbour
    for vtx in G.nodes():
        for j in range(l):
            G_rep.add_node((vtx,j))
    # Create the edges of the replacement product graph from the original graph G
    for vtx in G.nodes():
        for j in range(l):
            j_th_neighbour = adj_list_G[vtx][j]
            idx_of_vtx = find_vtx_index(j_th_neighbour, vtx, adj_list_G)
            G_rep.add_edge((vtx,j),(j_th_neighbour,idx_of_vtx))
    # Create the edges to the cycle of the replacement product graph
    for vtx in G.nodes():
        for j in range(l):
            if j == l-1:
                k = 0
                G_rep.add_edge((vtx,j),(vtx,k))
            else:
                G_rep.add_edge((vtx,j),(vtx,j+1))
    return G_rep

def compute_dist(index, n):
    #Write the value of l here
    l = 7
    G1 = nx.random_regular_graph(l,n)
    G1_r = replacement_prod(G1, l)
    G2 = nx.random_regular_graph(2*l,n)
    G2_r = replacement_prod(G2, 2*l)
    eig_g1 = eig_normAdj(G1_r)
    eig_g2 = eig_normAdj(G2_r)
    w1 = w1_dist(eig_g1, eig_g2)
    print("n = " + str(n), "w1 = " + str(w1))
    return (n, w1)
