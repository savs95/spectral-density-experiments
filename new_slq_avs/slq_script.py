import cmath
import numpy as np
import networkx as nx
from tqdm import tqdm
import pickle
import scipy as sp
from multiprocessing import Pool
import parallel_prod
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Lasso
import math
from multiprocessing import Pool
import multiprocessing as mp
from scipy.sparse import eye, csr_matrix


def eig_normAdj(G):
    #Returns normalized adjacency eigenvalues
    return np.ones(nx.number_of_nodes(G)) - nx.normalized_laplacian_spectrum(G).real

def get_normAdj(G):
    # Calculate the normalized Laplacian matrix (already in sparse format)
    norm_laplacian = nx.normalized_laplacian_matrix(G)
    # Identity matrix in sparse format, directly created as sparse for efficiency
    I = eye(G.number_of_nodes(), format='csr')
    # Subtract the normalized Laplacian matrix from the identity matrix
    sparse_norm_adj_matrix = I - norm_laplacian
    return sparse_norm_adj_matrix

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

def set_param(matrix_size, eps, failure_prob):
    #According to theorem 1 of the paper
    #Setting n_v to be just max 10 for now
    n_v = (4/(matrix_size+2))*(eps**(-2))*np.log(2*matrix_size/failure_prob)
    if n_v > 10:
        n_v = 10
    else:
        n_v = int(np.ceil(n_v))
    k = 12/eps + 0.5
    if k > matrix_size:
        k = matrix_size
    return n_v, int(np.ceil(k))

if __name__ == "__main__":
    data_list = []
    for l in range(3,100,2):
            nodes_init = int(np.ceil(np.log(l) * 2 ** l))
            n = nodes_init * l
            if nodes_init % 2 == 1: 
                    nodes_init = nodes_init + 1
            print(nodes_init)
            print("\t Starting Graph Creation")        
            G2 = nx.random_regular_graph(2*l,nodes_init)
            G2_r = replacement_prod(G2, 2*l)
            AG2_r = get_normAdj(G2_r)
                    
            G1 = nx.random_regular_graph(l,nodes_init)
            G1_r = replacement_prod(G1, l)
            AG1_r = get_normAdj(G1_r)   
            
            print("\t Graphs Created")     
            print("\t graph created")
            error = 1/(2*l**3)
            nv1, k1 = set_param(n, error, 0.001)
            nv2, k2 = set_param(n, error, 0.001)
            args_list_1 = [(AG1_r, k1) for _ in range(nv1)]
            args_list_2 = [(AG2_r, k2) for _ in range(nv2)]
            with Pool(10) as p:
                D1s = p.starmap(parallel_prod.lanczos_GQ, args_list_1)
                D2s = p.starmap(parallel_prod.lanczos_GQ, args_list_2)
            D1 = parallel_prod.get_ave_distr(D1s)
            D2 = parallel_prod.get_ave_distr(D2s)
            print("\t computing w1 distance...")
            w1 = parallel_prod.d_W(D1, D2)
            print("\t l = " + str(l) + "; n = " + str(n) + "; w1 = " + str(w1))
            data_list.append({'l': l, 'w1': w1})
            # Save l and w1 to a pickle file
            with open('data.pickle', 'wb') as f:
                pickle.dump(data_list, f)