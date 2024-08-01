import cmath
import numpy as np
import matplotlib.pyplot as plt
from sympy import primerange
import random
import networkx as nx
from tqdm import tqdm
import pickle
from scipy.stats import wasserstein_distance as w1_dist
import scipy as sp
from multiprocessing import Pool
from dist_utils import *
from lanczos import * 

def eig_normAdj(G):
    #Returns normalized adjacency eigenvalues
    return np.ones(nx.number_of_nodes(G)) - nx.normalized_laplacian_spectrum(G).real

def get_normAdj(G):
    # Returns the eigenvalues of the normalized adjacency matrix
    L = nx.normalized_laplacian_matrix(G).toarray()
    I = np.eye(G.number_of_nodes())
    A = I - L
    return A
    
def replacement_prod(G,l):
    adj_list_G = {}
    for vtx in G.nodes():
        neighbour_list = list(G.neighbors(vtx))
        adj_list_G[vtx] = neighbour_list
    def find_vtx_index(vtx_u, vtx_v):
        for idx in range(len(adj_list_G[vtx_u])):
            if adj_list_G[vtx_u][idx] == vtx_v:
                return idx
    G_new = nx.Graph()
    for vtx in G.nodes():
        for j in range(l):
            G_new.add_node((vtx,j))
    for vtx in G.nodes():
        for j in range(l):
            j_th_neighbour = adj_list_G[vtx][j]
            neighbour_no_of_vtx = find_vtx_index(j_th_neighbour, vtx)
            G_new.add_edge((vtx,j),(j_th_neighbour,neighbour_no_of_vtx))
    for vtx in G.nodes():
        for j in range(l):
            if j == l-1:
                k = 0
                G_new.add_edge((vtx,j),(vtx,k))
            else:
                G_new.add_edge((vtx,j),(vtx,j+1))
    return G_new


def compute_dist(): 
    
    l = []
    for i in range(5,21,2):
        l.append(i)
    w1_lst = []
    for i in l:
        n = int(np.ceil(np.log(i) * 2 ** i))
        if n % 2 == 1: 
            n = n + 1
        G1 = nx.random_regular_graph(i,n)
        G1_r = replacement_prod(G1, i)
        G2 = nx.random_regular_graph(2*i,n)
        G2_r = replacement_prod(G2, 2*i)
        # eig_g1r = eig_normAdj(G1_r)
        # eig_g2r = eig_normAdj(G2_r)
        print("generating adjacency matrix...")
        AG1_r = get_normAdj(G1_r)
        AG2_r = get_normAdj(G2_r)
        print("running lanczos iterations...")
        Q1,(a1,b1) = exact_lanczos(AG1_r)
        Q2,(a2,b2) = exact_lanczos(AG2_r)
        print("averaging distributions; spawning child processes...")
        reps = 1000
        args_list_1 = [(a1, b1) for _ in range(reps)]
        args_list_2 = [(a2, b2) for _ in range(reps)]
        # Using multiprocessing to run get_GQ_distr using starmap
        with Pool(5) as pool:
            D1s = pool.starmap(get_GQ_distr, args_list_1)
            D2s = pool.starmap(get_GQ_distr, args_list_2)
        D1 = get_ave_distr(D1s)
        D2 = get_ave_distr(D2s)
        print("computing w1 distance...")
        w1 = d_W(D1, D2)
        print("l = " + str(i) + "; n = " + str(n) + "; w1 = " + str(w1))
        w1_lst.append(w1)
    return w1_lst

w1_lst = compute_dist()

# Define the file path
file_path = 'w1_lst.pkl'
# Save w1_lst as a pickle file
with open(file_path, 'wb') as file:
    pickle.dump(w1_lst, file)