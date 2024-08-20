from math import inf
import random
import time
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
from torch_geometric.utils import (negative_sampling, add_self_loops, to_networkx, degree)
import torch_sparse 
from tqdm import tqdm
import scipy.sparse as ssp
from scipy.sparse import csr_matrix
import pickle
from collections import Counter, defaultdict
from data_analysis.function.homophily import * 
import torch_sparse
import scipy.sparse as sp

def get_path_score(args, edge_index, links, predefine_group="D2AD2"):
    # GNN can be viewed as counting the similar paths
    # for the last hop, we will remove the redudant, may not do direct calculation
    device = links.device
    # edge_index = edge_index.T
    edge_index, edge_value = get_adjacent_matrix(edge_index, predefine_group)
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    edge_index, edge_value = edge_index.to(device), edge_value.to(device)
        
    adj = torch.sparse_coo_tensor(edge_index, edge_value, torch.Size([num_nodes, num_nodes]))
    adj = adj.coalesce()
    adj_last = adj.clone()
    
    for i in tqdm(range(args.num_hops - 1)):
        torch.cuda.empty_cache()
        adj_last = torch.sparse.mm(adj_last, adj)
        adj_last = adj_last.coalesce()
    
    indices, values = adj_last.indices(), adj_last.values()
    diagnoal_tensor = torch.zeros(num_nodes).to(device)
    self_indices_mask = (indices[0, :] == indices[1, :])
    self_values = values[self_indices_mask]
    self_indice_value = indices[0, :][self_indices_mask]
    diagnoal_tensor[self_indice_value] = self_values
    src_scores, tgt_scores = diagnoal_tensor[links[:, 0]], diagnoal_tensor[links[:, 1]]
    
    link_scores = torch.abs(src_scores - tgt_scores)

    return link_scores

    '''
    # another version without the final hop information for saving memory (Not ready yet)
    for i in tqdm(range(num_hops - 2)): 
        torch.cuda.empty_cache()
        adj_last = torch.sparse.mm(adj_last, adj)
        adj_last = adj_last.coalesce()
    graph = nx.Graph()
    graph.add_edges_from(sparse_tensor.cpu().t().tolist())
    
    link_scores = np.zeros([np.max(links.shape)])
    for idx, link in enumerate(links):
        src, tgt = link[0], link[1]
        src_neighbors, tgt_neighbors = graph.neighbors(src), graph.neighbors(tgt)
        common_neighbors = list(src_neighbors & tgt_neighbors)
        for neighbor in src_neighbors:
            link_scores[idx] += graph.edges[(src, neighbor)]['weight'] * graph[(neighbor, tgt)]['weight']
    
    link_scores = np.abs(link_scores)
    '''         
    

def get_path_score2(args, edge_index, links, predefine_group="D2AD2"):
    # GNN can be viewed as counting the similar paths
    # for the last hop, we will remove the redudant, may not do direct calculation
    device = links.device
    # edge_index = edge_index.T
    edge_index, edge_value = get_adjacent_matrix(edge_index, predefine_group)
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    edge_index, edge_value = edge_index.to(device), edge_value.to(device)
        
    adj = torch.sparse_coo_tensor(edge_index, edge_value, torch.Size([num_nodes, num_nodes]))
    adj = adj.coalesce()
    adj_last = adj.clone()
    
    for i in tqdm(range(args.num_hops - 1)):
        torch.cuda.empty_cache()
        adj_last = torch.sparse.mm(adj_last, adj)
        adj_last = adj_last.coalesce()
    
    indices = adj_last.indices().cpu().numpy()
    adj_last = sp.coo_matrix((adj_last.values().cpu().numpy(), (indices[0], indices[1])), shape=(num_nodes, num_nodes)).tocsr()
    links = links.cpu().numpy()
    src, tgt = links[:, 0], links[:, 1]
    # import ipdb; ipdb.set_trace()    
    link_scores = np.squeeze(np.abs(adj_last[src, src] - adj_last[tgt, tgt]))
    link_scores = np.max(link_scores) - link_scores
    # reverse the order, the smaller the better 
    link_scores = torch.tensor(link_scores)

    return link_scores




def structure_role(edge_index):
    pass


def degree(edge_index):
    # TODO: check whether we need to divide by 2
    num_nodes = torch.max(edge_index).item() + 1
    return degree(edge_index, num_nodes)




# TODO: Katz centrality, but they are similar
def eigencentrality(edge_index, max_iter=100):
    # TODO: if there are different groups we can utilize this function
    # L, U = torch.lobpcg(M, k=k, largest=True, method='ortho')
    # Create a sparse matrix
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    edge_weight = edge_index
    edge_weight = torch.ones(num_edges).to(torch.float32)   #.to(torch.float)
    edge_index, edge_weight = edge_index.cuda(), edge_weight.cuda()
    
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]))
    adj = adj.coalesce() 
    
    # Power iteration method to find the eigenvector
    max_iter = 100  # Maximum number of iterations
    tolerance = 1e-4  # Convergence tolerance

    # Initialize a random vector
    v = torch.randn(num_nodes, dtype=torch.float32)

    for i in range(max_iter):
        v_prev = v.clone()
        matrix_v = matrix @ v
        v = matrix_v / torch.norm(matrix_v)

        # Check for convergence
        if torch.norm(v - v_prev) < tolerance:
            break

    # Get the eigenvector corresponding to the largest eigenvalue
    largest_eigenvector = v

    return largest_eigenvector
 