"""
A selection of heuristic methods (Personalized PageRank, Adamic Adar and Common Neighbours) for link prediction
"""

import numpy as np
from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import mask_to_index, to_undirected, to_dense_adj, add_remaining_self_loops, dense_to_sparse, to_scipy_sparse_matrix, remove_self_loops, add_self_loops, k_hop_subgraph, contains_self_loops
from torch_scatter import scatter_add
from scipy.sparse import coo_matrix
import numpy as np
import torch.nn.functional as F
import dgl
import torch_geometric

from torch_geometric.utils import coalesce, to_undirected
from ogb.linkproppred import Evaluator
from evaluation_new import get_metric_score
import pickle
import networkx as nx
from sklearn.linear_model import LogisticRegression
from time import time


@torch.no_grad()
def CN(A, edge_index, batch_size=100000):
    """
    Common neighbours
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index
    :param batch_size: int
    :return: FloatTensor [edges] of scores, pyg edge_index
    """
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Common Neighbours for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index

@torch.no_grad()
def CN_new(A1, A2, edge_index, batch_size=100000):
    """
    Common neighbours
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index
    :param batch_size: int
    :return: FloatTensor [edges] of scores, pyg edge_index
    """
    edge_index = edge_index.cpu()
    # import ipdb; ipdb.set_trace()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A1[src].multiply(A2[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Common Neighbours for {len(scores)} edges')
    return scores, edge_index


@torch.no_grad()
def generalized_CN(args, num_hop1, num_hop2, batch_size, dataset, path, know_links, pos_edges, neg_edges, adjs=None):
    '''
    CN include CN with differnt distance
    '''
    edge_index = dataset.data.edge_index
    device = edge_index.device
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    
    if adjs == None:
        with open(f"{path}_adj.txt", 'rb') as f:
            adjs = pickle.load(f)
        with open(f"{path}_dis.txt", 'rb') as f:
            adjs_indices = pickle.load(f)
    num_hops = len(adjs_indices)
    adj_graphs = get_adj_indices_with_value(adjs, adjs_indices, device)
    num_pos_edges = np.max(pos_edges.shape)
    num_neg_edges = np.max(neg_edges.shape)
    edges = torch.cat([pos_edges, neg_edges], dim=0)
    edge_loader = DataLoader(range(np.max(edges.shape)), batch_size, shuffle=False)
    
    preds = torch.zeros([num_pos_edges + num_neg_edges]).to(device)
    for batch_edge_idx in tqdm(edge_loader):
        batch_edges = edges[batch_edge_idx]
        rows, cols = batch_edges[:, 0], batch_edges[:, 1]
        adj_graph1 = adj_graphs[num_hop1]
        adj_graph2 = adj_graphs[num_hop2]
        
        # adj1_value = subgraph(rows, adj_graph1.indices(), adj_graph1.values(), relabel_nodes=False, num_nodes=num_nodes)
        adj_graph1_index, adj_graph1_value = subgraph(rows, adj_graph1.indices(), adj_graph1.values(), relabel_nodes=False, num_nodes=num_nodes)   
        adj_graph2_index, adj_graph2_value = subgraph(cols, adj_graph2.indices(), adj_graph2.values(), relabel_nodes=False, num_nodes=num_nodes)   

        if args.algorithm == "CN" or args.algorithm == "Jaccard":
            result_index, result_value = torch_sparse.spspmm(adj_graph1_index, adj_graph1_value, adj_graph2_index, adj_graph2_value, num_nodes, num_nodes, num_nodes, coalesced=True)
        elif args.algorithm == "AA":
            row, col = adj_graph1_index
            weighted_deg = scatter_add(adj_graph1_value, row, dim=0, dim_size=num_nodes)
            tmp1_value = weighted_deg[row] * adj_graph1_value   
            result_index, result_value = torch_sparse.spspmm(adj_graph1_index, tmp1_value, adj_graph2_index, adj_graph2_value, num_nodes, num_nodes, num_nodes, coalesced=True)
        elif args.algorithm == "RA":
            row, col = adj_graph1_index
            weighted_deg = torch.log(scatter_add(adj_graph1_value, row, dim=0, dim_size=num_nodes))
            tmp1_value = weighted_deg[row] * adj_graph1_value   
            result_index, result_value = torch_sparse.spspmm(adj_graph1_index, tmp1_value, adj_graph2_index, adj_graph2_value, num_nodes, num_nodes, num_nodes, coalesced=True)
        
        result_matrix = torch.sparse_coo_tensor(result_index, result_value, torch.Size([num_nodes, num_nodes]))
        # select value from edge_index
        result_matrix = result_matrix.coalesce()

        tmp = torch.sparse_coo_tensor(batch_edges.T, torch.zeros([np.max(batch_edges.shape)]).to(device), result_matrix.shape)
        result_matrix = result_matrix + tmp
        result_matrix = result_matrix.coalesce()
        tmp = torch.sparse_coo_tensor(batch_edges.T, torch.ones([np.max(batch_edges.shape)]).to(device), result_matrix.shape)
        result_tmp = result_matrix - tmp
        result_tmp = result_tmp.coalesce()
        values = result_matrix.values()[(result_matrix.values() - result_tmp.values()) != 0]         
        # import ipdb; ipdb.set_trace()

        if args.algorithm == "Jaccard":
            row1, col1 = adj_graph1_index
            degs1 = degree(row1, num_nodes=num_nodes)
            row2, col2 = adj_graph2_index
            degs2 = degree(row2, num_nodes=num_nodes)
            row, col = batch_edges[:, 0], batch_edges[:, 1]
            neighbor_sum = degs[row] + degs[col]
            values_denominator = neighbor_sum - values 
            values /= values_denominator


        preds[batch_edge_idx] = values

    new_preds = torch.split(preds, [num_pos_edges, num_neg_edges], dim=-1)
    pos_preds, neg_preds = new_preds[0], new_preds[1]    
        
    # import ipdb; ipdb.set_trace()

    return pos_preds, neg_preds



@torch.no_grad()
def generalized_CN_new(A1, A2, edge_index, algorithm, batch_size=100000):
    # args, num_hop1, num_hop2, batch_size, dataset, path, know_links, pos_edges, neg_edges, adjs=None
    '''
    CN include CN with differnt distance
    '''
    # print(algorithm)
    edge_index = edge_index.cpu()
    # import ipdb; ipdb.set_trace()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    node_degrees1 = np.asarray(np.sum(A1, 1)).flatten()
    node_degrees2 = np.asarray(np.sum(A2, 1)).flatten()
    # , keepdims=False
    # import ipdb; ipdb.set_trace()

    # node_degrees1 = np.reshape(node_degrees1, -1)
    # node_degrees2 = np.reshape(node_degrees2, -1)
    
    scores = []

    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        if algorithm == "CN":
            cur_scores = np.array(np.sum(A1[src].multiply(A2[dst]), axis=1)).flatten()
        elif algorithm == "Jaccard":
            src_degree, dst_degree = node_degrees1[src], node_degrees2[dst]
            cur_scores = np.array(np.sum(A1[src].multiply(A2[dst]), 1)).flatten()
            deminator = src_degree + dst_degree - cur_scores
            deminator = np.where(deminator == 0, deminator, 1)
            cur_scores = cur_scores / deminator
        elif algorithm == "RA":
            node_degrees1 = np.where(node_degrees1 != 0, node_degrees1, 2)
            # node_degrees1 = np.expand_dims(node_degrees1, -1)
            # import ipdb; ipdb.set_trace()
            cur_scores = np.array(np.sum(A1[src].multiply(1 / node_degrees1).multiply(A2[dst]), 1)).flatten()  
            # cur_scores = np.array(np.sum((A1[src] / node_degrees1[src]).tocsr().multiply(A2[dst]), 1)).flatten()
        elif algorithm == "AA":
            node_degrees1 = np.where(node_degrees1 != 0, node_degrees1, 2)
            cur_scores = np.array(np.sum(A1[src].multiply(1/ np.log(node_degrees1)).multiply(A2[dst]), 1)).flatten()  
            # cur_scores = np.array(np.sum(np.divide(A1[src].multiply(A2[dst]), np.log(node_degrees1[src])), 1)).flatten()
        elif algorithm == "PA":
            cur_scores = np.array(node_degrees1[src] * node_degrees2[dst]).flatten()
        # elif algorithm == "katz":
        #     # adapt four hop information
        #     import ipdb; ipdb.set_trace()
        #     cur_scores = np.array(np.sum(A1[src].multiply(A2[dst]), 1))
        #     cur_scores = np.array(np.sum(cur_scores[src].multiply(cur_scores[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    
    print(f"evaluated {algorithm} for {len(scores)} edges")
    
    return scores, edge_index

    
    edge_index = dataset.data.edge_index
    device = edge_index.device
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    
    if adjs == None:
        with open(f"{path}_adj.txt", 'rb') as f:
            adjs = pickle.load(f)
        with open(f"{path}_dis.txt", 'rb') as f:
            adjs_indices = pickle.load(f)
    num_hops = len(adjs_indices)
    adj_graphs = get_adj_indices_with_value(adjs, adjs_indices, device)
    num_pos_edges = np.max(pos_edges.shape)
    num_neg_edges = np.max(neg_edges.shape)
    edges = torch.cat([pos_edges, neg_edges], dim=0)
    edge_loader = DataLoader(range(np.max(edges.shape)), batch_size, shuffle=False)
    
    preds = torch.zeros([num_pos_edges + num_neg_edges]).to(device)
    for batch_edge_idx in tqdm(edge_loader):
        batch_edges = edges[batch_edge_idx]
        rows, cols = batch_edges[:, 0], batch_edges[:, 1]
        adj_graph1 = adj_graphs[num_hop1]
        adj_graph2 = adj_graphs[num_hop2]
        
        # adj1_value = subgraph(rows, adj_graph1.indices(), adj_graph1.values(), relabel_nodes=False, num_nodes=num_nodes)
        adj_graph1_index, adj_graph1_value = subgraph(rows, adj_graph1.indices(), adj_graph1.values(), relabel_nodes=False, num_nodes=num_nodes)   
        adj_graph2_index, adj_graph2_value = subgraph(cols, adj_graph2.indices(), adj_graph2.values(), relabel_nodes=False, num_nodes=num_nodes)   

        if args.algorithm == "CN" or args.algorithm == "Jaccard":
            result_index, result_value = torch_sparse.spspmm(adj_graph1_index, adj_graph1_value, adj_graph2_index, adj_graph2_value, num_nodes, num_nodes, num_nodes, coalesced=True)
        elif args.algorithm == "AA":
            row, col = adj_graph1_index
            weighted_deg = scatter_add(adj_graph1_value, row, dim=0, dim_size=num_nodes)
            tmp1_value = weighted_deg[row] * adj_graph1_value   
            result_index, result_value = torch_sparse.spspmm(adj_graph1_index, tmp1_value, adj_graph2_index, adj_graph2_value, num_nodes, num_nodes, num_nodes, coalesced=True)
        elif args.algorithm == "RA":
            row, col = adj_graph1_index
            weighted_deg = torch.log(scatter_add(adj_graph1_value, row, dim=0, dim_size=num_nodes))
            tmp1_value = weighted_deg[row] * adj_graph1_value   
            result_index, result_value = torch_sparse.spspmm(adj_graph1_index, tmp1_value, adj_graph2_index, adj_graph2_value, num_nodes, num_nodes, num_nodes, coalesced=True)
        
        result_matrix = torch.sparse_coo_tensor(result_index, result_value, torch.Size([num_nodes, num_nodes]))
        # select value from edge_index
        result_matrix = result_matrix.coalesce()

        tmp = torch.sparse_coo_tensor(batch_edges.T, torch.zeros([np.max(batch_edges.shape)]).to(device), result_matrix.shape)
        result_matrix = result_matrix + tmp
        result_matrix = result_matrix.coalesce()
        tmp = torch.sparse_coo_tensor(batch_edges.T, torch.ones([np.max(batch_edges.shape)]).to(device), result_matrix.shape)
        result_tmp = result_matrix - tmp
        result_tmp = result_tmp.coalesce()
        values = result_matrix.values()[(result_matrix.values() - result_tmp.values()) != 0]         
        # import ipdb; ipdb.set_trace()

        if args.algorithm == "Jaccard":
            row1, col1 = adj_graph1_index
            degs1 = degree(row1, num_nodes=num_nodes)
            row2, col2 = adj_graph2_index
            degs2 = degree(row2, num_nodes=num_nodes)
            row, col = batch_edges[:, 0], batch_edges[:, 1]
            neighbor_sum = degs[row] + degs[col]
            values_denominator = neighbor_sum - values 
            values /= values_denominator


        preds[batch_edge_idx] = values

    new_preds = torch.split(preds, [num_pos_edges, num_neg_edges], dim=-1)
    pos_preds, neg_preds = new_preds[0], new_preds[1]    
        
    # import ipdb; ipdb.set_trace()

    return pos_preds, neg_preds


@torch.no_grad()
def PPR(A, edge_index):
    """
    The Personalized PageRank heuristic score.
    Need to install fast_pagerank by "pip install fast-pagerank"
    Too slow for large datasets now.
    :param A: A CSR matrix using the 'message passing' edges
    :param edge_index: The supervision edges to be scored
    :return:
    """
    from fast_pagerank import pagerank_power
    num_nodes = A.shape[0]
    src_index, sort_indices = torch.sort(edge_index[:, 0])
    dst_index = edge_index[sort_indices, 1]
    edge_reindex = torch.stack([src_index, dst_index])
    scores = []
    visited = set([])
    j = 0
    for i in tqdm(range(edge_reindex.shape[1])):
        if i < j:
            continue
        src = edge_reindex[0, i]
        personalize = np.zeros(num_nodes)
        personalize[src] = 1
        # get the ppr for the current source node
        ppr = pagerank_power(A, p=0.85, personalize=personalize, tol=1e-7)
        j = i
        # get ppr for all links that start at this source to save recalculating the ppr score
        while edge_reindex[0, j] == src:
            j += 1
            if j == edge_reindex.shape[1]:
                break
        all_dst = edge_reindex[1, i:j]
        cur_scores = ppr[all_dst]
        if cur_scores.ndim == 0:
            cur_scores = np.expand_dims(cur_scores, 0)
        scores.append(np.array(cur_scores))

    scores = np.concatenate(scores, 0)
    print(f'evaluated PPR for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_reindex


def PPR_new(known_links, edge_index, is_old_neg=0, batch_size=100000):
    # import ipdb; ipdb.set_trace()
    origin_edge_index = edge_index
    num_nodes = torch.max(edge_index).item() + 1
    known_links = known_links.T
    neighbors, neighbor_weights = get_ppr_matrix(known_links, num_nodes)
    # import ipdb; ipdb.set_trace()
    # for neighbor, neighbor_weight in zip(neighbors, neighbor_weights):
    #     neighbor, neighbor_weight = np.array(neighbor), np.array(neighbor_weight)
    #     indices = np.argsort(neighbor_weight)[::-1]
    #     neighbor, neighbor_weight = neighbor[indices], neighbor_weight[indices]
    #     neighbor, neighbor_weight = np.delete(neighbor, 0), np.delete(neighbor_weight, 0)
    num_neighbors = 0
    for i in range(len(neighbors)):
        num_neighbors += (len(neighbors[i]) - 1)
    train_edge_index = np.zeros([num_neighbors, 2], dtype=np.int32)
    train_edge_value = np.zeros([num_neighbors])
    
    prev_idx = 0
    for i in tqdm(range(len(neighbors))):
        num = len(neighbors[i]) - 1
        train_edge_index[prev_idx:prev_idx+num, 0] = i
        train_edge_index[prev_idx:prev_idx+num, 1] = neighbors[i][1:]
        train_edge_value[prev_idx:prev_idx+num] = neighbor_weights[i][1:]
        prev_idx += num
    # import ipdb; ipdb.set_trace()
    train_edge_index, train_edge_value = torch.LongTensor(train_edge_index.T), torch.FloatTensor(train_edge_value)
    
    A = torch.sparse_coo_tensor(train_edge_index, train_edge_value, torch.Size([num_nodes, num_nodes]))
    A = A.coalesce()
    edge_index = edge_index.cpu().numpy()
    rows, cols = edge_index[:, 0], edge_index[:, 1]
    record_index = np.lexsort((cols, rows))
    edge_index = edge_index[record_index]
    edge_index = torch.LongTensor(edge_index)
    
    # repeat_mask = torch.zeros([np.max(edge_index.shape)])
    if is_old_neg == 0:
        ind = torch.cat([torch.arange(1, np.max(edge_index.shape)), torch.tensor([0])], dim=0)
        edge_index_compare = torch.index_select(edge_index, 0, ind)
        not_repeat = ~(torch.sum((edge_index == edge_index_compare), dim=-1)[:-1] == 2)
        repeat_index = torch.cat([torch.tensor([0]), torch.cumsum(not_repeat, 0)], dim=0)
        
        not_repeat = torch.cat([torch.tensor([True]), not_repeat], dim=0)
        # import ipdb; ipdb.set_trace()
        edge_index = edge_index[not_repeat, :]                                          
          
        # import ipdb; ipdb.set_trace()
        '''
        count = 0
        repeat_index_tmp = np.zeros([np.max(edge_index.shape)])
        for i in tqdm(range(1, np.max(edge_index.shape))):
            if (edge_index[i-1] == edge_index[i]).sum().item() == 2:
                repeat_index_tmp[i] = count
            else:            
                count += 1
                repeat_index_tmp[i] = count

        repeat_index_tmp = torch.LongTensor(repeat_index_tmp)
        data = repeat_index == repeat_index_tmp
        import ipdb; ipdb.set_trace()
        '''
    # import ipdb; ipdb.set_trace()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    result_indexes, result_values = [], []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        index = torch.stack([src, dst], dim=0)
        origin_index_adj = torch.sparse_coo_tensor(index, torch.ones(index.shape[1]), torch.Size([num_nodes, num_nodes]))
        origin_index_adj = origin_index_adj.coalesce()
        index_adj = origin_index_adj * A
        index_adj = index_adj.coalesce()
        ind_tmp = (index_adj.values() != 0)
        values = index_adj.values()[ind_tmp]
        valued_index = index_adj.indices().T[ind_tmp]
        tmp_adj = torch.sparse_coo_tensor(valued_index.T, torch.ones(valued_index.shape[0]), torch.Size([num_nodes, num_nodes]))
        if ind_tmp.sum().item() != 0:
            other_adj = origin_index_adj - tmp_adj
        else:
            other_adj = origin_index_adj
        other_adj = other_adj.coalesce()
        ind_tmp = (other_adj.values() != 0)
        other_index = other_adj.indices().T[ind_tmp]
        other_values = torch.zeros([np.max(other_index.shape)])
        
        indexes = torch.cat([valued_index, other_index], dim=0)        
        values = torch.cat([values, other_values], dim=0)
        # import ipdb; ipdb.set_trace()
        indexes, values = torch_sparse.coalesce(indexes.T, values, num_nodes, num_nodes, op="add")
        indexes = indexes.T
        result_indexes.append(indexes)
        result_values.append(values)
        # print(values.shape[0] 
        '''
        datas = torch.cat([valued_index, values], dim=-1)
        datas = datas.cpu().numpy()
        rows, cols, values = datas[:, 0], datas[:, 1], datas[:, 2] 
        ind = np.lexsort((values, cols, rows))
        datas = datas[ind]
        '''
    result_indexes = torch.cat(result_indexes, dim=0) 
    result_values = torch.cat(result_values, dim=-1)     

    result_indexes = result_indexes.cpu().numpy()
    rows, cols = result_indexes[:, 0], result_indexes[:, 1]
    ind = np.lexsort((cols, rows))
    result_indexes = result_indexes[ind]
    result_values = result_values[ind]
    result_indexes = torch.tensor(result_indexes)
    # for i in range(np.max(result_indexes.shape)):
    #     if (result_indexes[i] == result_indexes[i+1]).sum().item() == 2:
    #         import ipdb; ipdb.set_trace()
    
    if is_old_neg == 0:
        # import ipdb; ipdb.set_trace()
        result_indexes, result_values = torch.index_select(result_indexes, 0, repeat_index), torch.index_select(result_values, 0, repeat_index)
        
    edge_index = torch.LongTensor(edge_index)

    # origin_index = np.flip(np.argsort(record_index)).copy()
    origin_index = np.argsort(record_index)
    result_indexes, result_values = torch.index_select(result_indexes, 0, torch.LongTensor(origin_index)), torch.index_select(result_values, 0, torch.LongTensor(origin_index))

    return result_values





def katz_apro(A, edge_index, beta=0.005, path_len=3, remove=False):
    scores = []
    G = nx.from_scipy_sparse_matrix(A)
    path_len = int(path_len)
    count = 0
    add_flag1 = 0
    add_flag2 = 0
    count1 = count2 = 0
    betas = np.zeros(path_len)
    print('remove: ', remove)
    for i in range(len(betas)):
        betas[i] = np.power(beta, i+1)
    
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        if s == t:
            count += 1
            scores.append(0)
            continue
        
        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
                
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1


        paths = np.zeros(path_len)
        for path in nx.all_simple_paths(G, source=s, target=t, cutoff=path_len):
            paths[len(path)-2] += 1  
        
        kz = np.sum(betas * paths)

        scores.append(kz)
        
        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
        
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores)






def PPR_correct(known_links, edge_index, dataset, is_old_neg=0, batch_size=100000):
    origin_edge_index = edge_index
    num_nodes = torch.max(edge_index).item() + 1
    known_links = known_links.T
    neighbors, neighbor_weights = get_ppr_matrix(known_links, num_nodes)
    num_neighbors = 0
    for i in range(len(neighbors)):
        num_neighbors += (len(neighbors[i]) - 1)
    train_edge_index = np.zeros([num_neighbors, 2], dtype=np.int32)
    train_edge_value = np.zeros([num_neighbors])
    
    prev_idx = 0
    for i in tqdm(range(len(neighbors))):
        num = len(neighbors[i]) - 1
        train_edge_index[prev_idx:prev_idx+num, 0] = i
        train_edge_index[prev_idx:prev_idx+num, 1] = neighbors[i][1:]
        train_edge_value[prev_idx:prev_idx+num] = neighbor_weights[i][1:]
        prev_idx += num
    train_edge_index, train_edge_value = torch.LongTensor(train_edge_index.T), torch.FloatTensor(train_edge_value)
    
    A = torch.sparse_coo_tensor(train_edge_index, train_edge_value, torch.Size([num_nodes, num_nodes]))
    A = A.coalesce()
    edge_index = edge_index.cpu().numpy()
    rows, cols = edge_index[:, 0], edge_index[:, 1]
    record_index = np.lexsort((cols, rows))
    edge_index = edge_index[record_index]
    edge_index = torch.LongTensor(edge_index)
    
    if is_old_neg == 0:
        ind = torch.cat([torch.arange(1, np.max(edge_index.shape)), torch.tensor([0])], dim=0)
        edge_index_compare = torch.index_select(edge_index, 0, ind)
        not_repeat = ~(torch.sum((edge_index == edge_index_compare), dim=-1)[:-1] == 2)
        repeat_index = torch.cat([torch.tensor([0]), torch.cumsum(not_repeat, 0)], dim=0)
        
        not_repeat = torch.cat([torch.tensor([True]), not_repeat], dim=0)
        edge_index = edge_index[not_repeat, :]                                          
          
    
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    result_indexes, result_values = [], []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        index_list = [torch.stack([dst, src], dim=0), torch.stack([src, dst], dim=0)]
        # index = torch.stack([src, dst], dim=0)
        values_list = []
        for index in index_list:
            origin_index_adj = torch.sparse_coo_tensor(index, torch.ones(index.shape[1]), torch.Size([num_nodes, num_nodes]))
            origin_index_adj = origin_index_adj.coalesce()
            index_adj = origin_index_adj * A
            index_adj = index_adj.coalesce()
            ind_tmp = (index_adj.values() != 0)
            values = index_adj.values()[ind_tmp]
            valued_index = index_adj.indices().T[ind_tmp]
            tmp_adj = torch.sparse_coo_tensor(valued_index.T, torch.ones(valued_index.shape[0]), torch.Size([num_nodes, num_nodes]))
            if ind_tmp.sum().item() != 0:
                other_adj = origin_index_adj - tmp_adj
            else:
                other_adj = origin_index_adj
            other_adj = other_adj.coalesce()
            ind_tmp = (other_adj.values() != 0)
            other_index = other_adj.indices().T[ind_tmp]
            other_values = torch.zeros([np.max(other_index.shape)])
            
            indexes = torch.cat([valued_index, other_index], dim=0)        
            values = torch.cat([values, other_values], dim=0)
            indexes, values = torch_sparse.coalesce(indexes.T, values, num_nodes, num_nodes, op="add")
            indexes = indexes.T
            values_list.append(values)
        result_indexes.append(indexes)
        result_values.append(values_list[0] + values_list[1])
    result_indexes = torch.cat(result_indexes, dim=0) 
    result_values = torch.cat(result_values, dim=-1)     

    result_indexes = result_indexes.cpu().numpy()
    rows, cols = result_indexes[:, 0], result_indexes[:, 1]
    ind = np.lexsort((cols, rows))
    result_indexes = result_indexes[ind]
    result_values = result_values[ind]
    result_indexes = torch.tensor(result_indexes)
    
    if is_old_neg == 0:
        result_indexes, result_values = torch.index_select(result_indexes, 0, repeat_index), torch.index_select(result_values, 0, repeat_index)
        
    edge_index = torch.LongTensor(edge_index)

    origin_index = np.argsort(record_index)
    result_indexes, result_values = torch.index_select(result_indexes, 0, torch.LongTensor(origin_index)), torch.index_select(result_values, 0, torch.LongTensor(origin_index))

    return result_values

def SimRank_correct(known_links, edge_index, dataset_name, is_old_neg=0, batch_size=100000):
    with open(f"/egr/research-dselab/haitaoma/LinkPrediction/subgraph-sketching/src/intermedia_result/simrank_adjs/{dataset_name}.txt", "rb") as f:
        data = pickle.load(f)
    
    num_nodes = torch.max(edge_index).item() + 1
    known_links = known_links.T
    
    sim_edge_index, sim_edge_value = torch.LongTensor(data["index"]).T, torch.FloatTensor(data["value"])
    
    A = torch.sparse_coo_tensor(sim_edge_index, sim_edge_value, torch.Size([num_nodes, num_nodes]))
    A = A.coalesce()
    edge_index = edge_index.cpu().numpy()
    rows, cols = edge_index[:, 0], edge_index[:, 1]
    record_index = np.lexsort((cols, rows))
    edge_index = edge_index[record_index]
    edge_index = torch.LongTensor(edge_index)
    
    if is_old_neg == 0:
        ind = torch.cat([torch.arange(1, np.max(edge_index.shape)), torch.tensor([0])], dim=0)
        edge_index_compare = torch.index_select(edge_index, 0, ind)
        not_repeat = ~(torch.sum((edge_index == edge_index_compare), dim=-1)[:-1] == 2)
        repeat_index = torch.cat([torch.tensor([0]), torch.cumsum(not_repeat, 0)], dim=0)
        
        not_repeat = torch.cat([torch.tensor([True]), not_repeat], dim=0)
        edge_index = edge_index[not_repeat, :]                                          
          
        
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    result_indexes, result_values = [], []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        index_list = [torch.stack([dst, src], dim=0), torch.stack([src, dst], dim=0)]
        # index = torch.stack([src, dst], dim=0)
        values_list = []
        for index in index_list:
            origin_index_adj = torch.sparse_coo_tensor(index, torch.ones(index.shape[1]), torch.Size([num_nodes, num_nodes]))
            origin_index_adj = origin_index_adj.coalesce()
            index_adj = origin_index_adj * A
            index_adj = index_adj.coalesce()
            ind_tmp = (index_adj.values() != 0)
            values = index_adj.values()[ind_tmp]
            valued_index = index_adj.indices().T[ind_tmp]
            tmp_adj = torch.sparse_coo_tensor(valued_index.T, torch.ones(valued_index.shape[0]), torch.Size([num_nodes, num_nodes]))
            if ind_tmp.sum().item() != 0:
                other_adj = origin_index_adj - tmp_adj
            else:
                other_adj = origin_index_adj
            other_adj = other_adj.coalesce()
            ind_tmp = (other_adj.values() != 0)
            other_index = other_adj.indices().T[ind_tmp]
            other_values = torch.zeros([np.max(other_index.shape)])
            
            indexes = torch.cat([valued_index, other_index], dim=0)        
            values = torch.cat([values, other_values], dim=0)
            indexes, values = torch_sparse.coalesce(indexes.T, values, num_nodes, num_nodes, op="add")
            indexes = indexes.T
            values_list.append(values)
        result_indexes.append(indexes)
        result_values.append(values_list[0] + values_list[1])
    result_indexes = torch.cat(result_indexes, dim=0) 
    result_values = torch.cat(result_values, dim=-1)     

    result_indexes = result_indexes.cpu().numpy()
    rows, cols = result_indexes[:, 0], result_indexes[:, 1]
    ind = np.lexsort((cols, rows))
    result_indexes = result_indexes[ind]
    result_values = result_values[ind]
    result_indexes = torch.tensor(result_indexes)
    
    if is_old_neg == 0:
        result_indexes, result_values = torch.index_select(result_indexes, 0, repeat_index), torch.index_select(result_values, 0, repeat_index)
        
    edge_index = torch.LongTensor(edge_index)

    origin_index = np.argsort(record_index)
    result_indexes, result_values = torch.index_select(result_indexes, 0, torch.LongTensor(origin_index)), torch.index_select(result_values, 0, torch.LongTensor(origin_index))

    return result_values




def PPR_old(known_links, edge_index, batch_size=100000):
    origin_edge_index = edge_index
    num_nodes = torch.max(edge_index).item() + 1
    known_links = known_links.T
    neighbors, neighbor_weights = get_ppr_matrix(known_links, num_nodes)
    num_neighbors = 0
    for i in range(len(neighbors)):
        num_neighbors += (len(neighbors[i]) - 1)
    train_edge_index = np.zeros([num_neighbors, 2], dtype=np.int32)
    train_edge_value = np.zeros([num_neighbors])
    
    prev_idx = 0
    for i in tqdm(range(len(neighbors))):
        num = len(neighbors[i]) - 1
        train_edge_index[prev_idx:prev_idx+num, 0] = i
        train_edge_index[prev_idx:prev_idx+num, 1] = neighbors[i][1:]
        train_edge_value[prev_idx:prev_idx+num] = neighbor_weights[i][1:]
        prev_idx += num
    train_edge_index, train_edge_value = torch.LongTensor(train_edge_index.T), torch.FloatTensor(train_edge_value)
    
    A = torch.sparse_coo_tensor(train_edge_index, train_edge_value, torch.Size([num_nodes, num_nodes]))
    A = A.coalesce()
    edge_index = edge_index.cpu().numpy()
    rows, cols = edge_index[:, 0], edge_index[:, 1]
    record_index = np.lexsort((cols, rows))
    edge_index = edge_index[record_index]
    edge_index = torch.LongTensor(edge_index)
    
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    result_indexes, result_values = [], []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        index = torch.stack([src, dst], dim=0)
        
        origin_index_adj = torch.sparse_coo_tensor(index, torch.ones(index.shape[1]), torch.Size([num_nodes, num_nodes]))
        origin_index_adj = origin_index_adj.coalesce()
        index_adj = origin_index_adj * A
        index_adj = index_adj.coalesce()
        ind = (index_adj.values() != 0)
        values = index_adj.values()[ind]
        valued_index = index_adj.indices().T[ind]
        tmp_adj = torch.sparse_coo_tensor(valued_index.T, torch.ones(valued_index.shape[0]), torch.Size([num_nodes, num_nodes]))
        other_adj = origin_index_adj - tmp_adj
        other_adj = other_adj.coalesce()
        ind = (other_adj.values() != 0)
        other_index = other_adj.indices().T[ind]
        other_values = torch.zeros([np.max(other_index.shape)])
        
        indexes = torch.cat([valued_index, other_index], dim=0)        
        values = torch.cat([values, other_values], dim=0)
        # import ipdb; ipdb.set_trace()
        indexes, values = torch_sparse.coalesce(indexes.T, values, num_nodes, num_nodes, op="add")
        indexes = indexes.T
        result_indexes.append(indexes)
        result_values.append(values)
        '''
        datas = torch.cat([valued_index, values], dim=-1)
        datas = datas.cpu().numpy()
        rows, cols, values = datas[:, 0], datas[:, 1], datas[:, 2] 
        ind = np.lexsort((values, cols, rows))
        datas = datas[ind]
        '''
    # import ipdb; ipdb.set_trace()
    result_indexes = torch.cat(result_indexes, dim=0) 
    result_values = torch.cat(result_values, dim=-1)     

    result_indexes = result_indexes.cpu().numpy()
    rows, cols = result_indexes[:, 0], result_indexes[:, 1]
    ind = np.lexsort((cols, rows))
    result_indexes = result_indexes[ind]
    result_values = result_values[ind]
    result_indexes = torch.tensor(result_indexes)
    
    origin_index = np.argsort(record_index)
    result_indexes, result_values = torch.index_select(result_indexes, 0, torch.LongTensor(origin_index)), torch.index_select(result_values, 0, torch.LongTensor(origin_index))

    return result_values

@torch.no_grad()
def Katz(edge_index, links, dataset_name, num_hop=3, is_old_neg=1, beta=0.005, batch_size=10000, eps=1e-3):
    if dataset_name in ['ogbl-collab']:
        batch_size = 1000
    if dataset_name in ['ogbl-ppa']:
        batch_size = 1000
    """
    Calc katz 
    """
    num_links = np.max(links.shape)
    
    num_nodes = torch.max(edge_index).item() + 1
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    device = edge_index.device
    edge_value = torch.ones(edge_index.size(0), dtype=torch.float32, device=device)
    A = torch.sparse_coo_tensor(edge_index.T, edge_value, torch.Size([num_nodes, num_nodes]))
    A = A.coalesce()
    
    link_loader = DataLoader(range(links.size(0)), batch_size, shuffle=False)
    results = torch.zeros([num_links])
    
    for ind in tqdm(link_loader):
        torch.cuda.empty_cache()
        num_batch = ind.shape[0]
        rows, cols = links[ind, 0], links[ind, 1]
        rows_matrix, cols_matrix = torch.zeros([num_batch, num_nodes], device=device), torch.zeros([num_nodes, num_batch], device=device)
        id_index = torch.arange(num_batch).to(device)
        rows_matrix[id_index, rows] = 1
        cols_matrix[cols, id_index] = 1
        for i in range(num_hop):
            if i == 0:
                katz_results = beta * torch.sparse.mm(A, cols_matrix)
                katz_results = torch.where(katz_results > eps, katz_results, 0)
                katz_results_sum = katz_results.clone()
            else:
                katz_results = beta * torch.sparse.mm(A, katz_results)
                katz_results = torch.where(katz_results > eps, katz_results, 0)
                katz_results_sum += katz_results
                
        katz_results = torch.sparse.mm(rows_matrix, katz_results_sum)
        katz_results = torch.diag(katz_results)

        katz_results = katz_results.cpu()    
        results[ind] = katz_results

    return results


def get_ppr_matrix(edge_index, num_nodes, alpha=0.15, eps=5e-5):
    """
    Calc PPR data

    Returns scores and the corresponding nodes

    Adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/transforms/gdc.py
    """
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_index_np = edge_index.cpu().numpy()

    # Assumes sorted and coalesced edge indices (NOTE: coalesce also sorts edges)
    indptr = torch._convert_indices_from_coo_to_csr(edge_index[0], num_nodes).cpu().numpy()
    
    out_degree = indptr[1:] - indptr[:-1]
    
    start = time()
    print("Calculating PPR...", flush=True)
    neighbors, neighbor_weights = get_calc_ppr()(indptr, edge_index_np[1], out_degree, alpha, eps)
    print(f"Time: {time()-start:.2f} seconds")

    print("\n# Nodes with 0 PPR scores:", sum([len(x) == 1 for x in neighbors]))  # 1 bec. itself
    print(f"Mean # of scores per Node: {np.mean([len(x) for x in neighbors]):.1f}")

    return neighbors, neighbor_weights


def get_calc_ppr():
    """
    Courtesy of https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/transforms/gdc.py
    """
    import numba

    @numba.jit(nopython=True, parallel=True)
    def calc_ppr(
        indptr: np.ndarray,
        indices: np.ndarray,
        out_degree: np.ndarray,
        alpha: float,
        eps: float,
    ):
        r"""Calculate the personalized PageRank vector for all nodes
        using a variant of the Andersen algorithm
        (see Andersen et al. :Local Graph Partitioning using PageRank Vectors.)

        Args:
            indptr (np.ndarray): Index pointer for the sparse matrix
                (CSR-format).
            indices (np.ndarray): Indices of the sparse matrix entries
                (CSR-format).
            out_degree (np.ndarray): Out-degree of each node.
            alpha (float): Alpha of the PageRank to calculate.
            eps (float): Threshold for PPR calculation stopping criterion
                (:obj:`edge_weight >= eps * out_degree`).

        :rtype: (:class:`List[List[int]]`, :class:`List[List[float]]`)
        """
        alpha_eps = alpha * eps
        js = [[0]] * len(out_degree)
        vals = [[0.]] * len(out_degree)
        for inode_uint in numba.prange(len(out_degree)):
            inode = numba.int64(inode_uint)
            p = {inode: 0.0}
            r = {}
            r[inode] = alpha
            q = [inode]
            while len(q) > 0:
                unode = q.pop()

                res = r[unode] if unode in r else 0
                if unode in p:
                    p[unode] += res
                else:
                    p[unode] = res
                r[unode] = 0
                for vnode in indices[indptr[unode]:indptr[unode + 1]]:
                    _val = (1 - alpha) * res / out_degree[unode]
                    if vnode in r:
                        r[vnode] += _val
                    else:
                        r[vnode] = _val

                    res_vnode = r[vnode] if vnode in r else 0
                    if res_vnode >= alpha_eps * out_degree[vnode]:
                        if vnode not in q:
                            q.append(vnode)
            js[inode] = list(p.keys())
            vals[inode] = list(p.values())

        return js, vals

    return calc_ppr
    

    



@torch.no_grad()
def generalized_PA(args, num_hop1, num_hop2, batch_size, dataset, path, know_links, pos_edges, neg_edges, adjs=None):
    '''
    PA include PA with differnt distance
    '''
    edge_index = dataset.data.edge_index
    device = edge_index.device
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    
    if adjs == None:
        with open(f"{path}_adj.txt", 'rb') as f:
            adjs = pickle.load(f)
        with open(f"{path}_dis.txt", 'rb') as f:
            adjs_indices = pickle.load(f)
    num_hops = len(adjs_indices)
    adj_graphs = get_adj_indices_with_value(adjs, adjs_indices, device)
    num_pos_edges = np.max(pos_edges.shape)
    num_neg_edges = np.max(neg_edges.shape)
    edges = torch.cat([pos_edges, neg_edges], dim=0)
    edge_loader = DataLoader(range(np.max(edges.shape)), batch_size, shuffle=False)
    
    preds = torch.zeros([num_pos_edges + num_neg_edges]).to(device)
    for batch_edge_idx in tqdm(edge_loader):
        batch_edges = edges[batch_edge_idx]
        rows, cols = batch_edges[:, 0], batch_edges[:, 1]
        adj_graph1 = adj_graphs[num_hop1]
        adj_graph2 = adj_graphs[num_hop2]
        
        # adj1_value = subgraph(rows, adj_graph1.indices(), adj_graph1.values(), relabel_nodes=False, num_nodes=num_nodes)
        adj_graph1_index, adj_graph1_value = subgraph(rows, adj_graph1.indices(), adj_graph1.values(), relabel_nodes=False, num_nodes=num_nodes)   
        adj_graph2_index, adj_graph2_value = subgraph(cols, adj_graph2.indices(), adj_graph2.values(), relabel_nodes=False, num_nodes=num_nodes)   

        row1, col1 = adj_graph1_index
        degs1 = degree(row1, num_nodes=num_nodes)
        row2, col2 = adj_graph2_index
        degs2 = degree(row2, num_nodes=num_nodes)
        row, col = batch_edges[:, 0], batch_edges[:, 1]
        values = degs1[row] * degs2[col]
        preds[batch_edge_idx] = values
            
    new_preds = torch.split(preds, [num_pos_edges, num_neg_edges], dim=-1)
    pos_preds, neg_preds = new_preds[0], new_preds[1]    

    return pos_preds, neg_preds
    

@torch.no_grad()
def AA(A, edge_index, batch_size=100000):
    """
    Adamic Adar
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index
    :param batch_size: int
    :return: FloatTensor [edges] of scores, pyg edge_index
    """
    multiplier = 1 / np.log(A.sum(axis=0))
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Adamic Adar for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index





@torch.no_grad()
def Jaccard(A, edge_index, batch_size=100000):
    """
    Jaccard
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index
    :param batch_size: int
    :return: FloatTensor [edges] of scores, pyg edge_index
    """
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    num_nodes = torch.max(edge_index).item() + 1
    node_degrees = degree(edge_index, num_nodes)

    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        src_degree, dst_degree = node_degrees[src], node_degrees[dst]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        deminator = src_degree + dst_degree - cur_scores
        scores.append(cur_scores / deminator)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Common Neighbours for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index

@torch.no_grad()
def RA(A, edge_index, batch_size=100000):
    """
    Resource Allocation https://arxiv.org/pdf/0901.0553.pdf
    :param A: scipy sparse adjacency matrix
    :param edge_index: pyg edge_index
    :param batch_size: int
    :return: FloatTensor [edges] of scores, pyg edge_index
    """
    multiplier = 1 / A.sum(axis=0)
    multiplier[np.isinf(multiplier)] = 0
    A_ = A.multiply(multiplier).tocsr()
    link_loader = DataLoader(range(edge_index.size(0)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[ind, 0], edge_index[ind, 1]
        cur_scores = np.array(np.sum(A[src].multiply(A_[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Resource Allocation for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index


def shortest_path(A, edge_index, remove=False):
    scores = []
    G = nx.from_scipy_sparse_matrix(A)
    add_flag1 = 0
    add_flag2 = 0
    count = 0
    count1 = count2 = 0
    print('remove: ', remove)
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()
        if s == t:
            count += 1
            scores.append(999)
            continue

        # if (s,t) in train_pos_list: train_pos_list.remove((s,t))
        # if (t,s) in train_pos_list: train_pos_list.remove((t,s))


        # G = nx.Graph(train_pos_list)
        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1

        if nx.has_path(G, source=s, target=t):

            sp = nx.shortest_path_length(G, source=s, target=t)
            # if sp == 0:
            #     print(1)
        else:
            sp = 999
        

        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
    

        scores.append(1/(sp))
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores)

def katz_apro(A, edge_index, beta=0.005, path_len=3, remove=False):
    scores = []
    G = nx.from_scipy_sparse_matrix(A)
    path_len = int(path_len)
    count = 0
    add_flag1 = 0
    add_flag2 = 0
    count1 = count2 = 0
    betas = np.zeros(path_len)
    print('remove: ', remove)
    for i in range(len(betas)):
        betas[i] = np.power(beta, i+1)
    
    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        if s == t:
            count += 1
            scores.append(0)
            continue
        
        if remove:
            if (s,t) in G.edges: 
                G.remove_edge(s,t)
                add_flag1 = 1
                count1 += 1
                
            if (t,s) in G.edges: 
                G.remove_edge(t,s)
                add_flag2 = 1
                count2 += 1

        paths = np.zeros(path_len)
        for path in nx.all_simple_paths(G, source=s, target=t, cutoff=path_len):
            paths[len(path)-2] += 1  
        
        kz = np.sum(betas * paths)

        scores.append(kz)
        
        if add_flag1 == 1: 
            G.add_edge(s,t)
            add_flag1 = 0

        if add_flag2 == 1: 
            G.add_edge(t, s)
            add_flag2 = 0
        
    print('equal number: ', count)
    print('count1: ', count1)
    print('count2: ', count2)

    return torch.FloatTensor(scores)


def katz_close(A, edge_index, beta=0.005):

    scores = []
    G = nx.from_scipy_sparse_matrix(A)

    adj = nx.adjacency_matrix(G, nodelist=range(len(G.nodes)))
    aux = adj.T.multiply(-beta).todense()
    np.fill_diagonal(aux, 1+aux.diagonal())
    sim = np.linalg.inv(aux)
    np.fill_diagonal(sim, sim.diagonal()-1)

    for i in range(edge_index.size(1)):
        s = edge_index[0][i].item()
        t = edge_index[1][i].item()

        scores.append(sim[s,t])

    
    return torch.FloatTensor(scores)

