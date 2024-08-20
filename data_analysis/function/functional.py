from math import inf
import random
import time
import torch
# from data_analysis.function.heuristics import *
from data_analysis.function.heuristics import CN, CN_new, generalized_CN_new, PPR
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
from torch_geometric.utils import (negative_sampling, add_self_loops, to_networkx, degree, subgraph, k_hop_subgraph, is_undirected, to_undirected, remove_self_loops)
import torch_sparse 
from tqdm import tqdm
import scipy.sparse as ssp
from scipy.sparse import csr_matrix
import pickle
from collections import Counter, defaultdict
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import scipy.sparse as sp
from torch_scatter import scatter_mean, scatter_sum
from sklearn.linear_model import LogisticRegression

# since the graph is directed, remove the reductance one for efficiency
@torch.no_grad()
def remove_redudant_edge(links):
    links = [(link[0], link[1]) for link in links]
            
    # Step 1: Create a dictionary to store unique edges and their indices
    unique_edges = {}

    # Iterate over the edge list and store unique edges with their indices
    for idx, edge in enumerate(links):
        sorted_edge = tuple(sorted(edge))  # Sort the edge to handle undirectedness
        if sorted_edge not in unique_edges:
            unique_edges[sorted_edge] = idx

    # Step 2: Extract unique edges and indices from the dictionary
    reduced_edge_list = list(unique_edges.keys())
    remaining_indices = list(unique_edges.values())

    return reduced_edge_list, remaining_indices
    

# TODO: supporting for the subgraph 
@torch.no_grad()
def matrix_multiply(edge_index, path, num_hops):
    # path is for the reloading of the adjacent matrix
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    edge_weight = edge_index
    edge_weight = torch.ones(num_edges).to(torch.float32)   #.to(torch.float)
    edge_index, edge_weight = edge_index.cuda(), edge_weight.cuda()
    
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]))
    adj = adj.coalesce() 
    adj_origin = adj.clone()
    
    adjs = [adj_origin.cpu] # record the original adjacent matrix
    adj_indices_record = tensor2set(adj_origin)  # record the existing entrace
    adj_dis = set2tensor(adj_indices_record)
    # node pair with distance one two xxx 
    adjs_dis = [adj_indices_record]

    for i in tqdm(range(num_hops - 1)):
        start_time = time.time()
        torch.cuda.empty_cache()
        adj = torch.sparse.mm(adj, adj_origin)
        adj = adj.coalesce()
        adjs.append(adj)
        # record the adjacent matrix

        adj_indices = tensor2set(adj)
        adj_indice_difference = adj_indices - adj_indices_record
        adjs_dis.append(set2tensor(adj_indice_difference))
        # find the new edge in the new hop
        adj_indices_record = adj_indices_record.union(adj_indices)
        # update the original adj_indices
        
    with open(f"{path}_adj.txt", 'wb') as f:
        pickle.dump(adjs, f)
    with open(f"{path}_dis.txt", 'wb') as f:
        pickle.dump(adjs_dis, f)

    # return adjs, adjs_dis


@torch.no_grad()
def matrix_multiply2(edge_index, path, num_hops, device):
    # path is for the reloading of the adjacent matrix
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    edge_weight = edge_index
    edge_weight = torch.ones(num_edges).to(torch.float32)   #.to(torch.float)
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
    
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]))
    adj = adj.coalesce() 
    adj_origin = adj.clone()
    
    adjs = [adj_origin] # record the original adjacent matrix
    adjs_dis = [adj.indices()]

    for i in tqdm(range(num_hops - 1)):
        torch.cuda.empty_cache()
        adj = torch.sparse.mm(adjs[i], adj_origin)
        adj = adj.coalesce()
        adjs.append(adj)
        # record the adjacent matrix
        # start_time =  time.time()
        for prev_adj in adjs:
            adj -= prev_adj 
        adj = adj.coalesce()
        negative_index = adj.values() < 0
        # find negative values, it may have those does not appear, and also those in the previous
        negative_tensor = torch.sparse_coo_tensor(adj.indices()[:, negative_index], adj.values()[negative_index], torch.Size([num_nodes, num_nodes]))
        results = adjs[-1] + negative_tensor
        results = results.coalesce()
        results_index = results.indices()[:,  results.values == 0]
        # print(time.time() - start_time)
        
    with open(f"{path}_adj.txt", 'wb') as f:
        pickle.dump(adjs, f)
    with open(f"{path}_dis.txt", 'wb') as f:
        pickle.dump(adjs_dis, f)

    return adjs, adjs_dis


@torch.no_grad()
def get_distance(edge_index, path, num_hops):
    # path is for the reloading of the adjacent matrix
    device = edge_index.device
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    edge_weight = torch.ones(num_edges).to(torch.float32)   #.to(torch.float)
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
    
    adj = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]))
    adj = adj.coalesce() 
    adj_origin = adj.clone()
    
    adjs = [adj_origin.cpu()] # record the adjacent matrix [A, A^2, A^3]
    adjs_dis = [adj.indices().cpu()]  
    adj_last = adj_origin
    
    for i in tqdm(range(num_hops - 1)):
        torch.cuda.empty_cache()
        adj = torch.sparse.mm(adj_last, adj_origin)
        adj = adj.coalesce()
        adj_last = adj
        # the above are in GPU
        adj = adj.cpu()
        adjs.append(adj.clone())
        # record the adjacent matrix
        # print(adj.indices().shape)
        # start_time = time.time()
        
        # adj (row, col, value1) (value2)
        # prev_adj (row, col, 0)
        # new_adj (row, col, -value1)
        # new_adj + adj (0)
        
        # find the entry
        for prev_adj in adjs[:-1]:
            adj -= prev_adj
        adj = adj.coalesce()
        # find the new edge in the new hop 

        positive_index = adj.values() > 0
        # find negative values, it may have those does not appear, and also those in the previous
        positive_tensor = torch.sparse_coo_tensor(adj.indices()[:, positive_index], adj.values()[positive_index], torch.Size([num_nodes, num_nodes]))
        # identify the value that equal to the original matrix, indicating, minuend is 0
        results = positive_tensor - adjs[-1]
        results = results.coalesce()
        results_index = results.indices()[:,  results.values() == 0]
        # print(time.time() - start_time)

        adjs_dis.append(results_index)
    
    with open(f"{path}_adj.txt", 'wb') as f:
        pickle.dump(adjs, f)
    with open(f"{path}_dis.txt", 'wb') as f:
        pickle.dump(adjs_dis, f)

    return adjs, adjs_dis


@torch.no_grad()
def target_edge_removal(dataset, path, num_nodes, links=None, adjs=None, adjs_indices=None):
    '''
    current version only support some links, not all the edges, too computational expensive for large dataset
    We only support the distance up to 3 hop

    This function aims to find distance for each node pair without the distance on the tyarget node
    The logit of this function is shown as follows:
    For an give edge: 
    1. first check whether the original distance path does not all fully rely on the neighborhood information
    How to check, whether number of edges (a->c) equal to (a->b->c)
    2. if not, find whether higher order is reachable
  
    output: generate a counter matrix for each node pair
    '''
    edge_index = dataset.data.edge_index
    
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    
    if adjs == None:
        with open(f"{path}_adj.txt", 'rb') as f:
            origin_adjs = pickle.load(f)
        with open(f"{path}_dis.txt", 'rb') as f:
            adjs_indices = pickle.load(f)
    
    adjs, indices = {}, {}
    max_degree = -1
    for idx, (adj, adj_indices) in enumerate(zip(origin_adjs, adjs_indices)):
        dis = idx + 1
        network = to_networkx(Data(edge_index=adj_indices,num_nodes=num_nodes), to_undirected=True)
        print(f"{idx} {len(network.nodes())}")
        indices[dis] = network
        adjs[dis] = adj
        adj = adj.coalesce()
        tmp_max_degree = torch.max(degree(adj.indices()[0])).item()
        max_degree = tmp_max_degree if max_degree < tmp_max_degree else max_degree
    max_degree = int(max_degree)
    
    num_hops = 2
    
    # TODO: remove here for efficiency
    # num_hops = len(adjs)
    
    # for the undirected graph, we remove half the edge
    links = dataset.links if links == None else links.cpu().numpy().tolist()
    # TODO: check the type of the dataset links and links
    reduced_edge_list, remaining_indices = remove_redudant_edge(links)
    tradic_counter = Counter()
    
    dist_results = np.zeros([num_edges, 2, num_hops, max_degree], dtype=np.int32)
    if num_hops > 2: remain_nodes = np.zeros([max_degree], dtype=np.int32)
    remain_idx = 0
    
    # import ipdb; ipdb.set_trace()
    num_edges = len(reduced_edge_list)
    for edge_id, edge in tqdm(enumerate(reduced_edge_list)):
        src_id, dst_id = edge[0], edge[1]
        # calculate the distance to each node
        # if larger than both max hop, just remove it
        # TODO: if only larger than one, just set the longest, this may not be a good choice
        # dist_label = defaultdict(lambda: [False, False])
        # False indicates have not visit , -1 indicates does not in the scope
        for idx, node_id in enumerate([src_id, dst_id]):
            other_idx = np.abs(1 - idx)
            other_node_id = edge[other_idx]
            # if node_id == src_id, then other_node_id is dst_id
            for dis, indices_graph in zip(indices.keys(), indices.values()):
                # TODO: potential remove here
                if dis == 3:
                    break
                
                # abstract the k hop neighborhood
                neighbor_count = 0
                try:
                    neighbor_nodes = indices_graph.neighbors(node_id)
                except:
                    import ipdb; ipdb.set_trace()
                # original neighbor_nodes is an iterator
                neighbor_nodes = [neighbor_node for neighbor_node in neighbor_nodes]
                
                if dis > 2: 
                    # append those detected not in loop two
                    remain_nodes = np.trim_zeros(remain_nodes, 'b')
                    neighbor_nodes = np.concatenate((neighbor_nodes, remain_nodes), axis=-1)
                    # reset the matrix 
                    remain_nodes = np.zeros([max_degree], dtype=np.int32)
                    remain_idx = 0
                    
                # for node in 
                for i in range(len(neighbor_nodes)):
                    neigbor_node_id = int(neighbor_nodes[i])
                    # the number of path (a, c)
                    try:
                        num_path = adjs[dis][node_id, neigbor_node_id]
                    except:
                        import ipdb; ipdb.set_trace()
                    # (a,c)
                    num_target_path = 0 # (a,b,c)
                    if dis > 1:
                        tmp_dis = dis - 1
                        while tmp_dis > 0:
                            try:
                                # there will be self loop if higher order 
                                # (a, b) dist 1: 1 (a,b) dist 3   (b, c)  # (a, b, c)
                                num_self_path = 1 if (dis - tmp_dis) == 1 else adjs[dis - tmp_dis - 1][node_id, other_node_id] # + 1
                                new_num_target_path = num_self_path  * adjs[tmp_dis][other_node_id, neigbor_node_id] 
                            except:
                                new_num_target_path = 0
                            num_target_path += new_num_target_path
                            tmp_dis -= 2                        
                                
                    if num_path - num_target_path > 0: 
                        dist_results[edge_id][idx][dis-1][neighbor_count]=neigbor_node_id 
                        neighbor_count += 1
                    # origin dist 2, may num_hop 3
                    elif dis == 2 and num_hops > 2:
                        remain_nodes[remain_idx] = neigbor_node_id
                        remain_idx += 1
                        # refind the nearest neighbor
                        # currently, since we only use the up to three hop neighbor, then we only consider the two hop neighbor
                    else:
                        continue
                        # dist_results = np.sum(dist_results, axis=-1)
    dist_results = np.count_nonzero(dist_results, axis=-1)
    # import ipdb; ipdb.set_trace()
    # dist_results = np.sum(dist_results, axis=1)
    import ipdb; ipdb.set_trace()
    return dist_results


@torch.no_grad()
def target_edge_removal_new(dataset, path, links=None, adjs=None, adjs_indices=None):
    '''
    current version only support some links, not all the edges, too computational expensive for large dataset
    We only support the distance up to 3 hop

    This function aims to find distance for each node pair without the distance on the tyarget node
    The logit of this function is shown as follows:
    For an give edge: 
    1. first check whether the original distance path does not all fully rely on the neighborhood information
    How to check, whether number of edges (a->c) equal to (a->b->c)
    2. if not, find whether higher order is reachable
  
    output: generate a counter matrix for each node pair
    '''
    edge_index = dataset.data.edge_index
    
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    
    if adjs == None:
        with open(f"{path}_adj.txt", 'rb') as f:
            origin_adjs = pickle.load(f)
        with open(f"{path}_dis.txt", 'rb') as f:
            adjs_indices = pickle.load(f)
    
    
    adjs, indices = {}, {}
    max_degree = -1
    for idx, (adj, adj_indices) in enumerate(zip(origin_adjs, adjs_indices)):
        dis = idx + 1
        indices[dis] = to_networkx(Data(edge_index=adj_indices,num_nodes=num_nodes), to_undirected=True)
        adjs[dis] = adj
        adj = adj.coalesce()
        tmp_max_degree = torch.max(degree(adj.indices()[0])).item()
        # print(tmp_max_degree)
        max_degree = tmp_max_degree if max_degree < tmp_max_degree else max_degree
    max_degree = int(max_degree)
    
    num_hops = len(adjs)
    # for the undirected graph, we remove half the edge
    links = dataset.links if links == None else dataset.links
    # TODO: check the type of the dataset links and links
    reduced_edge_list, remaining_indices = remove_redudant_edge(links)
    tradic_counter = Counter()
    
    dist_results = np.zeros([num_edges, 2, num_hops, max_degree], dtype=np.int32)
    if num_hops > 2: remain_nodes = np.zeros([max_degree], dtype=np.int32)
    remain_idx = 0
    
    # import ipdb; ipdb.set_trace()
    for edge_id, edge in tqdm(enumerate(reduced_edge_list)):
        src_id, dst_id = edge[0], edge[1]
        # calculate the distance to each node
        # if larger than both max hop, just remove it
        # TODO: if only larger than one, just set the longest, this may not be a good choice
        # dist_label = defaultdict(lambda: [False, False])
        # False indicates have not visit , -1 indicates does not in the scope
        for idx, node_id in enumerate([src_id, dst_id]):
            other_idx = np.abs(1 - idx)
            other_node_id = edge[other_idx]
            # if node_id == src_id, then other_node_id is dst_id
            for dis, indices_graph in zip(indices.keys(), indices.values()):
                # abstract the k hop neighborhood
                neighbor_count = 0
                neighbor_nodes = indices_graph.neighbors(node_id)
                neighbor_nodes = [neighbor_node for neighbor_node in neighbor_nodes]
                
                if dis > 2: 
                    # append those detected not in loop two
                    remain_nodes = np.trim_zeros(remain_nodes, 'b')
                    neighbor_nodes = np.concatenate((neighbor_nodes, remain_nodes), axis=-1)
                    # reset the matrix 
                    remain_nodes = np.zeros([max_degree], dtype=np.int32)
                    remain_idx = 0
                    
                for i in range(len(neighbor_nodes)):
                    neigbor_node_id = int(neighbor_nodes[i])
                    # the number of path (a, c)
                    try:
                        num_path = adjs[dis][node_id, neigbor_node_id]
                    except:
                        import ipdb; ipdb.set_trace()
                    num_target_path = 0 # (a,b,c)
                    if dis > 1:
                        tmp_dis = dis - 1
                        while tmp_dis > 0:
                            try:
                                # there will be self loop if higher order 
                                # (a, b) dist 1: 1 (a,b) dist 3   (b, c)  # (a, b, c)
                                num_self_path = 1 if (dis - tmp_dis) == 1 else adjs[dis - tmp_dis - 1][node_id, other_node_id] # + 1
                                new_num_target_path = num_self_path  * adjs[tmp_dis][other_node_id, neigbor_node_id] 
                            except:
                                new_num_target_path = 0
                            num_target_path += new_num_target_path
                            tmp_dis -= 2                        
                                
                    if num_path - num_target_path > 0: 
                        try:
                            dist_results[edge_id][idx][dis-1][neighbor_count]=neigbor_node_id 
                        except:
                            import ipdb; ipdb.set_trace()
                        neighbor_count += 1
                    # origin dist 2, may num_hop 3
                    elif dis == 2 and num_hops > 2:
                        remain_nodes[remain_idx] = neigbor_node_id
                        remain_idx += 1
                        # refind the nearest neighbor
                        # currently, since we only use the up to three hop neighbor, then we only consider the two hop neighbor
                    else:
                        continue
                        dist_results = np.sum(dist_results, axis=-1)
    dist_results = np.count_nonzero(dist_results, axis=-1)
    # import ipdb; ipdb.set_trace()
    return dist_results




@torch.no_grad()
def tradic_count_removal(dataset, path, links=None, adjs=None, adjs_indices=None, save_pair=False):
    dist_results = target_edge_removal(dataset, path, links=None, adjs=None, adjs_indices=None)
    # [num_edges, 2, num_hops, max_degree]
    num_edges, _ ,  num_hops, max_degree = dist_results.shape

    tradic_counts = defaultdict(int)
    num_hops = np.max(adj_graphs.keys()) + 1
    tradic_results = np.zeros(num_hops, num_hops)

    if save_pair:
        dist_pairs = np.zeros([2, num_edges * max_degree])
        dist_count = 0
    else:
        dist_pairs = None

    for edge_idx in range(num_edges):
        dist_result = dist_results[edge_idx]
        dist_label = defaultdict(lambda: (False, False))
        
        # record the node pair distance
        for i in range(2):
            neighbors = dist_result[i]
            for hop_id in range(num_hops):
                for node_id in range(max_degree):
                    if neighbors[hop_id][node_id] != 0:     
                        dist_label[neighbors[hop_id][node_id]][i] = hop_id  
        
        # record the whole distance 
        for node_pair in dist_label.keys():
            node_pair_dis = dist_label[key]
            srt_node_dis, tgt_node_dis = node_pair_dis[0], node_pair_dis[1]
            if not srt_node_dis and not tgt_node_dis:
                continue
            elif not srt_node_dis and tgt_node_dis:
                dist_results[0][tgt_node_dis] += 1
            elif srt_node_dis and not tgt_node_dis:
                dist_results[src_node_dis][0] += 1
            elif srt_node_dis and tgt_node_dis:
                dist_results[src_node_dis][tgt_node_dis] += 1
            if save_pair:
                dist_pairs[0][dist_count] = srt_node_dis 
                dist_pairs[1][dist_count] = tgt_node_dis 
                dist_count += 1

    return dist_results, dist_pairs


@torch.no_grad()
def whole_tradic_count_nonremoval(dataset, path, adjs=None, adjs_indices=None):
    '''
    Count on each node, this version is not a tensor GPU version, the original for effectiveness
    '''
    edge_index = dataset.data.edge_index
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    
    if adjs == None:
        with open(f"{path}_adj.txt", 'rb') as f:
            adjs = pickle.load(f)
        with open(f"{path}_dis.txt", 'rb') as f:
            adjs_indices = pickle.load(f)
    
    # import ipdb; ipdb.set_trace()
    adj_graphs = {}
    # [num_hopm [2, num_edges]]
    for idx, adj_indices in enumerate(adjs_indices):
        dis = idx + 1
        adj_graphs[dis] = to_networkx(Data(edge_index=adj_indices,num_nodes=num_nodes), to_undirected=True)
    num_hops = len(adjs)
    # for the undirected graph, we remove half the edge
    
    dist_results = defaultdict(list)
    for num_hop1, adj_graph1 in tqdm(enumerate(adj_graphs)):
        for num_hop2, adj_graph2 in enumerate(adj_graphs):
            nodes_nbrs = adj_graph1.items()
            for v, v_nbrs in nodes_nbrs:
                vs = set(v_nbrs) - {v}
                gen_degree = Counter(len(vs & (set(adj_graph2[w]) - {w})) for w in vs)
                ntriangles = sum(k * val for k, val in gen_degree.items()) // 2
                dist_results[num_hop1 + 1][num_hop2 + 1].append(ntriangles)
    # TODO: here may be a little bit problematic          
    return dist_results
            

@torch.no_grad()
def tradic_count_nonremoval_sum(dataset, path, adjs=None, adjs_indices=None):
    '''
    Count on all nodes
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
    
    adj_graphs = get_adj_indices_with_value(adjs, adjs_indices, device)
    # A^2 (1, 2)
    num_hops = len(adj_graphs) + 1
    tradic_results = np.zeros([num_hops, num_hops])
    
    for num_hop1, adj_graph1 in enumerate(adj_graphs):
        for num_hop2, adj_graph2 in enumerate(adj_graphs):
            batch_size = 1e6 if adj_graph1._nnz() > 1e6 and adj_graph2._nnz() > 1e6 else np.min([adj_graph1._nnz(), adj_graph2._nnz()])  
            tradic_results[num_hop1 + 1][num_hop2 + 1] = sparse_matrix_reduce_sum(adj_graph1, adj_graph2, batch_size=batch_size)
    
    return tradic_results


@torch.no_grad()
def tradic_count_nonremoval(known_links, dataset, path, args, pos_edges, neg_edges, is_test, batch_size=500, adjs=None, adjs_indices=None):
    # known_links, 
    '''
    Count on each edge, we can also have no cuda version, put it later
    '''
    edge_index = dataset.data.edge_index
    device = edge_index.device
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    
    if args.is_load:
        try:
            with open(f"intermedia_result/tradic_preds/{args.dataset_name}_{is_test}_results.txt", "rb") as f:
                results = pickle.load(f)
                pos_results, neg_results = results["pos"], results["neg"]
                return pos_results, neg_results
        except:
            print("generate results from skectch")
    
    import ipdb; ipdb.set_trace()
    
    if adjs == None:
        with open(f"{path}_adj.txt", 'rb') as f:
            adjs = pickle.load(f)
        with open(f"{path}_dis.txt", 'rb') as f:
            adjs_indices = pickle.load(f)
    adj_graphs = get_adj_indices_with_value(adjs, adjs_indices, device)

    for i in range(len(adj_graphs)):
        indices = adj_graphs[i].indices().cpu().numpy()
        # import ipdb; ipdb.set_trace()
        adj_graphs[i] = sp.coo_matrix((adj_graphs[i].values().cpu().numpy(), (indices[0], indices[1])), shape=(num_nodes, num_nodes)).tocsr()
        # adj_graphs[i] = sp.csr_array((adj_graphs[i].values().cpu().numpy(), (indices[0], indices[1])), shape=(num_nodes, num_nodes)).toarray()

    num_pos_edges = np.max(pos_edges.shape)
    num_neg_edges = np.max(neg_edges.shape)
    edges = torch.cat([pos_edges, neg_edges], dim=0)
    num_hops = len(adj_graphs)
    results = np.zeros([num_hops, num_hops, num_pos_edges + num_neg_edges])
    
    for num_hop1, adj_graph1 in enumerate(adj_graphs):
        for num_hop2, adj_graph2 in enumerate(adj_graphs):
            # import ipdb; ipdb.set_trace()
            preds, _ = generalized_CN_new(adj_graph1, adj_graph2,edges, batch_size=10000)
            results[num_hop1][num_hop2] += preds
    
    results = np.split(results, [num_pos_edges], axis=-1)
    # import ipdb; ipdb.set_trace()
    
    pos_results, neg_results = results[0], results[1]
    
    with open(f"intermedia_result/tradic_preds/{args.dataset_name}_{is_test}_results.txt", "wb") as f:
        pickle.dump({"pos": pos_results, "neg": neg_results}, f)
        
    
    return pos_results, neg_results
    



@torch.no_grad()
def tradic_count_logits(known_links, dataset, path, args, pos_edges, neg_edges, is_test, batch_size=100000, adjs=None, adjs_indices=None):
    '''
    Count on each edge, we can also have no cuda version, put it later
    '''
    edge_index = dataset.data.edge_index
    device = edge_index.device
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    
    if args.is_load:
        try:
            with open(f"intermedia_result/tradic_preds/{args.dataset_name}_{args.algorithm}_{args.encode_type}_{is_test}_{args.is_old_neg}_preds.txt", "rb") as f:
                results = pickle.load(f)
                pos_results, neg_results = results["pos"], results["neg"]
                return pos_results, neg_results
        except:
            print("generate results from skectch")
    
    if adjs == None:
        with open(f"{path}_adj.txt", 'rb') as f:
            adjs = pickle.load(f)
        with open(f"{path}_dis.txt", 'rb') as f:
            adjs_indices = pickle.load(f)
    adj_graphs = get_adj_indices_with_value(adjs, adjs_indices, device)

    for i in range(len(adj_graphs)):
        indices = adj_graphs[i].indices().cpu().numpy()
        # import ipdb; ipdb.set_trace()
        
        adj_graphs[i] = sp.coo_matrix((adj_graphs[i].values().cpu().numpy(), (indices[0], indices[1])), shape=(num_nodes, num_nodes)).tocsr()
        # adj_graphs[i] = sp.csr_array((adj_graphs[i].values().cpu().numpy(), (indices[0], indices[1])), shape=(num_nodes, num_nodes)).toarray()

    num_hops = len(adj_graphs)
    # conduct structural feature
    num_pos_edges = np.max(pos_edges.shape)
    num_neg_edges = np.max(neg_edges.shape)
    edges = torch.cat([pos_edges, neg_edges], dim=0)
    results = np.zeros([num_hops, num_hops, num_pos_edges + num_neg_edges])
    
    for num_hop1, adj_graph1 in enumerate(adj_graphs):
        for num_hop2, adj_graph2 in enumerate(adj_graphs):
            print(args.algorithm)
            preds, _ = generalized_CN_new(adj_graph1, adj_graph2, edges, args.algorithm, batch_size)
            results[num_hop1][num_hop2] += preds
    # results: [num_hop, num_hop, num_edges]
    
    # encode_type: drnl de zo
    
    '''
    data = torch.tensor(np.array(
        [[1, 2, 3],
         [4,5,6],
         [7,8,9]
        ]))
    
    data = torch.reshape(data, [-1])
    data = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    '''
    # import ipdb; ipdb.set_trace()
    results = np.reshape(results, [-1, num_pos_edges + num_neg_edges])
    
    dis_dicts= {2: {0: (1,1), 1: (1,2), 2: (2,1), 3: (2,2)}, 3: {0: (1,1), 1: (1,2), 2: (1,3), 3: (2,1), 4: (2,2), 5: (2,3), 6: (3,1), 7: (3,2), 8: (3,3)}}
    
    dis_dict = dis_dicts[num_hops]
    # index: distance
    
    # reverse key and value in a dict
    #  k: {v: k for k, v in dis_dicts[k].items()}
    dis_dict_reverse = {dis_dict[key]: key for key in dis_dict.keys()}
    # distance: index
    
    def drnl_hash(dis1, dis2):
        dis_sum = dis1 + dis2
        return 1 + min(dis1, dis2) + (dis_sum // 2) *  ((dis_sum // 2) + (dis_sum % 2) - 1)
    
    torch.cuda.empty_cache()
    if args.encode_type == "de":
        # just count all distance
        results = results
    elif args.encode_type == "drnl":
        # generate the seal label
        labels = []
        for ds in dis_dict_reverse.keys():
            label = drnl_hash(ds[0], ds[1])
            labels.append(label) 
            # reduce the label value by the magnitude
            # change to [0, 1, 2, 3]
        label_values = list(set(labels))
        new_labels = np.argsort(label_values)
        
        for i in range(len(labels)):
            labels[i] = new_labels[label_values.index(labels[i])]       
        
        results = torch.tensor(results).to(device)
        labels = torch.tensor(labels).to(device)
        results = scatter_sum(results, labels, dim=0).cpu().numpy()
    else:
        # use zero one label just do not aware any different common neighborhood
        results = np.sum(results, axis=0, keepdims=True)
    
    # TODO: add each dimension represents which distance
    results = np.split(results, [num_pos_edges], axis=-1)
    pos_results, neg_results = results[0], results[1]

    pos_results, neg_results = pos_results.T, neg_results.T          
    with open(f"intermedia_result/tradic_preds/{args.dataset_name}_{args.algorithm}_{args.encode_type}_{is_test}_{args.is_old_neg}_preds.txt", "wb") as f:
        results = {"pos": pos_results, "neg": neg_results}
        pickle.dump(results, f)

    return pos_results, neg_results
        


@torch.no_grad()
def tradic_logistic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path):
    pos_valid_features, neg_valid_features = tradic_count_logits(known_links, dataset, path, args, valid_pos_links, valid_neg_links, is_test=0)
    pos_test_features, neg_test_features = tradic_count_logits(known_links, dataset, path, args, test_pos_links, test_neg_links, is_test=1)

    pos_valid_labels, neg_valid_labels = torch.ones(pos_valid_features.shape[0]), torch.zeros(neg_valid_features.shape[0])
    logits_model = LogisticRegression(random_state=0).fit(np.concatenate([pos_valid_features, neg_valid_features], axis=0), np.expand_dims(np.concatenate([pos_valid_labels, neg_valid_labels], axis=0), axis=-1))
    pos_test_results = logits_model.predict_proba(pos_test_features)
    neg_test_results = logits_model.predict_proba(neg_test_features)
   
    # the probability of positive 
    pos_test_results = pos_test_results[:, 1]
    neg_test_results = neg_test_results[:, 1]
     
    return pos_test_results, neg_test_results
    



@torch.no_grad()
def feature_importance_tradic_logistic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path):
    pos_valid_features, neg_valid_features = tradic_count_logits(known_links, dataset, path, args, valid_pos_links, valid_neg_links, is_test=0)
    pos_test_features, neg_test_features = tradic_count_logits(known_links, dataset, path, args, test_pos_links, test_neg_links, is_test=1)
    
    pos_valid_labels, neg_valid_labels = torch.ones(pos_valid_features.shape[0]), torch.zeros(neg_valid_features.shape[0])
    features = np.concatenate([pos_valid_features, neg_valid_features], axis=0)
    features = features / np.linalg.norm(features, axis=0, keepdims=True)
    logits_model = LogisticRegression(random_state=0).fit(features, np.expand_dims(np.concatenate([pos_valid_labels, neg_valid_labels], axis=0), axis=-1))
    
    weights = np.squeeze(logits_model.coef_)
    # check the weight of logistic
    # import ipdb; ipdb.set_trace()
    
    features = np.concatenate([pos_test_features, neg_test_features], axis=0)
    
    features = features / np.linalg.norm(features, axis=0, keepdims=True)

    num_pos_edges = pos_test_features.shape[0]
    features = np.split(features, [num_pos_edges], axis=0)

    pos_test_features, neg_test_features = features[0], features[1]
    pos_test_results = logits_model.predict_proba(pos_test_features)
    neg_test_results = logits_model.predict_proba(neg_test_features)

    # the probability of positive 
    pos_test_results = pos_test_results[:, 1]
    neg_test_results = neg_test_results[:, 1]
    
    
    return pos_test_results, neg_test_results, weights




@torch.no_grad()
def tradic_algorithm(known_links,dataset, path, args, pos_edges, neg_edges, batch_size=500, adjs=None, adjs_indices=None):
    # known_links, 
    '''
    Count on each edge, we can also have no cuda version, put it later
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
    adj_graphs = get_adj_indices_with_value(adjs, adjs_indices, device)

    for i in range(len(adj_graphs)):
        indices = adj_graphs[i].indices().cpu().numpy()
        # import ipdb; ipdb.set_trace()
        adj_graphs[i] = sp.coo_matrix((adj_graphs[i].values().cpu().numpy(), (indices[0], indices[1])), shape=(num_nodes, num_nodes)).tocsr()
        # adj_graphs[i] = sp.csr_array((adj_graphs[i].values().cpu().numpy(), (indices[0], indices[1])), shape=(num_nodes, num_nodes)).toarray()

    num_pos_edges = np.max(pos_edges.shape)
    num_neg_edges = np.max(neg_edges.shape)
    edges = torch.cat([pos_edges, neg_edges], dim=0)
    num_hops = len(adj_graphs)
    results = np.zeros([num_hops, num_hops, num_pos_edges + num_neg_edges])
    
    for num_hop1, adj_graph1 in enumerate(adj_graphs):
        for num_hop2, adj_graph2 in enumerate(adj_graphs):
            # import ipdb; ipdb.set_trace()
            if args.algorithm in ['katz', 'ppr']:
                if args.algorithm == "katz":
                    preds, _ = generalized_CN_new(adj_graph1, adj_graph2,edges, args.algorithm, batch_size=10000)
                elif args.algorithm == "ppr":
                    preds, _ = PPR(adj_graphs[0], edges.cpu())
                    preds = preds.numpy()
                # import ipdb; ipdb.set_trace()
                results[0][0] += preds
                results = np.split(results, [num_pos_edges], axis=0)
                pos_results, neg_results = results[0], results[1]

                return pos_results, neg_results
            else: 
                preds, _ = generalized_CN_new(adj_graph1, adj_graph2, edges, args.algorithm, batch_size=10000)
            results[num_hop1][num_hop2] += preds
    
    results = np.split(results, [num_pos_edges], axis=-1)
    # import ipdb; ipdb.set_trace()
    
    pos_results, neg_results = results[0], results[1]
    
    return pos_results, neg_results
    
    
@torch.no_grad()
def tradic_count_nonremoval_old(known_links,dataset, path, args, pos_edges, neg_edges, batch_size=500, adjs=None, adjs_indices=None):
    # known_links, 
    '''
    Count on each edge, we can also have no cuda version, put it later
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
            # [num_hops, num_node, num_nodes]
    num_hops = len(adjs_indices)
    adj_graphs = get_adj_indices_with_value(adjs, adjs_indices, device)
    # adj_graph = adj_graphs[0]
    # adj_tmp_graph = torch.sparse_coo_tensor(known_links.T, torch.ones([np.max(known_links.shape)]).to(device), torch.Size([num_nodes, num_nodes]))
    # tmp = adj_graph - adj_tmp_graph
    # tmp = tmp.coalesce()
    # print(torch.sum(tmp.values()).item())
    # import ipdb; ipdb.set_trace()
    num_pos_edges = np.max(pos_edges.shape)
    num_neg_edges = np.max(neg_edges.shape)
    edges = torch.cat([pos_edges, neg_edges], dim=0)
    edge_loader = DataLoader(range(np.max(edges.shape)), batch_size, shuffle=False)

    results = torch.zeros([num_hops, num_hops, num_pos_edges + num_neg_edges]).to(device)
    for batch_edge_idx in tqdm(edge_loader):
        batch_edges = edges[batch_edge_idx]
        # batch_edges = edges
        rows, cols = batch_edges[:, 0], batch_edges[:, 1]
        for num_hop1, adj_graph1 in enumerate(adj_graphs):
            for num_hop2, adj_graph2 in enumerate(adj_graphs):
                torch.cuda.empty_cache()
                # import ipdb; ipdb.set_trace()
                # print(adj_graph1._nnz())
                # print(adj_graph2._nnz())
                # adj1_value = subgraph(rows, adj_graph1.indices(), adj_graph1.values(), relabel_nodes=False, num_nodes=num_nodes)
                
                # adj_graph1_index, adj_graph1_value = subgraph(rows, adj_graph1.indices(), adj_graph1.values(), relabel_nodes=False, num_nodes=num_nodes)   
                # adj_graph2_index, adj_graph2_value = subgraph(cols, adj_graph2.indices(), adj_graph2.values(), relabel_nodes=False, num_nodes=num_nodes)   
                
                _, adj_graph1_index, _, edge_mask1 = k_hop_subgraph(rows, 1, adj_graph1.indices(), directed=False,relabel_nodes=False, num_nodes=num_nodes)   
                adj_graph1_value = adj_graph1.values()[edge_mask1]
                adj_graph1_index, adj_graph1_value = to_undirected(adj_graph1_index, adj_graph1_value)
                adj_graph1_index, adj_graph1_value = remove_self_loops(adj_graph1_index, adj_graph1_value)
                
                _, adj_graph2_index, _, edge_mask2 = k_hop_subgraph(cols, 1, adj_graph2.indices(), directed=False, relabel_nodes=False, num_nodes=num_nodes)   
                adj_graph2_value = adj_graph2.values()[edge_mask2]
                adj_graph2_index, adj_graph2_value = to_undirected(adj_graph2_index, adj_graph2_value)
                adj_graph2_index, adj_graph2_value = remove_self_loops(adj_graph2_index, adj_graph2_value)
                
                result_index, result_value = torch_sparse.spspmm(adj_graph1_index, adj_graph1_value, adj_graph2_index, adj_graph2_value, num_nodes, num_nodes, num_nodes, coalesced=True)
                # i, j, v

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

                results[num_hop1, num_hop2, batch_edge_idx] += values
                torch.cuda.empty_cache()
                # print(results.sum().item())                
    results = torch.split(results, [num_pos_edges, num_neg_edges], dim=-1)
    pos_results, neg_results = results[0], results[1]
    # import ipdb; ipdb.set_trace()
    return pos_results, neg_results
    
    


@torch.no_grad()
def get_adj_indices_with_value(adjs, adjs_indices, device):
    device = torch.device("cpu")
    adj_valued_indices = []
    for adj, adj_indices in tqdm(zip(adjs, adjs_indices)):
        torch.cuda.empty_cache()
        # import ipdb; ipdb.set_trace()
        adj, adj_indices = adj.to(device), adj_indices.to(device)
        adj = adj.coalesce()
        tmp = torch.sparse_coo_tensor(adj_indices, torch.ones([np.max(adj_indices.shape)]).to(device), adj.shape)
        adj_tmp = adj - tmp
        adj_tmp = adj_tmp.coalesce()
        values = adj.values()[adj.values() != adj_tmp.values()] 
        # import ipdb; ipdb.set_trace()
        adj_valued_indices.append(torch.sparse_coo_tensor(adj_indices, values, adj.shape).coalesce())
        
    return adj_valued_indices


@torch.no_grad()
def flatten_neg_edges(origin_neg_edges, is_remove_redudant):
    # for heart method, after flatten, remove the redudant edges 
    device = origin_neg_edges.device
    num_nodes = torch.max(origin_neg_edges).item() + 1
    dims = len(origin_neg_edges.shape)
    if dims == 2:
        return origin_neg_edges
    else:
        # num_pos, num_neg_per, 2
        neg_edges = torch.flatten(origin_neg_edges, end_dim=-2)
        if is_remove_redudant: neg_edges, _ = torch_sparse.coalesce(neg_edges, torch.ones([np.max(neg_edges.shape)]).to(device), num_nodes, num_nodes)

        return neg_edges

@torch.no_grad()
def recover_neg_edges(neg_edges, num_pos):
    return torch.reshape(neg_edges, [num_pos, neg_edges.shape[0] // num_pos, 2])

@torch.no_grad()
def sparse_matrix_reduce_sum(adj1, adj2, batch_size):
    # TODO: cuda version
    device = adj1.device
    elements_per_batch1 = int(batch_size)
    elements_per_batch2 = int(batch_size)

    # Split the indices and values into batches
    index1_batches = torch.split(adj1.indices(), elements_per_batch1, dim=1)
    value1_batches = torch.split(adj1.values(), elements_per_batch1)
    index2_batches = torch.split(adj2.indices(), elements_per_batch2, dim=1)
    value2_batches = torch.split(adj2.values(), elements_per_batch2)

    # Create batched sparse matrices
    batched_sum = 0
    for index1_batch, value1_batch in zip(index1_batches, value1_batches):
        for index2_batch, value2_batch in zip(index2_batches, value2_batches):
            matrix1 = torch.sparse_coo_tensor(index1_batch, value1_batch, size=adj1.shape).to(device)
            matrix2 = torch.sparse_coo_tensor(index2_batch, value2_batch, size=adj2.shape).to(device)
            batched_sum += torch.sum(torch.sparse.mm(matrix1, matrix2).values()).item()
    
    return batched_sum
    
@torch.no_grad()
def np_matrix_multiply(edge_index, num_hops=3):
    num_edges = np.max(edge_index.shape)
    num_nodes = torch.max(edge_index).item() + 1
    edge_weight = torch.ones(num_edges).to(torch.int16).numpy()   #.to(torch.float)
    # import ipdb; ipdb.set_trace()
    adj = csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
    origin_adj = adj.copy()
    for i in range(num_hops):
        adj = adj.dot(origin_adj)
    
    '''    
    original_edge_index, original_edge_weight = edge_index.clone(), edge_weight.clone()
    
    import ipdb; ipdb.set_trace()
    for i in range(num_hops):
        edge_index, edge_weight = torch_sparse.spspmm(edge_index, edge_weight, 
            original_edge_index, original_edge_weight, num_nodes, num_nodes, num_nodes, coalesced=True)
    '''
    # import ipdb; ipdb.set_trace()
    # print()

'''
def set2tensor(adj):
    return torch.tensor(list(adj)).t()
    
def tensor2set(adj):
    return set(map(tuple, adj.indices().t().cpu().tolist()))
'''

@torch.no_grad()
def set2tensor(adj):
    return torch.tensor(list(adj)).t()
    
@torch.no_grad()
def tensor2set(adj):
    return set(zip(adj.indices()[0].cpu().numpy(), adj.indices()[1].cpu().numpy()))

@torch.no_grad()
def neighborhood(adj, nodes):
    neighbor_nodes = adj[nodes].to_dense()