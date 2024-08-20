import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import degree, mask_to_index, to_undirected, to_dense_adj, add_remaining_self_loops, dense_to_sparse, to_scipy_sparse_matrix, remove_self_loops, add_self_loops, k_hop_subgraph, contains_self_loops
from torch_scatter import scatter_add
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from torch import sparse
import torch_sparse
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

'''
In this file, we conduct functions on measuring the 
    1. homophily measurement (before and after aggregation)
    2. network property (typically network hetedrophily )
    3. also the labal homophily ratio related with the plantoid datasets.

'''


# this function contains different measurement on the homophily and heterophily with difference 


# This function should support calculation on all kinds of edges, but positive edges and negative edges should calculated seperately
@torch.no_grad()
def feature_homophily_ratio(args, dataset, edge_index, pos_links, neg_links, dis_func_name="cos", is_norm=False, is_feature_norm=False, predefine_group="D2AD2", edge_values=None, link_values=None, mask=None):
    # TODO: the normalization are majorly put on the front
    # edge_index is the existing edge for training, also can be utilized for inference
    # links are the edges that we need to calculate the homophily ratio
    num_edges = np.max(edge_index.shape)
    edge_index = edge_index.T
    edge_index, _ = remove_self_loops(edge_index)
    
    # is_feature_norm = True
    hidden = dataset.data.x
    num_nodes = hidden.shape[0]
    if is_feature_norm:  hidden /= torch.norm(hidden, dim=-1, keepdim=True)
    edge_index, edge_values = get_adjacent_matrix(edge_index,  predefine_group)
    
    # if args.num_hops == 0:
    #     print(torch.sum(hidden[0][:200]))
        # import ipdb; ipdb.set_trace()
    # citation2
    with torch.no_grad():
        for hop in range(args.num_hops):
            hidden = hidden.to(torch.float32)
            torch.cuda.empty_cache()
            # spare directly
            # import ipdb; ipdb.set_trace()
            A = torch.sparse_coo_tensor(edge_index, edge_values, (num_nodes, num_nodes))
            hidden = torch.sparse.mm(A, hidden)
            # torch_sparse.spmm(edge_index, edge_values, num_nodes, num_nodes, hidden)
    # if args.num_hops == 0:
    #     print(hidden[0][:10])
    
    # TODO: need to think about the normalized version, current, we do not have normalization 
    '''
    # take subgraph into consideration
    
    if mask != None:
        mask_idx = mask_to_index(mask)
        _, edge_index, _, edge_mask = k_hop_subgraph(mask_idx, 1, edge_index, )
        edge_value = edge_value[edge_mask]
    '''

    # generate a new dgl graph since it is more adapted to the edge operation and comparison
    links_dict = {"pos": pos_links, "neg": neg_links}
    results_dict, segregation_dict = {}, {}
    dis_func = select_dis_func(dis_func_name)
    for key in links_dict.keys():
        links = links_dict[key]
        # is_norm = True
        hidden = hidden / torch.norm(hidden, dim=-1, keepdim=True) if is_norm else hidden
        torch.cuda.empty_cache()
        num_links = np.max(links.shape)
        max_batch_size = 40000
        
        if max_batch_size >= num_links:
            results = dis_func(hidden[links[:, 0]], hidden[links[:, 1]]).to(torch.float32)
        else:
            for i in range(0, num_links, max_batch_size):
                if i+max_batch_size < num_links:
                    batch_links = links[i:i+max_batch_size]
                else:
                    batch_links = links[i:]
                    
                batch_hidden1 = hidden[batch_links[:, 0]]
                batch_hidden2 = hidden[batch_links[:, 1]]
                cur_results = dis_func(batch_hidden1, batch_hidden2).to(torch.float32)
                if i == 0:
                    results = cur_results
                else:
                    results = torch.cat((results, cur_results), dim=0)
        # import ipdb; ipdb.set_trace()
        segregation_ratio = torch.mean(results)
        
        '''
        if link_values is None:
            link_values = torch.ones((links.size(0),), device=links.device)
                
        link_graph = dgl.graph((links[:, 0], links[:, 1]), num_nodes=num_nodes)
        link_graph.ndata['h'] = hidden / torch.norm(hidden, -1, keepdim=True) if is_norm else hidden
        link_graph.ndata['d'] = torch.unsqueeze(degrees, dim=-1)
        dis_func = select_dis_func(dis_func_name)
        # import ipdb; ipdb.set_trace()
        link_graph.apply_edges(lambda edges: {'e': dis_func(edges.src['h'], edges.dst['h'])})
        # torch.sum(torch.square(edges.src['h'] / torch.sqrt(edges.src['d']) - edges.dst['h'] / torch.sqrt(edges.dst['d'])), dim=-1)
        # graph.apply_edges(lambda edges: {'x': torch.sum(torch.square(edges.src['h'] / torch.sqrt(edges.src['d']) - edges.dst['h'] / torch.sqrt(edges.dst['d'])), dim=-1)})

        results = link_graph.edata['e']
        segregation_ratio = torch.mean(link_graph.edata['e'])
        '''
        results_dict[key] = results 
        segregation_dict[key] = segregation_ratio
    
    return results_dict["pos"], results_dict["neg"]


@torch.no_grad()
def homophily_logistic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, dis_func_name="cos", is_norm=False, is_feature_norm=False, predefine_group="D2AD2", edge_values=None, link_values=None, mask=None):
    pos_valid_features, neg_valid_features = feature_homophily_ratio(args, dataset, known_links, valid_pos_links, valid_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)
    pos_test_features, neg_test_features = feature_homophily_ratio(args, dataset, known_links, test_pos_links, test_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)

    pos_valid_labels, neg_valid_labels = torch.ones(pos_valid_features.shape[0]), torch.zeros(neg_valid_features.shape[0])
    logits_model = LogisticRegression(random_state=0).fit(np.concatenate([pos_valid_features, neg_valid_features], axis=0), np.expand_dims(np.concatenate([pos_valid_labels, neg_valid_labels], axis=0), axis=-1))
    pos_test_results = logits_model.predict_proba(pos_test_features)
    neg_test_results = logits_model.predict_proba(neg_test_features)
   
    # the probability of positive 
    pos_test_results = pos_test_results[:, 1]
    neg_test_results = neg_test_results[:, 1]
    
     
    return pos_test_results, neg_test_results





@torch.no_grad()
def feature_importance_homo_logistic(pos_valid_features, neg_valid_features, pos_test_features, neg_test_features):
    pos_valid_labels, neg_valid_labels = torch.ones(pos_valid_features.shape[0]), torch.zeros(neg_valid_features.shape[0])
    features = np.concatenate([pos_valid_features, neg_valid_features], axis=0)
    features = np.where(~np.isnan(features) , features, 0) # != np.nan
    # import ipdb; ipdb.set_trace()
    # origin_features = features
    # replace nan with 0
    features = features / np.linalg.norm(features, axis=0, keepdims=True)
    
    if np.any(np.isnan(features)):
        import ipdb; ipdb.set_trace()
    logits_model = LogisticRegression(random_state=0).fit(features, np.expand_dims(np.concatenate([pos_valid_labels, neg_valid_labels], axis=0), axis=-1))
    
    weights = np.squeeze(logits_model.coef_)
    # check the weight of logistic
    # import ipdb; ipdb.set_trace()
    
    features = np.concatenate([pos_test_features, neg_test_features], axis=0)
    
    features = features / np.linalg.norm(features, axis=0, keepdims=True)   
    features = np.where(~np.isnan(features) , features, 0) # != np.nan
    
    # if np.any(np.isnan(features)):
    #     import ipdb; ipdb.set_trace()
    
    num_pos_edges = pos_test_features.shape[0]
    features = np.split(features, [num_pos_edges], axis=0)

    pos_test_features, neg_test_features = features[0], features[1]
    pos_test_results = logits_model.predict_proba(pos_test_features)
    neg_test_results = logits_model.predict_proba(neg_test_features)

    # the probability of positive 
    pos_test_results = pos_test_results[:, 1]
    neg_test_results = neg_test_results[:, 1]
    
    return pos_test_results, neg_test_results, weights
    


# This function should support calculation on all kinds of edges, but positive edges and negative edges should calculated seperately
@torch.no_grad()
def logits_homophily_ratio_old(args, dataset, edge_index, pos_links, neg_links, dis_func_name="cos", is_norm=False, is_feature_norm=False, predefine_group="D2AD2", edge_values=None, link_values=None, mask=None):
    # TODO: current version does not have training procedure, it does not remove the train edge during aggregation
    # TODO: this somehow lead to data leakage
    # edge_index is the existing edge for training, also can be utilized for inference
    # links are the edges that we need to calculate the homophily ratio
    num_edges = np.max(edge_index.shape)
    device = edge_index.device
    edge_index = edge_index.T
    edge_index, _ = remove_self_loops(edge_index)
    
    hidden = dataset.data.x
    num_nodes = hidden.shape[0]
    if is_feature_norm:  hidden /= torch.norm(hidden, dim=-1, keepdim=True)
    edge_index, edge_values = get_adjacent_matrix(edge_index,  predefine_group)
    
    for hop in range(args.num_hops):
        hidden = torch_sparse.spmm(edge_index, edge_values, num_nodes, num_nodes, hidden)

    # hidden = hidden / torch.norm(hidden, dim=-1, keepdim=True)
    pos_train_link = edge_index
    neg_train_link = torch.randint(0, num_nodes, pos_train_link.size(), dtype=torch.long, device=device)
    
    pos_train_hidden =  (hidden[pos_train_link[:, 0]] * hidden[pos_train_link[:, 1]]).to(torch.float32).cpu().numpy()
    neg_train_hidden =  (hidden[neg_train_link[:, 0]] * hidden[neg_train_link[:, 1]]).to(torch.float32).cpu().numpy()
    train_hidden = np.concatenate((pos_train_hidden, neg_train_hidden), axis=0)
    
    pos_train_label = np.ones((pos_train_link.size(0),), dtype=np.int32)
    neg_train_label = np.zeros((neg_train_link.size(0),), dtype=np.int32)
    train_label = np.concatenate((pos_train_label, neg_train_label), axis=0)
    
    logits_model = LogisticRegression(random_state=0).fit(train_hidden, train_label)
    
    # generate a new dgl graph since it is more adapted to the edge operation and comparison
    links_dict = {"pos": pos_links, "neg": neg_links}
    results_dict = {}
    for key in links_dict.keys():
        links = links_dict[key]
        # TODO: Do we need to open the normalization here
        # import ipdb; ipdb.set_trace()
        # print()
        # hidden = hidden / torch.norm(hidden, dim=-1, keepdim=True) if is_norm else hidden
        # hidden = hidden / torch.norm(hidden, dim=-1, keepdim=True)
        torch.cuda.empty_cache()
        test_hidden = (hidden[links[:, 0]] * hidden[links[:, 1]]).cpu().numpy()
        results = logits_model.predict_proba(test_hidden)
        results_dict[key] = torch.tensor(results).to(device) 
        
    return results_dict["pos"], results_dict["neg"]



@torch.no_grad()
def label_homophily_ratio(args, dataset, edge_index, pos_links, neg_links, label_name="origin", num_labels=7, is_norm=False, is_feature_norm=False, predefine_group="D2AD2", edge_values=None, link_values=None, mask=None):
    # TODO: the normalization are majorly put on the front
    # edge_index is the existing edge for training, also can be utilized for inference
    # links are the edges that we need to calculate the homophily ratio
    data = dataset.data
    device = data.edge_index.device
    num_edges = np.max(edge_index.shape)
    edge_index = edge_index.T
    edge_index, _ = remove_self_loops(edge_index)
    
    hidden = data.x
    num_nodes = hidden.shape[0]
    if is_feature_norm:  hidden /= torch.norm(hidden, dim=-1, keepdim=True)
    edge_index, edge_values = get_adjacent_matrix(edge_index,  predefine_group)
    
    for hop in range(args.num_hops):
        hidden = torch_sparse.spmm(edge_index, edge_values, num_nodes, num_nodes, hidden)
    # this is not so useful
    degrees = calculate_degree(edge_index, edge_values, num_nodes)
    
    # generate label information
    assert label_name in ["origin", "kmeans", "GMM", "SC"]
    if label_name == "origin":
        assert args.dataset_name in ["Cora", "CiteSeer", "PubMed"]
        label = data.y
    else:
        label = cluster_assign_label(hidden, num_labels, label_name, device)
        label = torch.tensor(label).to(device)
    label = torch.squeeze(label)   
    # import ipdb; ipdb.set_trace()

    # generate a new dgl graph since it is more adapted to the edge operation and comparison
    links_dict = {"pos": pos_links, "neg": neg_links}
    results_dict, segregation_dict = {}, {}
    for key in links_dict.keys():
        links = links_dict[key]
        if link_values is None:
            link_values = torch.ones((links.size(0),), device=links.device)
        
        results = (label[links[:, 0]] == label[links[:, 1]]).to(torch.int32)
        segregation_ratio = torch.sum(results) / results.shape[0]
        
        '''
        link_graph = dgl.graph((links[:, 0], links[:, 1]), num_nodes=num_nodes)
        # link_graph.ndata['y'] = torch.unsqueeze(label, dim=-1)
        link_graph.ndata['y'] = label
        # import ipdb; ipdb.set_trace(())
        link_graph.apply_edges(lambda edges: {'e': edges.src['y'] == edges.dst['y']})
        link_graph.edata['e'] = link_graph.edata['e'].to(torch.int)
        results = link_graph.edata['e']
        segregation_ratio = torch.sum(link_graph.edata['e']) / link_graph.edata['e'].shape[0]
        '''
        results_dict[key] = torch.squeeze(results) 
        segregation_dict[key] = segregation_ratio
    
    return results_dict["pos"], results_dict["neg"]




@torch.no_grad()
def select_dis_func(dis_func_name):
    if dis_func_name == "jaccard":
        return jaccard
    elif dis_func_name == "l2":
        return l2
    elif dis_func_name == "cos":
        return cos


@torch.no_grad()
def cluster_assign_label(hidden, num_clusters, method_name, device):
    # TODO: Do not know just cluster on the original space or multiple times
    hidden = hidden.cpu().numpy()
    if method_name == "kmeans":
        # model = KMeans(n_clusters=num_clusters)
        # model.fit(hidden)
        # labels = model.labels_
        # centers = model.cluster_centers_
        
        hidden = torch.tensor(hidden).to(device)
        labels, centroids = kmeans(hidden, num_clusters, max_iters=100, tol=1e-4)       
        # model = KMEANS(n_clusters=num_clusters, max_iter=None, verbose=True, device=device)
        # labels = model.fit(hidden)
        labels = labels.cpu().numpy()
    elif method_name == "GMM":
        model = GaussianMixture(n_components=num_clusters)
        model.fit(hidden)
        labels = model.predict(hidden)
    elif method_name == "SC":
        model = SpectralClustering(n_clusters=num_clusters)
        model.fit(hidden)
        labels = model.labels_
    else:
        exit("do not find match cluster algorithm")
    return labels



@torch.no_grad()
def jaccard(tensor1, tensor2):
    intersection = torch.logical_and(tensor1, tensor2)
    union = torch.logical_or(tensor1, tensor2)

    intersection_count = torch.sum(intersection, dim=-1, keepdim=False)
    union_count = torch.sum(union, dim=-1, keepdim=False)

    jaccard_similarity = intersection_count / (union_count + 1e-4)
    # if torch.any(torch.isnan(jaccard_similarity)):
    #     import ipdb; ipdb.set_trace()
    return jaccard_similarity

@torch.no_grad()
def l2(tensor1, tensor2):
    tensor1 = tensor1.to(torch.float32)
    tensor2 = tensor2.to(torch.float32)

    return 1.0 / (1.0 + torch.norm(tensor1 - tensor2, dim=-1))
    # TODO: return 1.0 - torch.dist(tensor1, tensor2)

@torch.no_grad()
def cos(tensor1, tensor2):
    return torch.sum(tensor1 * tensor2, dim=-1)
    # return torch.squeeze(F.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0)))

@torch.no_grad()
def calculate_degree(edge_index, edge_value, num_nodes):
    row, col = edge_index
    deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    
    return deg

# TODO: give prediction edges a homophily rank, use the rank to get the homophily ratio.

@torch.no_grad()
def calculate_hidden_sim(hidden, max_node_batch=1e6, is_cos=False):
    device = hidden.device
    # .type == 'cpu' else False
    similarity_sum = 0
    num_nodes = hidden.shape[0]
    max_node_batch = num_nodes if num_nodes < max_node_batch else max_node_batch

    for i in range(0, num_nodes, max_node_batch):
        for j in range(0, num_nodes, max_node_batch):
            range1 = torch.arange(i, i + max_node_batch) if i + max_node_batch < num_nodes else torch.arange(i, num_nodes)
            range2 = torch.arange(j, j + max_node_batch) if j + max_node_batch < num_nodes else torch.arange(j, num_nodes)
            range1, range2 = range1.to(device), range2.to(device)
            hidden1_batch, hidden2_batch = hidden[range1], hidden[range2]
            if is_cos:
                sim = 1 - F.cosine_similarity(hidden1_batch.unsqueeze(1), hidden2_batch.unsqueeze(0), dim=-1)
            else:
                sim = torch.cdist(hidden1_batch, hidden2_batch)
            similarity_sum += torch.sum(sim)
    similarity_mean = similarity_sum.item() / (num_nodes * num_nodes)
        
    return similarity_mean

@torch.no_grad()
def aggregation(hidden, edge_index,  links, batch_size=1000, num_hops=0, is_feature_norm=False, predefine_group="D2AD2"):
    num_nodes = hidden.shape[0]
    if is_feature_norm:  hidden /= torch.norm(hidden, dim=-1, keepdim=True)
    edge_index, edge_value = get_adjacent_matrix(edge_index,  predefine_group)
    
    for hop in range(num_hops):
        hidden = torch_sparse.spmm(edge_index, edge_value, num_nodes, num_nodes, hidden)

    return hidden

    '''
    link_loader = DataLoader(range(np.max(links.shape)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src_idx, dst_idx = edge_index[ind, 0], edge_index[ind, 1]
        # [num_node]
        hidden_src = torch.index_select(hidden, 0, src_idx)
        hidden_dst = torch.index_select(hidden, 0, dst_idx)

    
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    scores = np.concatenate(scores, 0)
    print(f'evaluated Common Neighbours for {len(scores)} edges')
    return torch.FloatTensor(scores), edge_index
    '''
    


@torch.no_grad()
def get_adjacent_matrix(edge_index,  predefine_group="D2AD2"):
    predefine_dict = {
        "A": {"a":0 , "m1":1, "m2":0, "m3":0, "e1":0, "e2":0, "e3":0},
        "D-A": {"a":1 , "m1":-1, "m2":0, "m3":1, "e1":0, "e2":0, "e3":0},
        "D+A": {"a":1, "m1":1, "m2":0, "m3":1, "e1":0, "e2":0, "e3":0},
        "I-D-1A": {"a":0, "m1":-1, "m2":1, "m3":0, "e1":-1, "e2":0, "e3":0},
        "I-D2AD2": {"a":1, "m1":0, "m2":-1, "m3":1, "e1":0, "e2":-0.5, "e3":-0.5},
        "D2AD2": {"a":0 , "m1":1, "m2":0, "m3":0, "e1":-0.5, "e2":-0.5, "e3":1},
        "D-1A": {"a":0, "m1":1, "m2":0, "m3":0, "e1":-1, "e2":0, "e3":0},
    }
    if predefine_group:
        predefine_params = predefine_dict[predefine_group]
    a, m1, m2, m3, e1, e2, e3 = predefine_params["a"], predefine_params["m1"], predefine_params["m2"], predefine_params["m3"], predefine_params["e1"], predefine_params["e2"], predefine_params["e3"]
        
    edge_index, _ =  remove_self_loops(edge_index)
    num_nodes = torch.max(edge_index).item() + 1
    device = edge_index.device
    edge_value = torch.ones((edge_index.size(1),), device=device)
    edge_index, edge_value = add_self_loops(edge_index, edge_value, fill_value=a, num_nodes=num_nodes)
    
    if predefine_group == "A":
        return edge_index, edge_value
    
            
    row, col = edge_index
    # deg = scatter_add(edge_value, row, dim=0, dim_size=num_nodes)
    deg = degree(row, num_nodes=num_nodes)
    left_deg = m2 * deg.pow(e2)
    left_deg[left_deg == float('inf')] = 0
    right_deg = deg.pow(e3)
    right_deg[right_deg == float('inf')] = 0
    self_deg = deg.pow(e1)
    self_deg[self_deg == float('inf')] = 0
    
    diag_edge_index = torch.stack([torch.arange(num_nodes).long(),torch.arange(num_nodes).long()]).reshape([2, -1]).cuda()
    identify_edge_value = torch.ones(num_nodes).cuda()
    normalized_edge_value = left_deg[row] * edge_value * right_deg[col]

    final_edge_index = torch.cat((edge_index, diag_edge_index, diag_edge_index), dim=-1)
    final_edge_value = torch.cat((normalized_edge_value, identify_edge_value * m3, self_deg * m1), dim=-1)
    
    final_edge_index, final_edge_value = torch_sparse.coalesce(final_edge_index, final_edge_value, m=num_nodes, n=num_nodes, op='sum')
        
    return final_edge_index, final_edge_value

'''
@torch.no_grad()
def label_homophily_ratio(dataset, hidden, edge_index, links, edge_values=None, link_values=None, mask=None):
    # edge_index is the existing edge for training, also can be utilized for inference
    # links are the edges that we need to calculate the homophily ratio
    data = dataset.data
    
    num_nodes, num_edges, num_links = hidden.shape[0], np.max(edge_index.shape), np.max(links.shape)
    links, _ =  remove_self_loops(links)
    edge_index, _ = remove_self_loops(edge_index)
        
    if link_values is None:
        link_values = torch.ones((links.size(1),), device=links.device)
    if edge_values is None:
        edge_values = torch.ones((edge_index.size(1),), device=edge_index.device)
    degrees = calculate_degree(edge_index, edge_values, num_nodes)
    
    # TODO: need to think about the normalized version, current, we do not have normalization 
    # take subgraph into consideration
    # if mask != None:
    #     mask_idx = mask_to_index(mask)
    #     _, edge_index, _, edge_mask = k_hop_subgraph(mask_idx, 1, edge_index, )
    #     edge_value = edge_value[edge_mask]

    # generate a new dgl graph since it is more adapted to the edge operation and comparison
    link_graph = dgl.graph((links[0], links[1]), num_nodes=num_nodes)
    link_graph.ndata['y'] = torch.unsqueeze(data.y, dim=-1)
    link_graph.ndata['d'] = torch.unsqueeze(link_deg, dim=-1)
    dis_func = select_dis_func(dis_func_name)
    graph.apply_edges(lambda edges: {'x': edges.src['y'] == edges.dst['y']})
    # torch.sum(torch.square(edges.src['h'] / torch.sqrt(edges.src['d']) - edges.dst['h'] / torch.sqrt(edges.dst['d'])), dim=-1)

    result = torch.sum(graph.edata['x']) / num_links 
    
    return result
'''



def kmeans(X, K, max_iters=100, tol=1e-4):
    N, D = X.size()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = X.to(device).to(torch.float32)

    # Randomly initialize centroids
    centroids = X[torch.randperm(N)[:K]]
    centroids_old = torch.zeros_like(centroids)

    for it in range(max_iters):
        # Compute distances from each point to each centroid
        dists = torch.cdist(X, centroids)
        
        # Assign labels based on closest centroid
        labels = torch.argmin(dists, dim=1)

        # Update centroids
        centroids_old = centroids.clone()
        for k in range(K):
            centroids[k] = X[labels == k].mean(dim=0)

        # Check for convergence
        if torch.norm(centroids - centroids_old) < tol:
            break

    return labels, centroids


class KMEANS:
    def __init__(self, n_clusters, max_iter=None, verbose=True, device=torch.device("cpu")):
        # self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        # print(init_row.shape)    # shape 10
        init_points = x[init_row]
        # print(init_points.shape) # shape (10, 2048)
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            # if self.verbose:
            #     print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break

            self.count += 1

        return self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        # print(labels.shape)  # shape (250000)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        # print(dists.shape)   # shape (0, 10)
        for i, sample in enumerate(x):
            # print(self.centers.shape) # shape(10, 2048)
            # print(sample.shape)       # shape 2048
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            # print(dist.shape)         # shape 10
            labels[i] = torch.argmin(dist)
            # print(labels.shape)       # shape 250000
            # print(labels[:10])
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
            # print(dists.shape)        # shape (1,10)
            # print('*')
        self.labels = labels           # shape 250000
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists              # 250000, 10
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device) # shape (0, 250000)
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))  # 10, 2048
        self.centers = centers  # shape (10, 2048)

    def representative_sample(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        # print(self.dists.shape)
        self.representative_samples = torch.argmin(self.dists, 1)
        # print(self.representative_samples.shape)  # shape 250000
        # print('*')
        return self.representative_samples
