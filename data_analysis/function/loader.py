from math import inf
import random

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
from torch_geometric.utils import (negative_sampling, add_self_loops)
from torch_sparse import coalesce
from tqdm import tqdm
import scipy.sparse as ssp
from scipy.sparse.csgraph import shortest_path

from utils import get_src_dst_degree, neighbors, get_pos_neg_edges
# from labelling_tricks import drnl_node_labeling, de_node_labeling, de_plus_node_labeling
import scipy.io


class WholeDataset(Dataset):
    def __init__(self, data, num_hops, percent=1., use_coalesce=False,
                 ratio_per_hop=1.0, max_nodes_per_hop=None, max_dist=1000, directed=False, sign=False, k=None,
                 **kwargs):
        self.data = data
        self.num_hops = num_hops
        self.percent = percent
        self.use_coalesce = use_coalesce
        # self.node_label = node_label
        self.ratio_per_hop = ratio_per_hop
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_dist = max_dist
        self.directed = directed
        self.sign = sign
        self.k = k
        root="check"
        super(WholeDataset, self).__init__(root)

        self.links = data.edge_index.T.cpu().numpy().tolist()

        if self.use_coalesce:  # compress mutli-edge into edge with weight
            self.data.edge_index, self.data.edge_weight = coalesce(
                self.data.edge_index, self.data.edge_weight,
                self.data.num_nodes, self.data.num_nodes)

        if 'edge_weight' in self.data:
            edge_weight = self.data.edge_weight.view(-1)
        else:
            edge_weight = torch.ones(self.data.edge_index.size(1), dtype=int)
        self.A = ssp.csr_matrix(
            (edge_weight.cpu(), (self.data.edge_index[0].cpu(), self.data.edge_index[1].cpu())),
            shape=(self.data.num_nodes, self.data.num_nodes)
        )
        if self.directed:
            self.A_csc = self.A.tocsc()
        else:
            self.A_csc = None

    def len(self):
        return len(self.links)

    def get(self, idx):
        src, dst = self.links[idx]
        src_degree, dst_degree = get_src_dst_degree(src, dst, self.A, self.max_nodes_per_hop)
        x = self.data.x
        nodes, subgraph, dists, node_features, y = k_hop_subgraph(src, dst, self.num_hops, self.A, self.ratio_per_hop,
                             self.max_nodes_per_hop, node_features=x,
                             directed=self.directed, A_csc=self.A_csc)
        dists = tradic_counter(subgraph, 0, 1, max_dist=self.max_dist)
        # subgraphs = construct_pyg_graph(node_ids, adj, dists, node_features, y, self.max_dist, src_degree, dst_degree)
        
        return dists



def tradic_counter(adj, src, dst, is_include_target=False, max_dist=100):
    """
    The heuristic proposed in "Link prediction based on graph neural networks". It is an integer value giving the 'distance'
    to the (src,dst) edge such that src = dst = 1, neighours of dst,src = 2 etc. It implements
    z = 1 + min(d_x, d_y) + (d//2)[d//2 + d%2 - 1] where d = d_x + d_y
    z is treated as a node label downstream. Even though the labels measures of distance from the central edge, they are treated as
    categorical objects and embedded in an embedding table of size max_z * hidden_dim
    @param adj:
    @param src:
    @param dst:
    @return:
    """
    src, dst = (dst, src) if src > dst else (src, dst)

    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]

    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]

    if is_include_target:
        adj_wo_src = adj
        adj_wo_dst = adj

    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)
    # dist2src[dist2src > max_dist] = -1

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)
    # dist2dst[dist2dst > max_dist] = -1

    return (dist2src, dist2dst)



def get_datasets(dataset, train_data, args):
    sample = 'all' if not args.sample_size else args.sample_size
    use_coalesce = True if args.dataset_name == 'ogbl-collab' else False
    # get percents used only for naming the SEAL dataset files and caching
    path = "check"
    directed = False
    train_percent, val_percent, test_percent = 1 - (args.val_pct + args.test_pct), args.val_pct, args.test_pct
    # just a placeholder here
    
    dataset = WholeDataset(
        train_data,
        num_hops=args.num_hops,
        percent=train_percent,
        split='train',
        use_coalesce=use_coalesce,
        ratio_per_hop=args.ratio_per_hop,
        max_nodes_per_hop=args.max_nodes_per_hop,
        max_dist=args.max_dist,
        directed=directed,
        sign=args.model == 'sign',
        k=args.sign_k
    )
    return dataset


def upload_split(data_name, train_data, val_data, test_data):
    name_dict = {"Cora": "cora", "Citeseer": "citeseer", "Pubmed": "pubmed"}
    data_name = name_dict[data_name]
    device = train_data.x.device
    dir_path = "/egr/research-dselab/haitaoma/LinkPrediction/HeaRT/dataset/"
    
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
   
    ##############
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []
    node_set = set()
    
    for split in ['train', 'test', 'valid']:
        path = dir_path+ '{}/{}_pos.txt'.format(data_name, split)

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                
            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    # print('the number of nodes in ' + data_name + ' is: ', num_nodes)

    for split in ['test', 'valid']:

        path = dir_path + '{}/{}_neg.txt'.format(data_name, split)

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            # if sub == obj:
            #     continue
            
            if split == 'valid':  valid_neg.append((sub, obj))
               
            if split == 'test': test_neg.append((sub, obj))

    train_pos = torch.tensor(train_pos).to(device)
    train_neg = torch.tensor(train_neg).to(device)

    valid_pos = torch.tensor(valid_pos).to(device)
    valid_neg =  torch.tensor(valid_neg).to(device)

    test_pos =  torch.tensor(test_pos).to(device)
    test_neg =  torch.tensor(test_neg).to(device)

    idx = torch.randperm(train_pos.size(0)).to(device)
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos[idx]
    
    if len(train_neg.shape) > 1:
        train_data.edge_label_index = torch.cat([train_pos, train_neg], dim=0).permute([1, 0])
        train_data.edge_label = torch.cat([torch.ones(np.max(train_pos.shape)), torch.zeros(np.max(train_neg.shape))], dim=-1).to(device)
    else:
        train_data.edge_label_index = train_pos.permute([1, 0])
    train_data.edge_label = torch.ones(np.max(train_pos.shape)).to(device)
    val_data.edge_label_index = torch.cat([valid_pos, valid_neg], dim=0).permute([1, 0])
    val_data.edge_label = torch.cat([torch.ones(np.max(valid_pos.shape)), torch.zeros(np.max(valid_neg.shape))], dim=-1).to(device)
    test_data.edge_label_index = torch.cat([test_pos, test_neg], dim=0).permute([1, 0])
    test_data.edge_label = torch.cat([torch.ones(np.max(test_pos.shape)), torch.zeros(np.max(test_neg.shape))], dim=-1).to(device)

    # train_edge_label_index = torch.cat() 
    return train_data, val_data, test_data

def load_plantoid_heart_edge(args, device):
    data_name = args.dataset_name
    name_dict = {"Cora": "cora", "Citeseer": "citeseer", "Pubmed": "pubmed"}
    # the first is for all dataset
    if data_name in name_dict.keys():
        data_name = name_dict[data_name]
    
    dir_path = "/egr/research-dselab/haitaoma/LinkPrediction/HeaRT/dataset"
    
    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:
        path = dir_path+ '/{}/{}_pos.txt'.format(data_name, split)
    
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                
            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    # print('the number of nodes in ' + data_name + ' is: ', num_nodes)

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

    with open(f'{dir_path}/{data_name}/heart_valid_samples.npy', "rb") as f:
        valid_neg = np.load(f)
        valid_neg = torch.from_numpy(valid_neg)
    with open(f'{dir_path}/{data_name}/heart_test_samples.npy', "rb") as f:
        test_neg = np.load(f)
        test_neg = torch.from_numpy(test_neg)
    
    train_pos = torch.tensor(train_pos).to(device)
    train_neg = torch.tensor(train_neg).to(device)

    valid_pos = torch.tensor(valid_pos).to(device)
    valid_neg =  torch.tensor(valid_neg).to(device)

    test_pos =  torch.tensor(test_pos).to(device)
    test_neg =  torch.tensor(test_neg).to(device)

    return train_pos, train_neg, valid_pos, valid_neg, test_pos, test_neg



def load_ogb_heart_edge(args, device):
    dir_path = "/egr/research-dselab/haitaoma/LinkPrediction/HeaRT/dataset"
    data_name = args.dataset_name
    with open(f'{dir_path}/{data_name}/heart_valid_samples.npy', "rb") as f:
        valid_neg = np.load(f)
        valid_neg = torch.from_numpy(valid_neg)
    with open(f'{dir_path}/{data_name}/heart_test_samples.npy', "rb") as f:
        test_neg = np.load(f)
        test_neg = torch.from_numpy(test_neg)

    return valid_neg, test_neg
    


def load_social_data(data_name):
    # dataset list: blog facebook flickr googleplus twitter
    data = scipy.io.loadmat(f"/egr/research-dselab/haitaoma/LinkPrediction/williamweiwu.github.io/Graph_Network_Embedding/HashGNN/data/{data_name}/{data_name}.mat")
    # 'attributes'  'labels' 'network' 'testGraph' 'trainGraph'
    import ipdb; ipdb.set_trace()
    print()

    # with open("/egr/research-dselab/haitaoma/LinkPrediction/williamweiwu.github.io/Graph_Network_Embedding/HashGNN/data/blog/blog.adjlist.0.5", "r") as f:
    #     f.readline()
    #     import ipdb; ipdb.set_trace()
    #     print()

    '''
    with open("/egr/research-dselab/haitaoma/LinkPrediction/williamweiwu.github.io/Graph_Network_Embedding/HashGNN/data/blog/blog.adjlist.0.5", "r") as f:
        f.readline()
        import ipdb; ipdb.set_trace()
        print()
    '''    # each line corresponding to one line adjacent matrix.
    
    
    # with open("/egr/research-dselab/haitaoma/LinkPrediction/williamweiwu.github.io/Graph_Network_Embedding/HashGNN/data/blog/blog_0.5.mat", "rb") as f:
    #     f.readline()
        


def get_train_val_test_links(dataset, train_data, val_data, test_data, args):
    sample = 'all' if not args.sample_size else args.sample_size
    path = "check"
    # f'{dataset.root}_seal_{sample}_hops_{args.num_hops}_maxdist_{args.max_dist}_mnph_{args.max_nodes_per_hop}{args.data_appendix}'
    use_coalesce = True if args.dataset_name == 'ogbl-collab' else False
    # get percents used only for naming the SEAL dataset files and caching
    train_percent, val_percent, test_percent = 1 - (args.val_pct + args.test_pct), args.val_pct, args.test_pct
    # probably should be an attribute of the dataset and not hardcoded
    directed = False
    pos_train_edge, neg_train_edge = get_pos_neg_edges(train_data)
    pos_val_edge, neg_val_edge = get_pos_neg_edges(val_data)
    pos_test_edge, neg_test_edge = get_pos_neg_edges(test_data)
    # print(
    #     f'before sampling, considering a superset of {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
    #     f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
    #     f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')

    # pos_train_edge = sample_data(pos_train_edge, args.train_samples)
    # neg_train_edge = sample_data(neg_train_edge, args.train_samples)
    # pos_val_edge = sample_data(pos_val_edge, args.val_samples)
    # neg_val_edge = sample_data(neg_val_edge, args.val_samples)
    # pos_test_edge = sample_data(pos_test_edge, args.test_samples)
    # neg_test_edge = sample_data(neg_test_edge, args.test_samples)

    # print(
    #     f'after sampling, using {pos_train_edge.shape[0]} pos, {neg_train_edge.shape[0]} neg train edges '
    #     f'{pos_val_edge.shape[0]} pos, {neg_val_edge.shape[0]} neg val edges '
    #     f'and {pos_test_edge.shape[0]} pos, {neg_test_edge.shape[0]} neg test edges for supervision')

    return pos_train_edge, neg_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge 

def sample_data(data, sample_arg):
    if sample_arg <= 1:
        samples = int(sample_arg * len(data))
    elif sample_arg != inf:
        samples = int(sample_arg)
    else:
        samples = len(data)
    if samples != inf:
        sample_indices = torch.randperm(len(data))[:samples]
    return data[sample_indices]




def k_hop_subgraph(src, dst, num_hops, A, sample_ratio=1.0,
                   max_nodes_per_hop=None, node_features=None,
                   y=1, directed=False, A_csc=None):
    """
    Extract the k-hop enclosing subgraph around link (src, dst) from A.
    it permutes the node indices so the returned subgraphs are not immediately recognisable as subgraphs and it is not
    parallelised.
    For directed graphs it adds both incoming and outgoing edges in the BFS equally and then for the target edge src->dst
    it will also delete any dst->src edge, it's unclear if this is a feature or a bug.
    :param src: source node for the edge
    :param dst: destination node for the edge
    :param num_hops:
    :param A:
    :param sample_ratio: This will sample down the total number of neighbours (from both src and dst) at each hop
    :param max_nodes_per_hop: This will sample down the total number of neighbours (from both src and dst) at each hop
                            can be used in conjunction with sample_ratio
    :param node_features:
    :param y:
    :param directed:
    :param A_csc:
    :return:
    """
    nodes = [src, dst]
    dists = [0, 0]
    visited = set([src, dst])
    fringe = set([src, dst])
    for hop in range(1, num_hops + 1):
        if not directed:
            # import ipdb; ipdb.set_trace()
            fringe = neighbors(fringe, A)
        else:
            out_neighbors = neighbors(fringe, A)
            in_neighbors = neighbors(fringe, A_csc, False)
            fringe = out_neighbors.union(in_neighbors)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio * len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [hop] * len(fringe)
    # this will permute the rows and columns of the input graph and so the features must also be permuted
    subgraph = A[nodes, :][:, nodes]

    # Remove target link between the subgraph. Works as the first two elements of nodes are the src and dst node
    # this can throw warnings as csr sparse matrices aren't efficient for removing edges, but these graphs are quite sml
    subgraph[0, 1] = 0
    subgraph[1, 0] = 0

    if isinstance(node_features, list):
        node_features = [feat[nodes] for feat in node_features]
    elif node_features is not None:
        node_features = node_features[nodes]

    return nodes, subgraph, dists, node_features, y

def construct_pyg_graph(node_ids, adj, dists, node_features, y, node_label='drnl', max_dist=1000, src_degree=None,
                        dst_degree=None):
    """
    Constructs a pyg graph for this subgraph and adds an attribute z containing the node_label
    @param node_ids: list of node IDs in the subgraph
    @param adj: scipy sparse CSR adjacency matrix
    @param dists: an n_nodes list containing shortest distance (in hops) to the src or dst node
    @param node_features: The input node features corresponding to nodes in node_ids
    @param y: scalar, 1 if positive edges, 0 if negative edges
    @param node_label: method to add the z attribute to nodes
    @return:
    """
    u, v, r = ssp.find(adj)
    # TODO: what does this line do
    num_nodes = adj.shape[0]

    node_ids = torch.LongTensor(node_ids)
    u, v = torch.LongTensor(u), torch.LongTensor(v)
    r = torch.LongTensor(r)
    edge_index = torch.stack([u, v], 0)
    edge_weight = r.to(torch.float)
    y = torch.tensor([y])
    z = tradic_counter(adj, 0, 1, max_dist)
    # import ipdb; ipdb.set_trace()
    # data = Data(node_features, edge_index, edge_weight=edge_weight, y=y, z=z,
    #             node_id=node_ids, num_nodes=num_nodes, src_degree=src_degree, dst_degree=dst_degree)
    return data

def get_src_dst_degree(src, dst, A, max_nodes):
    """
    Assumes undirected, unweighted graph
    :param src: Int Tensor[edges]
    :param dst: Int Tensor[edges]
    :param A: scipy CSR adjacency matrix
    :param max_nodes: cap on max node degree
    :return:
    """
    src_degree = A[src].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    dst_degree = A[dst].sum() if (max_nodes is None or A[src].sum() <= max_nodes) else max_nodes
    return src_degree, dst_degree

def neighbors(fringe, A, outgoing=True):
    """
    Retrieve neighbours of nodes within the fringe
    :param fringe: set of node IDs
    :param A: scipy CSR sparse adjacency matrix
    :param outgoing: bool
    :return:
    """
    if outgoing:
        res = set(A[list(fringe)].indices)
    else:
        res = set(A[:, list(fringe)].indices)

    return res
