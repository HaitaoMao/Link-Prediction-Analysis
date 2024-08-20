from math import inf
import random

import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
from torch_geometric.utils import (negative_sampling, add_self_loops)
import torch_sparse
from tqdm import tqdm
import scipy.sparse as ssp
from torch.utils.data import DataLoader

# TODO: current the preprocess is for the original graph, do not consider the train and test negative samples
def preprocess(dataset, args, use_coalesce=True, directed=False):
    dataset.num_nodes = dataset.x.shape[0]
    if 'edge_weight' in dataset:
        dataset.edge_weight = dataset.edge_weight.view(-1)
    else:
        dataset.edge_weight = torch.ones(dataset.edge_index.size(1), dtype=int)

    if use_coalesce:  # compress mutli-edge into edge with weight
        dataset.edge_index, dataset.edge_weight = torch_sparse.coalesce(
            dataset.edge_index, dataset.edge_weight,
            dataset.num_nodes, dataset.num_nodes)

    if directed:  # make undirected graphs like citation2 directed
        print(f'this is a directed graph. Making the adjacency matrix undirected to propagate features and calculate subgraph features')
        dataset.edge_index, dataset.edge_weight = to_undirected(dataset.edge_index, dataset.edge_weight)
    else:
        dataset.edge_index = dataset.edge_index
    # import ipdb; ipdb.set_trace()

    dataset.A = ssp.csr_matrix(
        (dataset.edge_weight, (dataset.edge_index[0], dataset.edge_index[1])),
        shape=(dataset.num_nodes, dataset.num_nodes)
    )
    
    if directed:
        dataset.A_csc = dataset.A.tocsc()
    else:
        dataset.A_csc = None

    # TODO: 
    dataset.links = dataset.edge_index.T
    return dataset


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
            import ipdb; ipdb.set_trace()
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
    dist2src[dist2src > max_dist] = max_dist

    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)
    dist2dst[dist2dst > max_dist] = max_dist

    return (dist2src, dist2dst)




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
    z = tradic_counter(adj, src, dst)
    
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


