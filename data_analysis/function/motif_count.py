import torch 
import random
import math
from motif.subgraph_change_labeler import SubgraphChangeLabeler
from motif.graph_change_feature_counts import GraphChangeFeatureCounter
from motif.graph_change import *
from multiprocessing import Pool
import numpy as np
import copy

@torch.no_grad()
def motif_count(args, edge_index, added_links):
    from motif.graph_data import GraphData, DirectedGraphData
    edge_index, added_links = edge_index.cpu().numpy(), added_links.cpu().numpy()
    num_add_edges = added_links.shape[0]
    num_nodes = np.max(edge_index) + 1
    gd = GraphData()
        
    for i in range(num_nodes):
        gd.add_node(i)
    
    for i in range(len(edge_index)):
        gd.add_edge(edge_index[i][0], edge_index[i][1])

    changes = []
    for i in range(len(added_links)):
        added_link = added_links[i]
        changes.append(EdgeAddition(gd, added_link[0], added_link[1]))
    non_changes = []
    GCFC = GraphChangeFeatureCounter(gd, subgraph_size=4, use_counts=True, precompute=False, node_graphs_reduced=False)
    (counts, _, non_counts) = GCFC.get_change_counts(changes, [], non_changes)
    print(counts[1])
    counts = counts[1]

    sst_labeler = GCFC.get_subgraph_change_labeler()
    ct = GraphChange.EDGE_ADDITION
    num_labels = sst_labeler.num_of_change_cases_for_type(ct)  
    for l in range(0, num_labels):
        data = sst_labeler.get_representative_subgraph_change_from_label(l, ct)
    descriptions = []
    for label, count in counts[0].items():
        description = sst_labeler.get_representative_subgraph_change_from_label(label, GraphChange.EDGE_ADDITION)
        print(description)
        descriptions.append(description)
    # import ipdb; ipdb.set_trace()
    results = torch.zeros([num_add_edges, num_labels])
    for idx, count in enumerate(counts):
        for key in count.keys():
            results[idx, int(key)] = int(count[key])
    
    return results, descriptions
    