import math
import random
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
import numpy as np
from scipy.stats import spearmanr

def forward(model, data):
    pass

def load():
    # TODO: loads the existing result
    pass

# There are two kinds of ranking based on different evaluation metric
# Now the version is for an overall ranking

# TODO: provide the node id of the wrong predict edge, for finding the property later. 
def run_correlation(positive_scores1, negative_scores1, positive_scores2, negative_scores2, name1, name2, num_samples=-1):
    if num_samples != -1:
        positive_scores1 = positive_scores1[:num_samples] 
        negative_scores1 = negative_scores1[:num_samples]
        positive_scores2 = positive_scores2[:num_samples]
        negative_scores2 = negative_scores2[:num_samples]
    
    device = positive_scores1.device 
    num_edges = positive_scores1.shape[0] + negative_scores1.shape[0]
    
    labels = torch.cat([torch.ones(positive_scores1.shape[0]), torch.ones(negative_scores1.shape[0])], dim=-1)
    scores1 = torch.cat([positive_scores1, negative_scores1], dim=-1)
    sorted_scores1, sorted_indices1 = torch.sort(scores1, descending=True)
    sorted_labels1 = labels[sorted_indices1]

    scores2 = torch.cat([positive_scores2, negative_scores2], dim=-1)
    sorted_scores2, sorted_indices2 = torch.sort(scores2, descending=True)
    sorted_labels2 = labels[sorted_indices2]

    # the logic here seems to be a little bit complicated
    node_rank = torch.arange(num_edges)
    sorted_sorted_scores1, sorted_sorted_indices1 = torch.sort(sorted_indices1, descending=False)
    node_rank1 = node_rank[sorted_sorted_indices1]
    sorted_sorted_scores2, sorted_sorted_indices2 = torch.sort(sorted_indices2, descending=False)
    node_rank2 = node_rank[sorted_sorted_indices2]
    data = torch.cat([node_rank, node_rank1, node_rank2], dim=-1)

    node_rank1, node_rank2 = node_rank1.cpu().numpy(), node_rank2.cpu().numpy()
    corr_coef, p_value = spearmanr(node_rank1, node_rank2)

    return [node_rank1, node_rank2], corr_coef

    #TODO: require further analysis

    # TODO: one analysis: the positive sample should be at the front of the neghative samples


def run_correlation_simple(node_rank1, node_rank2, name1, name2):
    if isinstance(node_rank1, torch.Tensor):
        node_rank1, node_rank2 = node_rank1.cpu().numpy(), node_rank2.cpu().numpy()
    corr_coef, p_value = spearmanr(node_rank1, node_rank2)

    return corr_coef, p_value


