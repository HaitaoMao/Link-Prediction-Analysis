from data import get_data, get_loaders
from models.elph import ELPH, BUDDY
from models.seal import SEALDGCNN, SEALGCN, SEALGIN, SEALSAGE
from utils import ROOT_DIR, print_model_params, select_embedding, str2bool
from wandb_setup import initialise_wandb
from runners.train import get_train_func
from runners.inference import test
# from data_analysis.tradic_analysis import tradic_analysis
from data_analysis.function.loader import *
from data_analysis.function.functional import *
import torch
from torch_geometric.utils import (negative_sampling, remove_self_loops,add_self_loops, to_networkx, degree, subgraph, k_hop_subgraph, is_undirected, to_undirected)

# from data_analysis.function.heuristics import 
import networkx as nx
import numpy as np


def generate_distance(args, device):
    # this function is to generate node pair distance and save for further use
    # required hyperparameter: args.is_generate_train:  whether preprocess on just training set or entire dataset
    args.is_generate_train = 1
    args.is_old_neg = 0
    # TODO: add the analysis on the test nodes with
    dataset, splits, directed, eval_metric = get_data(args, device)
    train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
    if args.dataset_name in ["Cora", "Citeseer", "Pubmed"]:
        train_data, val_data, test_data = upload_split(args.dataset_name, train_data, val_data, test_data)
    # edge_label=[num_edge], edge_label_index=[2, num_edge]
    dataset = get_datasets(dataset, dataset[0], args) 
    # get the whole dataset, links are all the links. 
    # If you want to get the train test validation dataset, just replace the links 
    # current is a subgraph version
    if args.is_old_neg:
        pos_train_edge, neg_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge = \
            get_train_val_test_links(dataset, splits['train'], splits['valid'], splits['test'], args)
        pos_train_edge, neg_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge = \
            pos_train_edge.to(device), neg_train_edge.to(device), pos_val_edge.to(device), neg_val_edge.to(device), pos_test_edge.to(device), neg_test_edge.to(device)
    # TODO: we focus on the test set, do not take the analysis on validation set into consideration        
    else:
        # Use HeaRT 
        pos_train_edge, neg_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge = load_heart_edge(args, device)         
        # the output negative sample would [num_train_edge, num_neg_per_edge, 2]
    # TODO: need a new configuration on the test set
    
    
    if args.dataset_name == "ogbl-collab": pos_train_edge = torch.cat([pos_train_edge, pos_val_edge], dim=0) 
    
    # pos_train_edge, _ = remove_self_loops(to_undirected(pos_train_edge.T))
    # pos_train_edge = pos_train_edge.T
    args.num_pos_val_edge, args.num_pos_test_edge = pos_val_edge.shape[0], pos_test_edge.shape[0]

    start_time = time.time()
    path = f"intermedia_result/tradic_dis/{args.dataset_name}_train" if args.is_generate_train else f"intermedia_result/tradic_dis/{args.dataset_name}"
    
    if args.is_generate_train:
        # generate once then save
        adjs, adjs_dis = get_distance(pos_train_edge.permute([1, 0]), path, args.num_hops)
    else:
        adjs, adjs_dis = get_distance(dataset.data.edge_index, path, args.num_hops)
    # TODO: the specific finding for DDI dataset
    print("Time: ", time.time() - start_time, " seconds")
    
    '''
    # the following are the check function
    G = nx.Graph()
    # import ipdb; ipdb.set_trace()
    G.add_edges_from(pos_train_edge.cpu().numpy().tolist())
    for dis, adj_dis in enumerate(adjs_dis):
        print(f"distance: {dis}")
        for i in range(20):
            # import ipdb; ipdb.set_trace()
            pair = adj_dis[:, i].cpu().numpy()
            print(len(nx.shortest_path(G, source=pair[0],target=pair[1])) - 1)
    print()
    '''

def load_data(args, device):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    args.is_generate_train = 1   # whether preprocess on just training set or entire dataset
    args.is_old_neg = 1            # whether use the new heart negative sampling method
    args.analyze_mode = "test"  # "whole" "valid" "test"
    args.is_flatten = 1  # if use the heart, whether remove the redudant validation and test edge 
    args.is_remove_redudant = 1  # if use the heart, whether remove the redudant validation and test edge 

    args.batch_size = 1000
    args.ratio_per_hop = 1.0
    '''

    # TODO: add the analysis on the test nodes with
    dataset, splits, directed, eval_metric = get_data(args, device)
    train_data, val_data, test_data = splits['train'], splits['valid'], splits['test']
    if args.dataset_name in ["Cora", "Citeseer", "Pubmed"]:
        train_data, val_data, test_data = upload_split(args.dataset_name, train_data, val_data, test_data)
    # import ipdb; ipdb.set_trace()
    # edge_label=[num_edge], edge_label_index=[2, num_edge]
    dataset = get_datasets(dataset, dataset[0], args) 
    # get the whole dataset, links are all the links. 
    # If you want to get the train test validation dataset, just replace the links 
    # current is a subgraph version
    if args.is_old_neg:
        pos_train_edge, neg_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge = \
            get_train_val_test_links(dataset, splits['train'], splits['valid'], splits['test'], args)
        pos_train_edge, neg_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge = \
            pos_train_edge.to(device), neg_train_edge.to(device), pos_val_edge.to(device), neg_val_edge.to(device), pos_test_edge.to(device), neg_test_edge.to(device)
    # TODO: we focus on the test set, do not take the analysis on validation set into consideration        
    else:
        # Use HeaRT
        plantoid_names = ["Cora", "cora", "Citeseer", "citeseer", "Pubmed", "pubmed"]
        
        if args.dataset_name in plantoid_names:
            pos_train_edge, neg_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge = load_plantoid_heart_edge(args, device)  
        else:
            pos_train_edge, neg_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge = \
                get_train_val_test_links(dataset, splits['train'], splits['valid'], splits['test'], args)
            pos_train_edge, neg_train_edge, pos_val_edge, neg_val_edge, pos_test_edge, neg_test_edge = \
            pos_train_edge.to(device), neg_train_edge.to(device), pos_val_edge.to(device), neg_val_edge.to(device), pos_test_edge.to(device), neg_test_edge.to(device)
            neg_val_edge, neg_test_edge = load_ogb_heart_edge(args, device)  
            neg_val_edge, neg_test_edge = neg_val_edge.to(device), neg_test_edge.to(device)
                   
        # the output negative sample would [num_train_edge, num_neg_per_edge, 2]
    # TODO: need a new configuration on the test set
    # mask = (pos_train_edge == pos_val_edge[0]).sum()
    # import ipdb; ipdb.set_trace()
    print(pos_train_edge.shape[0])
    if args.dataset_name == "ogbl-collab": pos_train_edge = torch.cat([pos_train_edge, pos_val_edge], dim=0) 
    pos_train_edge, inverse_indices = torch.unique(pos_train_edge.t(), dim=0, return_inverse=True)
    pos_train_edge = pos_train_edge.t()
    print(pos_train_edge.shape[0])
    
    # pos_train_edge, _ = remove_self_loops(to_undirected(pos_train_edge.T))
    # pos_train_edge = pos_train_edge.T
    
    args.num_pos_val_edge, args.num_pos_test_edge = pos_val_edge.shape[0], pos_test_edge.shape[0]
    if args.is_flatten: 
        neg_val_edge, neg_test_edge\
            = flatten_neg_edges(neg_val_edge, args.is_remove_redudant), flatten_neg_edges(neg_test_edge, args.is_remove_redudant)

    path = f"intermedia_result/tradic_dis/{args.dataset_name}_train" if args.is_generate_train else f"intermedia_result/tradic_dis/{args.dataset_name}"
    
    # analyze_mode:  "whole" "validation" "test"
    # TODO: the specific finding for DDI dataset
    if args.analyze_mode == "whole":
        known_links, _ = remove_redudant_edge(dataset.links)
        # reduced_edge_list, remaining_indices = remove_redudant_edge(dataset.links)
        known_links = torch.tensor(known_links).to(dataset.data.x.device)
        path = f"intermedia_result/tradic_dis/{args.dataset_name}"
    elif args.analyze_mode == "valid":
        eval_pos_links, eval_neg_links = pos_val_edge, neg_val_edge
        known_links = pos_train_edge
        path = f"intermedia_result/tradic_dis/{args.dataset_name}_train"
    elif args.analyze_mode == "test":
        eval_pos_links, eval_neg_links = pos_test_edge, neg_test_edge
        known_links = pos_train_edge
        path = f"intermedia_result/tradic_dis/{args.dataset_name}_train"
    # shape: [num_edge, 2] Notice: it is differnt from the original edge index [2, num_edge]    

    return dataset, known_links, eval_pos_links, eval_neg_links, path