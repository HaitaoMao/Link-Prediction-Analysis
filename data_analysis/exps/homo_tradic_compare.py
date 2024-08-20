from data_analysis.function.homophily import *
from data_analysis.function.functional import *
from data_analysis.plot_exp import *
from data_analysis.generate_data import load_data
from data_analysis.plot_exp import *
from data_analysis.function.read_results import generate_rank_single
import scipy.sparse as sp
from evaluation_new import * 
import torch
import os
import pathlib
from data_analysis.function.F1 import F1, equal_split, count_bin, generate_split_masks, seperate_accuracy
from data_analysis.function.read_results import *
from data_analysis.function.heuristics import PPR_new


'''
This file is an adaptive function of the homophily and tradic closure, 
The function contente following analysis
    1. simple tradic prediction and tradic count with logistic regression
    2. The overlapping between tradic closure and heuristic, (we typically focus on CN and logistic), 
    heusitic we just select representative ones (F1)
    3. The analysis similar with harrym check the result seperately
    4. (potential: more analysis on different order of tradiic)
        1. the performance analysis between different tradic order
        2. the performance analysis between different weighted tradic
'''

def run_homophily_tradic_compare(args, device):
    args.is_generate_train = 1   # whether preprocess on just training set or entire dataset
    args.is_old_neg = 1            # whether use the new heart negative sampling method
    args.is_flatten = 1  # if use the heart, whether remove the redudant validation and test edge 
    args.is_remove_redudant = 1  # if use the heart, whether remove the redudant validation and test edge 

    args.batch_size = 1000
    args.ratio_per_hop = 1.0

    # args.is_feature = 1 
    # args.label_name = "kmeans" # [kmeans, origin, GMM, SC]
    # args.dis_func_name = "cos"   # cos jaccard l2
    # args.adj_norm = "D2AD2"
    args.is_norm = 0
    args.is_feature_norm = 0
    args.encode_type="drnl" 
    args.algorithm="CN"
   
    args.is_load = 1
    args.is_log = 0

    args.analyze_mode = "valid"  # "whole" "valid" "test"
    _, _, valid_pos_links, valid_neg_links, _ = load_data(args, device)
    args.analyze_mode = "test"  # "whole" "valid" "test"
    dataset, known_links, test_pos_links, test_neg_links, path = load_data(args, device)
    
    # analyze_tradic_hop(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)       
    # just analysis on different the tradic prediction and tradic count with logistic regression
    # import ipdb; ipdb.set_trace()
    analysis_overlapping(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)      
    # overlapping between tradic closure and heuristic, (we typically focus on CN and logistic)
    
    # analysis_overlapping_performance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)      
    # harry's analysis on the performance of different models
    
    # analyze_tradic_feature_importance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
    # the feature importance analysios, similar with buddy (normalized)
    
    # analyze_homo_feature_importance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
    # the homo feature importance analysios, similar with buddy (normalized)
    
    # analyze_model_performance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
    
    # num_hops = [0, 1, 2, 3, 4]
    # norm_types = ["D2AD2", "A", "D-1A"]
    # dis_func_names = ["l2", "cos", "jaccard"]
    # We do not analysis on the kmeans current, since the current version is a little 
    
    '''
    for norm_type in norm_types:
        for num_hop in num_hops:
            for dis_func_name in dis_func_names:
                args.is_feature = 1
                args.dis_func_name = dis_func_name
                args.num_hops = num_hop
                args.adj_norm = norm_type
                try:
                    run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
                except:
                    continue
    '''

    
def analyze_model_performance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path):
    pos_preds, neg_preds, results, ranks = load_ablation(args, args.dataset_name, is_std=True, prefix=None)
    plot_feat_struct_ablation_performance(results, args.dataset_name)
    
    # import ipdb; ipdb.set_trace()
    pos_preds, neg_preds, results, ranks = load_seal(args, args.dataset_name, is_std=True, prefix=None)
    plot_seal_ablation_performance(results, args.dataset_name)
    # import ipdb; ipdb.set_trace()

    print()



def best_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path, K = 3):
    # analysis one will focus on drawing results on different hops
    # we will use the same aggregation function and the same distance function
    num_hops = [0, 1, 2, 3, 4] 
    norm_types = ["D2AD2", "A", "D-1A"]
    dis_func_names = ["l2", "cos", "jaccard"]
    result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
   
    
    
    best_candidates_results, best_candidates_names, best_preds = [-1 for i in range(K)], [" " for i in range(K)], [-1 for i in range(K)]
    min_result = -1
    
    for dis_func_name in dis_func_names:
        for norm_type in norm_types:
            for num_hop in num_hops:
                args.is_feature = 1
                args.dis_func_name = dis_func_name
                args.num_hops = num_hop
                args.adj_norm = norm_type
                try:
                    pos_preds, neg_preds, result = run_single_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
                except:
                    continue
                result = result[result_key]
                # update
                if result > min_result:
                    if args.num_hops != 0:
                        algorithm_name = f"{args.dis_func_name}_{args.adj_norm}_{args.num_hops}" 
                    else:
                        algorithm_name = f"{args.dis_func_name}_{args.num_hops}" 
                    if algorithm_name in best_candidates_names or result in best_candidates_results:
                        continue
                    # avoid redudant
                    idx = best_candidates_results.index(min_result)
                    best_candidates_results[idx] = result
                    best_candidates_names[idx] = algorithm_name
                    best_preds[idx] = [pos_preds, neg_preds]
                    min_result = min(best_candidates_results)
                if args.num_hops == 0:
                    break
    
    
    homo_preds, homo_results = {}, {}
    sorted_idx = np.argsort(best_candidates_results)
    # import ipdb; ipdb.set_trace()
    # for idx, algorithm_name in enumerate(best_candidates_names):
    for i in range(len(sorted_idx)):
        idx = sorted_idx[i]
        algorithm_name = best_candidates_names[idx]
        homo_preds[algorithm_name] = best_preds[idx]
        homo_results[algorithm_name] = best_candidates_results[idx]
    # return best_candidates_names,  best_candidates_results, best_preds
    return homo_preds, homo_results

def analyze_homo_feature_importance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path):
    # just analyze on different tradic algorithm on accuracy
    num_hops = [0, 1, 2, 3, 4] 
    norm_types = ["D2AD2", "A", "D-1A"]
    dis_func_names = ["l2", "cos", "jaccard"]
    result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    best_result = -1
    for dis_func_name in dis_func_names:
        for norm_type in norm_types:
            # print(norm_type)
            # print(dis_func_name)
            pos_val_preds_list, neg_val_preds_list, pos_test_preds_list, neg_test_preds_list = [], [], [], []
            for num_hop in num_hops:
                # print(num_hop)
                args.is_feature = 1
                args.dis_func_name = dis_func_name
                args.num_hops = num_hop
                args.adj_norm = norm_type
                pos_val_preds, neg_val_preds, val_results = run_single_homo(args, device, dataset, known_links, valid_pos_links, valid_neg_links, path)       
                pos_test_preds, neg_test_preds, test_results = run_single_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path)       
                if isinstance(pos_val_preds, torch.Tensor):
                    pos_val_preds, neg_val_preds = pos_val_preds.cpu().numpy(), neg_val_preds.cpu().numpy()
                    pos_test_preds, neg_test_preds = pos_test_preds.cpu().numpy(), neg_test_preds.cpu().numpy()
                pos_val_preds_list.append(pos_val_preds)
                neg_val_preds_list.append(neg_val_preds)
                pos_test_preds_list.append(pos_test_preds)
                neg_test_preds_list.append(neg_test_preds)
            
            pos_val_preds_list = np.stack(pos_val_preds_list, axis=-1)
            neg_val_preds_list = np.stack(neg_val_preds_list, axis=-1)
            pos_test_preds_list = np.stack(pos_test_preds_list, axis=-1)
            neg_test_preds_list = np.stack(neg_test_preds_list, axis=-1)
            pos_preds, neg_preds, weights = feature_importance_homo_logistic(pos_val_preds_list, neg_val_preds_list, pos_test_preds_list, neg_test_preds_list)
            result = get_results(args, pos_preds, neg_preds, result_key)
            
            algorithm = f"homo_{dis_func_name}_{norm_type}"
            if result > best_result:
                best_result = result
                best_algorithm = algorithm
                best_preds = [pos_preds, neg_preds]
                best_weights = weights
    
    plot_homo_feature_importance(best_weights, best_algorithm, args.dataset_name, result_key, best_result)
        

def analyze_tradic_feature_importance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path):
    # just analyze on different tradic algorithm on accuracy
    algorithms = ["CN", "RA", "PA"] 
    results_dict = {}
    result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    
    for algorithm in algorithms:
        results = []
        args.algorithm = algorithm
        pos_preds, neg_preds, weights = feature_importance_tradic_logistic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
        result = get_results(args, pos_preds, neg_preds, result_key)
        plot_tradic_feature_importance(weights, algorithm, args.dataset_name, result_key, result)
        

                


    
def analyze_tradic_hop(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path):
    # just analyze on different tradic algorithm on accuracy
    algorithms = ["CN", "RA", "PA"] 
    results_dict = {}
    result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    
    for algorithm in algorithms:
        results = []

        args.algorithm = algorithm
        # print(args.algorithm)
        # import ipdb; ipdb.set_trace()    
        # pos_preds, neg_preds = feature_importance_tradic_logistic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
        pos_preds, neg_preds = tradic_logistic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
        # for logisitic one
        logit_result = get_results(args, pos_preds, neg_preds, result_key)
        results.append(logit_result)
        pos_preds_list, neg_preds_list = tradic_count_logits(known_links, dataset, path, args, test_pos_links, test_neg_links, is_test=1) 
        # for heurisitc one
        # [num_edge, num_catagories]
        num_hops = pos_preds_list.shape[1]
        for i in range(num_hops):
            pos_preds, neg_preds = pos_preds_list[:, i], neg_preds_list[:, i]
            results.append(get_results(args, pos_preds, neg_preds, result_key))
        results_dict[algorithm] = results
    # plot the results
    plot_tradic_hop(results_dict, len(results_dict[algorithms[0]]), args.dataset_name, result_key)

    print()


def get_ppr(args, device, dataset, known_links, test_pos_links, test_neg_links, path):
    # A = torch.sparse_coo_tensor(known_links.T, torch.ones([np.max(known_links.shape)]).to(device), known_links.shape)
    num_pos, num_neg = test_pos_links.shape[0], test_neg_links.shape[0]
    links = torch.cat([test_pos_links, test_neg_links], dim=0)
    PPR_new(known_links, links)
    
    
    


def analysis_overlapping(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path): 
    results_dict = {}
    result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    Ks = {"Cora": 100, "Citeseer": 100, "Pubmed": 100, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 100, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    get_ppr(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
    # neighbors, neighbor_weights = get_ppr_matrix(known_links, num_nodes, alpha=0.15, eps=5e-5)
    import ipdb; ipdb.set_trace()
    
    result_key = result_key_dict[args.dataset_name]
    
    # first extract CN performance
    tradic_algorithms, tradic_preds, tradic_results = [], [], []
    args.algorithm = "CN"
    pos_tradic_lr_preds, neg_tradic_lr_preds = tradic_logistic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
    # for logisitic one
    logit_result = get_results(args, pos_tradic_lr_preds, neg_tradic_lr_preds, result_key)
    tradic_results.append(logit_result)
    tradic_preds.append([pos_tradic_lr_preds, neg_tradic_lr_preds])
    tradic_algorithms.append(f"{args.algorithm}_LR")
    
    pos_preds_list, neg_preds_list = tradic_count_logits(known_links, dataset, path, args, test_pos_links, test_neg_links, is_test=1) 
    num_hops = pos_preds_list.shape[1]
    results = []
    for i in range(num_hops):
        pos_preds, neg_preds = pos_preds_list[:, i], neg_preds_list[:, i]
        results.append(get_results(args, pos_preds, neg_preds, result_key))
    
    results = np.array(results)
    max_idx = np.argmax(results)
    pos_preds_best, neg_preds_best = pos_preds_list[:, max_idx], neg_preds_list[:, max_idx]
    best_result = results[max_idx]
    tradic_results.append(best_result)
    tradic_preds.append([pos_preds_best, neg_preds_best])
    tradic_algorithms.append(f"{args.algorithm}")
    
    
    tradic_algorithms = [args.algorithm, f"{args.algorithm}_LR"]
    
    # then extract the homophily performance
    '''
    norm_types = ["D-1A", "A"]
    dis_func_name = "cos"
    # we do not select dis_func_name here
    num_hops = [0, 2]

    homo_algorithms, homo_preds, homo_results = [], [], []
    for num_hop in num_hops:
        for norm_type in norm_types:
            args.is_feature = 1
            args.dis_func_name = dis_func_name
            args.num_hops = num_hop
            args.adj_norm = norm_type
            pos_preds, neg_preds = feature_homophily_ratio(args, dataset, known_links, test_pos_links, test_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
            result = get_results(args, pos_preds, neg_preds, result_key)
            homo_model_name = f"{norm_type}_{num_hop}"  # homo_{dis_func_name}_      
            homo_algorithms.append(homo_model_name)
            homo_preds.append([pos_preds, neg_preds])
            homo_results.append(result)    
            if num_hop == 0:
                break
            # for the 0 hop, we do not need repeat run        
    '''
    
    homo_preds, homo_results = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
    homo_algorithms = list(homo_preds.keys())
    homo_results, homo_preds = list(homo_results.values()), list(homo_preds.values())
    num_pos = pos_preds.shape[0]
    num_tradic, num_homo = len(tradic_algorithms), len(homo_algorithms)
    
    result_keys = ["TP", "FP", "TN", "FN"]
    results_dict = {key: np.zeros([num_tradic, num_homo]) for key in result_keys}
    num_homo, num_tradic = len(homo_algorithms), len(tradic_algorithms)
    for tradic_idx, (tradic_algorithm, tradic_pred, tradic_result) in enumerate(zip(tradic_algorithms, tradic_preds, tradic_results)):
        tradic_pos_preds, tradic_neg_preds = tradic_pred[0], tradic_pred[1]
        for homo_idx, (homo_algorithm, homo_pred, homo_result) in enumerate(zip(homo_algorithms, homo_preds, homo_results)):
            homo_pos_preds, homo_neg_preds = homo_pred[0], homo_pred[1]
            
            rank1 = generate_rank_single(homo_pos_preds, homo_neg_preds)
            rank2 = generate_rank_single(tradic_pos_preds, tradic_neg_preds)

            F1_results = F1(rank1, rank2, num_pos=num_pos, K=K)
            
            for key in result_keys:            
                results_dict[key][tradic_idx, homo_idx] = F1_results[key]
            
            # for results_list
            # for key in F1_results.keys():
            #     differences_list[key].append(F1_results[key])
        
    # import ipdb; ipdb.set_trace()
    homophily_tradic_compare(results_dict, tradic_algorithms, homo_algorithms, args.dataset_name, result_key)
    
            
    
def analysis_overlapping_performance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path): 
    # There are many overlapping betweetn the second and the first function
    # We turn different hyperparameter seperatelyto enable easy tuning
    args.num_bin = 5
    results_dict = {}
    result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    Ks = {"Cora": 100, "Citeseer": 100, "Pubmed": 100, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 100, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    result_key = result_key_dict[args.dataset_name]
    
    # first extract CN performance
    tradic_preds, tradic_results = {}, {}
    args.algorithm = "CN"
    
    pos_tradic_lr_preds, neg_tradic_lr_preds = tradic_logistic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
    
    # for logisitic one
    logit_result = get_results(args, pos_tradic_lr_preds, neg_tradic_lr_preds, result_key)
    tradic_algorithm_name = f"{args.algorithm}_LR"
    tradic_results[tradic_algorithm_name] = logit_result
    tradic_preds[tradic_algorithm_name] = [pos_tradic_lr_preds, neg_tradic_lr_preds]
    
    pos_preds_list, neg_preds_list = tradic_count_logits(known_links, dataset, path, args, test_pos_links, test_neg_links, is_test=1) 
    
    num_hops = pos_preds_list.shape[1]
    results = []
    for i in range(num_hops):
        pos_preds, neg_preds = pos_preds_list[:, i], neg_preds_list[:, i]
        results.append(get_results(args, pos_preds, neg_preds, result_key))
    
    results = np.array(results)
    max_idx = np.argmax(results)
    pos_preds_best, neg_preds_best = pos_preds_list[:, max_idx], neg_preds_list[:, max_idx]
    
    best_result = results[max_idx]
    
    tradic_algorithm_name = f"{args.algorithm}"
    tradic_results[tradic_algorithm_name] = best_result
    tradic_preds[tradic_algorithm_name] = [pos_preds_best, neg_preds_best]
    
    homo_preds, homo_results = {}, {}
    
    # then extract the homophily performance
    
    '''
    norm_types = ["D-1A", "A"]
    dis_func_name = "cos"
    # we do not select dis_func_name here
    num_hops = [0, 2]

    homo_preds, homo_results = {}, {}
    for num_hop in num_hops:
        for norm_type in norm_types:
            args.is_feature = 1
            args.dis_func_name = dis_func_name
            args.num_hops = num_hop
            args.adj_norm = norm_type
            pos_preds, neg_preds = feature_homophily_ratio(args, dataset, known_links, test_pos_links, test_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
            result = get_results(args, pos_preds, neg_preds, result_key)
            homo_algorithm_name = f"{norm_type}_{num_hop}"  # homo_{dis_func_name}_      
            
            homo_preds[homo_algorithm_name] = [pos_preds, neg_preds]
            homo_results[homo_algorithm_name] = result    
            if num_hop == 0:
                break
            # for the 0 hop, we do not need repeat run        
    best_homo_names, best_homo_results, best_homo_preds = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
    '''
    # automatically select the best performance
    homo_preds, homo_results = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
    # import ipdb; ipdb.set_trace()
    # two output form, figure and performance table
    # How do we show a figure? A good strategy could be using the bar figure
    # TODO: how to automatic select the bin value, current it may not be suitable for many other datasets   
    
    preds_dict = {**tradic_preds, **homo_preds}
    results_dict = {**tradic_results, **homo_results}
    
    for compare_algorithm_name in preds_dict.keys():
        seperate_results = {}
        # compare_algorithm_name = 'CN'
        pos_compare_preds, neg_compare_preds = preds_dict[compare_algorithm_name][0], preds_dict[compare_algorithm_name][1]
        compare_result = results_dict[compare_algorithm_name]        
        split_values, num_pos_values = equal_split(pos_compare_preds, args.num_bin)
        _, num_neg_values = count_bin(neg_compare_preds, split_values)
        masks = generate_split_masks(pos_compare_preds, split_values)
        
        # check
        print(f"num edge: {np.sum(num_pos_values)} num edge: {pos_compare_preds.shape[0]}")
        # import ipdb; ipdb.set_trace()
        for i, num_pos_value in enumerate(num_pos_values):
            try:
                print(f"num: {num_pos_value}")
                print(f"mask: {np.sum(masks[i])}")
            except:
                import ipdb; ipdb.set_trace()
        # print(compare_result)
        # import ipdb; ipdb.set_trace()

        compare_results = seperate_accuracy(pos_compare_preds, neg_compare_preds, masks, num_pos_values, K)
        for algorithm_name in preds_dict.keys():
            if algorithm_name == compare_algorithm_name:
                continue
            pos_preds, neg_preds = preds_dict[algorithm_name][0], preds_dict[algorithm_name][1]
            result = results_dict[algorithm_name]
            
            results = seperate_accuracy(pos_preds, neg_preds, masks, num_pos_values, K)
            seperate_results[algorithm_name] = results
        
        # import ipdb; ipdb.set_trace()
        plot_compare_performance(seperate_results, compare_results, compare_algorithm_name, results_dict, args.dataset_name, split_values, num_pos_values, num_neg_values, result_key)
    
    '''    
    for tradic_algorithm_name in tradic_preds.keys():
        pos_tradic_preds, neg_tradic_preds = tradic_preds[tradic_algorithm_name][0], tradic_preds[tradic_algorithm_name][1]
        tradic_result = tradic_results[tradic_algorithm_name]        
        split_values, num_pos_values = equal_split(pos_tradic_preds, args.num_bin)
        _, num_neg_values = count_bin(neg_tradic_preds, split_values)
        masks = generate_split_masks(pos_tradic_preds, split_values)

        for homo_algorithm_name in homo_preds.keys():
            pos_homo_preds, neg_homo_preds = homo_preds[homo_algorithm_name][0], homo_preds[homo_algorithm_name][1]
            homo_result = homo_results[homo_algorithm_name]
            
            results = seperate_accuracy(pos_homo_preds, neg_homo_preds, masks, num_pos_values, K)
            import ipdb; ipdb.set_trace()
            print()

    '''

def get_results(args, pos_preds, neg_preds, metric):
    if isinstance(pos_preds, np.ndarray):
        pos_preds, neg_preds = torch.tensor(pos_preds), torch.tensor(neg_preds)
    
    if args.is_flatten == 0 and args.is_old_neg == 0:
        num_edges = args.num_pos_test_edge if args.analyze_mode == "valid" else args.num_pos_val_edge
        neg_preds = torch.reshape(neg_preds, [num_edges, -1])
        # pos_preds = torch.unsqueeze(pos_preds, dim=1) 
        # new_preds = torch.cat([pos_preds, neg_preds], dim=1)
        results = get_metric_score(pos_preds, neg_preds)
    elif args.dataset_name == "ogbl-citation2":
        pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
        results = get_metric_score(pos_preds, neg_preds)
    else:
        pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
        results = get_metric_score_origin(pos_preds, neg_preds)
    
    return results[metric]


def analysis_norm(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path):
    # analysis one will focus on drawing results on different hops
    # we will use the same aggregation function and the same distance function
    num_hops = [0, 1, 2, 3, 4]
    norm_types = ["D2AD2", "A", "D-1A"] 
    dis_func_names = ["l2", "cos", "jaccard"]
    result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    
    for dis_func_name in dis_func_names:
        # list of list, [norm_type, num_hop: result]
        for norm_type in norm_types:
            for num_hop in num_hops:
                args.is_feature = 1
                args.dis_func_name = dis_func_name
                args.num_hops = num_hop
                args.adj_norm = norm_type
            
    




def run_single_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path):
    data_path = f"output_analyze/results/{args.dataset_name}"
    folder_path = pathlib.Path(data_path) 
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    if args.is_log:
        file_name = f"homo_{args.num_hops}_{args.adj_norm}_{args.is_old_neg}"
    elif args.is_feature:
        file_name = f"homo_{args.num_hops}_{args.adj_norm}_{args.dis_func_name}_{args.is_feature}_{args.is_old_neg}"
    else:
        file_name = f"homo_{args.num_hops}_{args.adj_norm}_{args.label_name}_{args.is_feature}_{args.is_old_neg}"

    if not args.is_load:
        if args.is_log:
            pos_preds, neg_preds = logits_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
        elif args.is_feature:
            pos_preds, neg_preds = feature_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
        else:
            pos_preds, neg_preds = label_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, label_name=args.label_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
        '''
        args.num_hops = 2
        pos_preds2, neg_preds2 = feature_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
        import ipdb; ipdb.set_trace()
        print()
        '''
        with open(f"{data_path}/{file_name}.txt", "wb") as f:
            pickle.dump({"pos": pos_preds, "neg": neg_preds}, f)
    else:
        try:
            with open(f"{data_path}/{file_name}.txt", "rb") as f:
                data_dict = pickle.load(f)
            pos_preds = data_dict["pos"]
            neg_preds = data_dict["neg"]    
            # import ipdb; ipdb.set_trace() 
        except:
            if args.is_log:
                pos_preds, neg_preds = logits_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
            elif args.is_feature:
                pos_preds, neg_preds = feature_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
            else:
                pos_preds, neg_preds = label_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, label_name=args.label_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
            
            with open(f"{data_path}/{file_name}.txt", "wb") as f:
                pickle.dump({"pos": pos_preds, "neg": neg_preds}, f)
            
    # pos_preds, neg_preds = torch.tensor(pos_preds), torch.tensor(neg_preds)
    if args.is_flatten == 0 and args.is_old_neg == 0:
        num_edges = args.num_pos_test_edge if args.analyze_mode == "valid" else args.num_pos_val_edge
        neg_preds = torch.reshape(neg_preds, [num_edges, -1])
        # pos_preds = torch.unsqueeze(pos_preds, dim=1) 
        # new_preds = torch.cat([pos_preds, neg_preds], dim=1)
        results = get_metric_score(pos_preds, neg_preds)
    elif args.dataset_name == "ogbl-citation2":
        pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
        results = get_metric_score(pos_preds, neg_preds)
    else:
        pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
        results = get_metric_score_origin(pos_preds, neg_preds)

    if args.is_log:
        file_name = f"homo_result_{args.num_hops}_{args.adj_norm}_{args.is_old_neg}"
    elif args.is_feature:
        file_name = f"homo_result_{args.adj_norm}_{args.dis_func_name}_{args.is_feature}_{args.is_old_neg}"
    else:
        file_name = f"homo_result_{args.adj_norm}_{args.label_name}_{args.is_feature}_{args.is_old_neg}"

    if args.is_log:
        result_strings = [f"{args.dataset_name}_homo_{args.adj_norm}"]
    elif args.is_feature:
        result_strings = [f"{args.dataset_name}_homo_{args.adj_norm}_{args.dis_func_name}_{args.is_feature}"]
    else:
        result_strings = [f"{args.dataset_name}_homo_{args.adj_norm}_{args.label_name}_{args.is_feature}"]


    #  = [f"{args.dataset_name}_homo_{args.num_hops}_{args.norm_type}"]
    for key in results:
        result_strings.append(f"{key}:{results[key]}")
    sep = " "
    result_string = sep.join(result_strings)+"\n"

    if args.is_log:
        with open(f"{data_path}/ALL_homo_{args.adj_norm}.txt", "w") as f:
            f.write(result_string)
    elif args.is_feature:
        with open(f"{data_path}/ALL_homo_{args.adj_norm}_{args.dis_func_name}_{args.is_feature}.txt", "w") as f:
            f.write(result_string)
    else:
        with open(f"{data_path}/ALL_homo_{args.adj_norm}_{args.label_name}_{args.is_feature}.txt", "w") as f:
            f.write(result_string)
    
    with open(f"{data_path}/{file_name}.txt", "wb") as f:
        pickle.dump(results, f)
    
    result_name = f"output_analyze/all_results/{args.dataset_name}_{args.is_old_neg}.xlsx"
    metric_names = ["algorithm", 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@100', "MRR"]
    try:
        results_record = pd.read_excel(result_name)
    except:
        results_record =pd.DataFrame(columns=metric_names)
        # results_record.set_index("algorithm", inplace=True) 
    if args.is_log:
        algorithm_key = f"homo_{args.num_hops}_{args.adj_norm}"
    if args.is_feature:
        algorithm_key = f"homo_{args.num_hops}_{args.adj_norm}_{args.dis_func_name}"
    else:
        algorithm_key = f"homo_{args.num_hops}_{args.adj_norm}_{args.label_name}"

    if algorithm_key not in results_record["algorithm"]:
        tmp_dict = {"algorithm": [algorithm_key]}                
    for key in results.keys():
        tmp_dict[key] = [results[key]]
        # results_record.at[algorithm_key, key] = results[key]
    # results_record.append(tmp_dict, ignore_index=False)
    new_row = pd.DataFrame(tmp_dict)
    # import ipdb; ipdb.set_trace()

    results_record = pd.concat([results_record, new_row], ignore_index=True)
    results_record.to_excel(result_name, index=False)

    # with open(f"{data_path}/homo_result_{args.norm_type}_{args.is_old_neg}.txt", "wb") as f:
    return pos_preds, neg_preds, results
    