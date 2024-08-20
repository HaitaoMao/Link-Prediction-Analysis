from data_analysis.function.homophily import *
from data_analysis.function.functional import *
from data_analysis.plot_exp import *
from data_analysis.generate_data import load_data
from data_analysis.plot_exp import *
from data_analysis.function.read_results import generate_rank_single
from data_analysis.function.F1 import equal_split
import scipy.sparse as sp
from evaluation_new import * 
import torch
import os
import pathlib
from data_analysis.function.F1 import F1, F1_new, equal_split, count_bin, generate_split_masks, seperate_accuracy
from data_analysis.function.read_results import *
from data_analysis.function.heuristics import PPR_new, PPR_correct, SimRank_correct, Katz, generalized_CN_new
from torch_geometric.utils import mask_to_index, index_to_mask
import torch_geometric

'''
The new method support 
    1. load homophily, local, global, model performance
    2. all methods into both new split and original split
    3. plugin comparison between different groups
    
loading different hop informaiton
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

'''
ogbl-collab: we usually include the validation edges in the training graph when doing testing.
ogbl-ddi: It doesn't have node features. There is a weak relationship between the validation and test performance.
ogbl-ppa: The node feature is the 58 dimensional one-hot vector. The MLP has very bad performance.
ogbl-citation2:  In the validation/test,  positive samples have specific negative samples.
'''


def run_homophily_tradic_plugin(args, device):
    args.is_generate_train = 1   # whether preprocess on just training set or entire dataset
    args.is_old_neg = 1          # whether use the new heart negative sampling method
    args.is_flatten = 0  # if use the heart, whether remove the redudant validation and test edge 
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
    args.num_bin = 5
    

    args.analyze_mode = "valid"  # "whole" "valid" "test"
    _, _, valid_pos_links, valid_neg_links, _ = load_data(args, device)
    if not args.is_old_neg:
        args.neg_per_valid = valid_neg_links.shape[1]
        valid_neg_links = torch.reshape(valid_neg_links, [-1, 2])
    args.analyze_mode = "test"  # "whole" "valid" "test"
    # import ipdb; ipdb.set_trace()
    dataset, known_links, test_pos_links, test_neg_links, path = load_data(args, device)
    if not args.is_old_neg:
        args.neg_per_test = test_neg_links.shape[1]
        test_neg_links = torch.reshape(test_neg_links, [-1, 2])
    
    # "homo", "CN" "global" "model" 
    # TODO: model may need provide candidates
    # algorithms = [["model", "homo", "CN", "global"]]

    # ["homo", "global"], ["global", "CN"], ["homo", "CN"] 
    # ["model", "CN"] 
    # "ablation", 
    # TODO: load ablation has some issue
    models = ["mlp", "gcn", "sage", "seal", "buddy"]  # , "neognn"
    # models = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # models = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    
    algorithms = [["homo", "global"], ["global", "CN"], ["homo", "CN"]]
    algorithms = [["CN", "homo"]]
    algorithms = ["CN", "homo", "global"]
    
    model_both_pos_neg(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms)
    # model_both_pos_neg_same_cate(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms)
    # model_complementary_all_new(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms)
    # struct_feat_compare(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms)
    

    # algorithms_list = [["CN", "global", "homo"], ["homo", "global", "CN"], ["global", "CN", "homo"]]
    # for algorithms in algorithms_list:
    #     model_complementary_all(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms)
    
    
    # import ipdb; ipdb.set_trace()
    # tradic_decay(args, device, dataset, known_links, valid_pos_links, 
    #                        valid_neg_links, test_pos_links, test_neg_links, path, algorithm, models=models)
    # for algorithm in algorithms:
    
    # tradic_decay_ideal(args, device, dataset, known_links, valid_pos_links, 
    #                     valid_neg_links, test_pos_links, test_neg_links, path, models=models)       
    
    # homo_decay_ideal(args, device, dataset, known_links, valid_pos_links, 
    #                     valid_neg_links, test_pos_links, test_neg_links, path, models=models)       
    
    # calculate_degree(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
           
    '''
    algorithms = [["homo", "CN", "global"]]
    for algorithm in algorithms:
        model_complementary_all(args, device, dataset, known_links, valid_pos_links, 
                           valid_neg_links, test_pos_links, test_neg_links, path, algorithm, models=models)       
        model_complementary_all_ranks(args, device, dataset, known_links, valid_pos_links, 
                           valid_neg_links, test_pos_links, test_neg_links, path, algorithm, models=models)       
        model_complementary(args, device, dataset, known_links, valid_pos_links, 
                           valid_neg_links, test_pos_links, test_neg_links, path, algorithm, models=models)       
        model_correctness(args, device, dataset, known_links, valid_pos_links, 
                           valid_neg_links, test_pos_links, test_neg_links, path, algorithm, models=models)       
    
        
    base_models = ['empty', "gcn", 'mlp']
    # base_model = "empty" # 'gcn' 'homo' 'local' 'global's
    basis_heuristics = ["homo", 'local', 'global']  
    # for algorithm in algorithm:
    for base_model in base_models:
        for basis_heusistic in basis_heuristics: 
            args.base_model, args.basis_heuristic = base_model, basis_heusistic
            analyze_difference_with_base_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithm, models=models)       
                
                
    for base_model in base_models:
        for basis_heusistic1 in basis_heuristics:
            for basis_heusistic2 in basis_heuristics:
                if basis_heusistic1 == basis_heusistic2:
                    continue
                args.base_model, args.basis_heuristic1, args.basis_heuristic2 = base_model, basis_heusistic1, basis_heusistic2
                analyze_double_difference_with_base_model(args, device, dataset, known_links, valid_pos_links, 
                                    valid_neg_links, test_pos_links, test_neg_links, path, algorithm, models=models)       
    '''
    
    # property_scatter(args, device, dataset, known_links, valid_pos_links, 
    #                                valid_neg_links, test_pos_links, test_neg_links, path, algorithm, models=models)      
   
    
    # just analysis on different the tradic prediction and tradic count with logistic regression
    # import ipdb; ipdb.set_trace()
    # analysis_overlapping(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)      
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

    
def algorithm_check(algorithms, args):
    if args.dataset_name == 'ogbl-ddi':
        new_algorithms = []
        for algorithm in algorithms:
            if 'homo' in algorithm and len(algorithm) <= 2:
                continue
            else:
                new_algorithms.append(algorithm)
        return new_algorithms
    else:
        return algorithms    
def analyze_model_performance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path):
    pos_preds, neg_preds, results, ranks = load_ablation(args, args.dataset_name, is_std=True, prefix=None)
    plot_feat_struct_ablation_performance(results, args.dataset_name)
    
    # import ipdb; ipdb.set_trace()
    pos_preds, neg_preds, results, ranks = load_seal(args, args.dataset_name, is_std=True, prefix=None)
    plot_seal_ablation_performance(results, args.dataset_name)
    # import ipdb; ipdb.set_trace()

    print()


def calculate_degree(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path):
    known_links = known_links.T
    num_nodes = torch.max(known_links).item() + 1
    degrees = torch_geometric.utils.degree(known_links[0], num_nodes)

    degrees = degrees.cpu().numpy()
    with open(f"intermedia_result/degrees/{args.dataset_name}.txt", "wb") as f:
        pickle.dump(degrees, f)
    
def best_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path, K = 3):
    # analysis one will focus on drawing results on different hops
    # we will use the same aggregation function and the same distance function
    num_hops = [0, 1, 2, 3, 4] 
    norm_types = ["D2AD2", "A", "D-1A"]
    dis_func_names = ["l2", "cos", "jaccard"]
    # result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
    #                    "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
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
                # pos_preds, neg_preds, result = run_single_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
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

def best_model(args, data_name, model_names, prefix=None):
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
        
    preds_dict, ranks_dict, result_dict = {}, {}, {}
    # pos_preds, neg_preds, results, ranks = load_seal(args, data_name, False, selected_metric=result_key, prefix=None)
    # pos_preds, neg_preds, results, ranks = load_ablation(args, data_name, name="seal_drnl", selected_metric="Hits@100", prefix=None)
    # import ipdb; ipdb.set_trace()
    
    # TODO: whether move here
    if "mlp" in model_names and args.dataset_name == "ogbl-ddi":
        model_names = [model_name for model_name in model_names if model_name != 'mlp']
        
    
    for model_name in model_names:
        if model_name == "ablation":
            # load results for feature and strucutre ablation study
            ablation_preds_dict, ablation_ranks_dict, ablation_results_dict = load_ablation(args, args.dataset_name, selected_metric=result_key, prefix=None)
            preds_dict.update(ablation_preds_dict)
            ranks_dict.update(ablation_ranks_dict)
            result_dict.update(ablation_results_dict)
            continue
        # print(model_name)
        pos_preds, neg_preds, rank, single_results = load_results_with_multiseed(args, data_name, model_name, is_single=True, is_std=False, prefix=prefix)
        if isinstance(pos_preds, torch.Tensor):
            pos_preds = pos_preds.cpu().numpy()
            neg_preds = neg_preds.cpu().numpy()
    
        single_result = single_results[result_key]
        _, _, rank, multi_results = load_results_with_multiseed(args, data_name, model_name, is_single=True, is_std=False, prefix=prefix)
        multi_result = multi_results[result_key]
        
        result_dict[model_name] = multi_result
        preds_dict[model_name] = [pos_preds, neg_preds]    
        ranks_dict[model_name] = rank
        
    return preds_dict, result_dict, ranks_dict
    
def best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, num_algo = 3):
    results_dict = {}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    
    best_candidates_results, best_candidates_names, best_preds = [-1 for i in range(num_algo)], [" " for i in range(num_algo)], [-1 for i in range(num_algo)]
    min_result = -1
    
    # pos_preds, neg_preds = feature_importance_tradic_logistic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
    pos_preds, neg_preds = tradic_logistic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
    # for logisitic one
    result = get_results(args, pos_preds, neg_preds, result_key)
    if result > min_result:
        idx = best_candidates_results.index(min_result)
        best_candidates_results[idx] = result
        best_candidates_names[idx] = f"{args.algorithm}_LR"
        best_preds[idx] = [pos_preds, neg_preds]
        min_result = min(best_candidates_results)

    pos_preds_list, neg_preds_list = tradic_count_logits(known_links, dataset, path, args, test_pos_links, test_neg_links, is_test=1) 
    # for heurisitc one
    # [num_edge, num_catagories]
    num_hops = pos_preds_list.shape[1]
    for i in range(num_hops):
        pos_preds, neg_preds = pos_preds_list[:, i], neg_preds_list[:, i]
        if args.is_old_neg == 0:
            neg_preds = np.reshape(neg_preds, [pos_preds.shape[0], -1])
            
        result = get_results(args, pos_preds, neg_preds, result_key)
        if result > min_result:
            idx = best_candidates_results.index(min_result)
            best_candidates_results[idx] = result
            best_candidates_names[idx] = f"{args.algorithm}_{i}"
            best_preds[idx] = [pos_preds, neg_preds]
            min_result = min(best_candidates_results)
    
    tradic_preds, tradic_results = {}, {}
    sorted_idx = np.argsort(best_candidates_results)
    # import ipdb; ipdb.set_trace()
    # for idx, algorithm_name in enumerate(best_candidates_names):
    for i in range(len(sorted_idx)):
        idx = sorted_idx[i]
        algorithm_name = best_candidates_names[idx]
        tradic_preds[algorithm_name] = best_preds[idx]
        tradic_results[algorithm_name] = best_candidates_results[idx]
    # return best_candidates_n'ames,  best_candidates_results, best_preds
    return tradic_preds, tradic_results

def best_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path, num_algo=2):
    # A = torch.sparse_coo_tensor(known_links.T, torch.ones([np.max(known_links.shape)]).to(device), known_links.shape)
    # current, we only have PPR
    # TODO: katz has a little bit problem
    results_dict = {}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    
    all_algorithms = ["PPR", "SimRank", "Katz"] #
    if args.is_old_neg == 0:
        test_neg_links = torch.reshape(test_neg_links, [-1, 2])
    num_pos_edges, num_neg_edges = test_pos_links.shape[0], test_neg_links.shape[0]
    links = torch.cat([test_pos_links, test_neg_links], dim=0)
    
    is_old_neg = args.is_old_neg if args.dataset_name != "ogbl-citation2" else 0
    preds_dict, results_dict = {}, {}
    preds_name_dict = {"PPR": "ppr_preds", "SimRank": "simrank_preds", "Katz": "katz_preds"}
    function_name_dict = {"PPR": "PPR_correct", "SimRank": "SimRank_correct", "Katz": "Katz"}
    # import ipdb; ipdb.set_trace()
    for algo_name in all_algorithms:
        preds_name = preds_name_dict[algo_name]
        function_name = function_name_dict[algo_name]
        if args.is_load:
            try:
                with open(f"intermedia_result/{preds_name}/{args.dataset_name}_{args.is_old_neg}_results.txt", "rb") as f:
                    preds = pickle.load(f)
                pos_preds, neg_preds = preds["pos"], preds["neg"]
                result = get_results(args, pos_preds, neg_preds, result_key)
            except:
                preds = eval(function_name)(known_links, links, args.dataset_name, is_old_neg=is_old_neg)
                preds = preds.numpy()
                preds_list = np.split(preds, [num_pos_edges])
                pos_preds, neg_preds = preds_list[0], preds_list[1]
                result = get_results(args, pos_preds, neg_preds, result_key)
                with open(f"intermedia_result/{preds_name}/{args.dataset_name}_{args.is_old_neg}_results.txt", "wb") as f:
                    pickle.dump({"pos": pos_preds, "neg": neg_preds}, f)
        else:
            preds = eval(function_name)(known_links, links, args.dataset_name, is_old_neg=is_old_neg)
            preds = preds.numpy()
            preds_list = np.split(preds, [num_pos_edges])
            pos_preds, neg_preds = preds_list[0], preds_list[1]
            result = get_results(args, pos_preds, neg_preds, result_key)
            with open(f"intermedia_result/{preds_name}/{args.dataset_name}_{args.is_old_neg}_results.txt", "wb") as f:
                pickle.dump({"pos": pos_preds, "neg": neg_preds}, f)
    
        preds_dict[algo_name] = [pos_preds, neg_preds]
        results_dict[algo_name] = result
                        
    best_candidates_results, best_candidates_names, best_preds = [-1 for i in range(num_algo)], [" " for i in range(num_algo)], [-1 for i in range(K)]
    min_result = -1
    
    for algo_name in results_dict.keys():
        result = results_dict[algo_name]
        # len(np.unique(pos_basis_heuristic_preds))
        # print(len(np.unique(preds_dict[algo_name][0])))
        if len(np.unique(preds_dict[algo_name][0])) == 1:
            continue

        if result > min_result:
            if algo_name in best_candidates_names or result in best_candidates_results:
                continue
            idx = best_candidates_results.index(min_result)
            best_candidates_results[idx] = result
            best_candidates_names[idx] = algo_name
            best_preds[idx] = [pos_preds, neg_preds]
            min_result = min(best_candidates_results)
            
    global_preds, global_results = {}, {}
    sorted_idx = np.argsort(best_candidates_results)
    # import ipdb; ipdb.set_trace()
    for i in range(len(sorted_idx)):
        idx = sorted_idx[i]
        algorithm_name = best_candidates_names[idx]
        global_preds[algorithm_name] = preds_dict[algorithm_name]  # best_preds[idx]
        global_results[algorithm_name] = results_dict[algorithm_name]   # best_candidates_results[idx]
        
    return global_preds, global_results
            




def inner_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path, K = 3):
    num_hops = [0] 
    norm_types = ["A"]
    dis_func_names = ["l2", "cos"]
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    
    result_key = result_key_dict[args.dataset_name]
    
    best_candidates_results, best_candidates_names, best_preds = [-1 for i in range(K)], [" " for i in range(K)], [-1 for i in range(K)]
    min_result = -1
    best_preds = {}
    for dis_func_name in dis_func_names:
        for norm_type in norm_types:
            for num_hop in num_hops:
                args.is_feature = 1
                args.dis_func_name = dis_func_name
                args.num_hops = num_hop
                args.adj_norm = norm_type
                # pos_preds, neg_preds, result = run_single_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
                try:
                    pos_preds, neg_preds, result = run_single_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
                except:
                    continue
            
                result = result[result_key]
                # update
                best_preds[f"{dis_func_name}"] = [pos_preds, neg_preds]
    return best_preds
    
def inner_tradic(args, device, dataset, known_links, test_pos_links, test_neg_links, path, num_algo = 3):
    results_dict = {}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    
    best_preds = {}
    
    
    pos_preds_list, neg_preds_list = tradic_count_logits(known_links, dataset, path, args, test_pos_links, test_neg_links, is_test=1) 
    best_preds["CN"] = [pos_preds_list[:, 0], neg_preds_list[:, 0]]
    # train_edge_value = np.zeros([num_neighbors])
    # import ipdb; ipdb.set_trace()
    edge_index, edge_value = known_links.T, torch.zeros(np.max(known_links.shape)).to(device)
    num_nodes = torch.max(known_links).item() + 1
    # A = torch.sparse_coo_tensor(edge_index, edge_value, torch.Size([num_nodes, num_nodes]))
    # import ipdb; ipdb.set_trace()
    A = sp.coo_matrix((edge_value.cpu().numpy(), (edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())), shape=(num_nodes, num_nodes)).tocsr()
    pos_preds, _ = generalized_CN_new(A, A, test_pos_links, "RA")
    neg_preds, _ = generalized_CN_new(A, A, test_neg_links, "RA")
    # import ipdb; ipdb.set_trace()
    # pos_preds, neg_preds = torch.from_numpy(pos_preds).to(device), torch.from_numpy(neg_preds).to(device)
    best_preds["RA"] = [pos_preds, neg_preds]
    
    return best_preds

def inner_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path, num_algo=2):
    # TODO: katz has a little bit problem
    results_dict = {}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    
    all_algorithms = ["PPR", "Katz"] 
    if args.is_old_neg == 0:
        test_neg_links = torch.reshape(test_neg_links, [-1, 2])
    num_pos_edges, num_neg_edges = test_pos_links.shape[0], test_neg_links.shape[0]
    links = torch.cat([test_pos_links, test_neg_links], dim=0)
    
    is_old_neg = args.is_old_neg if args.dataset_name != "ogbl-citation2" else 0
    preds_dict, results_dict = {}, {}
    preds_name_dict = {"PPR": "ppr_preds", "SimRank": "simrank_preds", "Katz": "katz_preds"}
    function_name_dict = {"PPR": "PPR_correct", "SimRank": "SimRank_correct", "Katz": "Katz"}
    for algo_name in all_algorithms:
        preds_name = preds_name_dict[algo_name]
        function_name = function_name_dict[algo_name]
        if args.is_load:
            try:
                with open(f"intermedia_result/{preds_name}/{args.dataset_name}_{args.is_old_neg}_results.txt", "rb") as f:
                    preds = pickle.load(f)
                pos_preds, neg_preds = preds["pos"], preds["neg"]
                result = get_results(args, pos_preds, neg_preds, result_key)
            except:
                preds = eval(function_name)(known_links, links, args.dataset_name, is_old_neg=is_old_neg)
                preds = preds.numpy()
                preds_list = np.split(preds, [num_pos_edges])
                pos_preds, neg_preds = preds_list[0], preds_list[1]
                result = get_results(args, pos_preds, neg_preds, result_key)
                with open(f"intermedia_result/{preds_name}/{args.dataset_name}_{args.is_old_neg}_results.txt", "wb") as f:
                    pickle.dump({"pos": pos_preds, "neg": neg_preds}, f)
        else:
            preds = eval(function_name)(known_links, links, args.dataset_name, is_old_neg=is_old_neg)
            preds = preds.numpy()
            preds_list = np.split(preds, [num_pos_edges])
            pos_preds, neg_preds = preds_list[0], preds_list[1]
            result = get_results(args, pos_preds, neg_preds, result_key)
            with open(f"intermedia_result/{preds_name}/{args.dataset_name}_{args.is_old_neg}_results.txt", "wb") as f:
                pickle.dump({"pos": pos_preds, "neg": neg_preds}, f)
    
        preds_dict[algo_name] = [pos_preds, neg_preds]
        
    return preds_dict
            


def analyze_homo_feature_importance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path):
    # just analyze on different tradic algorithm on accuracy
    num_hops = [0, 1, 2, 3, 4] 
    norm_types = ["D2AD2", "A", "D-1A"]
    dis_func_names = ["l2", "cos", "jaccard"]
    # result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
    
    #                    "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    best_result = -1
    for dis_func_name in dis_func_names:
        for norm_type in norm_types:
            pos_val_preds_list, neg_val_preds_list, pos_test_preds_list, neg_test_preds_list = [], [], [], []
            for num_hop in num_hops:
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






def default_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path, K = 3):
    args.dis_func_name = "cos"
    args.is_feature = 1
    args.adj_norm = "A"
    args.num_hops = 0
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    pos_preds, neg_preds, result = run_single_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
    algorithm_name = "homo" 
    # {args.dis_func_name}_{args.num_hops}

    homo_preds, homo_results = {}, {}
    homo_preds[algorithm_name] = [pos_preds, neg_preds]
    homo_results[algorithm_name] = result
    return homo_preds, homo_results

def default_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, num_algo = 2):
    results_dict = {}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    pos_preds_list, neg_preds_list = tradic_count_logits(known_links, dataset, path, args, test_pos_links, test_neg_links, is_test=1) 
    pos_preds, neg_preds = pos_preds_list[:, 0], neg_preds_list[:, 0]
    num_hops = pos_preds_list.shape[1]
    result = get_results(args, pos_preds, neg_preds, result_key)
    
    algorithm_name = "CN"
    tradic_preds, tradic_results = {}, {}
    tradic_preds[algorithm_name] = [pos_preds, neg_preds]
    tradic_results[algorithm_name] = result
    
    return tradic_preds, tradic_results



def default_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path, num_algo=2):
    # A = torch.sparse_coo_tensor(known_links.T, torch.ones([np.max(known_links.shape)]).to(device), known_links.shape)
    # current, we only have PPR
    results_dict = {}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50", "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    num_pos_edges, num_neg_edges = test_pos_links.shape[0], test_neg_links.shape[0]
    links = torch.cat([test_pos_links, test_neg_links], dim=0)
    
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    # "PPR", "SimRank",
    data_algo_dict = {"Cora": "SimRank", "Citeseer": "SimRank", "Pubmed": "SimRank", "ogbl-collab": "Katz",  "ogbl-ddi": "PPR", "ogbl-ppa": "PPR"}
    preds_name_dict = {"PPR": "ppr_preds", "SimRank": "simrank_preds", "Katz": "katz_preds"}
    function_name_dict = {"PPR": "PPR_correct", "SimRank": "SimRank_correct", "Katz": "Katz"}
        
    algorithm = data_algo_dict[args.dataset_name]
    preds_name = preds_name_dict[algorithm]
    function_name = function_name_dict[algorithm]
    
    # algorithm = "Katz"
    # preds_name = "katz_preds"
    # function_name = "Katz"
    
    
    # algorithm = "PPR"
    # preds_name = "ppr_preds"
    # function_name = "PPR_correct"
    
    if args.is_load:
        try:
            with open(f"intermedia_result/{preds_name}/{args.dataset_name}_{args.is_old_neg}_results.txt", "rb") as f:
                preds = pickle.load(f)
            pos_preds, neg_preds = preds["pos"], preds["neg"]
            result = get_results(args, pos_preds, neg_preds, result_key)
        except:
            preds = eval(function_name)(known_links, links, args.dataset_name, is_old_neg=args.is_old_neg)
            preds = preds.numpy()
            preds_list = np.split(preds, [num_pos_edges])
            pos_preds, neg_preds = preds_list[0], preds_list[1]
            result = get_results(args, pos_preds, neg_preds, result_key)
            with open(f"intermedia_result/{preds_name}/{args.dataset_name}_{args.is_old_neg}_results.txt", "wb") as f:
                pickle.dump({"pos": pos_preds, "neg": neg_preds}, f)
    else:
        preds = eval(function_name)(known_links, links, args.dataset_name, is_old_neg=args.is_old_neg)
        preds = preds.numpy()
        preds_list = np.split(preds, [num_pos_edges])
        pos_preds, neg_preds = preds_list[0], preds_list[1]
        result = get_results(args, pos_preds, neg_preds, result_key)
        with open(f"intermedia_result/{preds_name}/{args.dataset_name}_{args.is_old_neg}_results.txt", "wb") as f:
            pickle.dump({"pos": pos_preds, "neg": neg_preds}, f)
    
    preds_dict, results_dict = {}, {}
    preds_dict[algorithm] = [pos_preds, neg_preds]
    results_dict[algorithm] = result

    return preds_dict, results_dict
    


def analyze_tradic_feature_importance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path):
    # just analyze on different tradic algorithm on accuracy
    algorithms = ["CN", "RA", "PA"] 
    results_dict = {}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    
    # result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
    #                    "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    
    for algorithm in algorithms:
        results = []
        args.algorithm = algorithm
        pos_preds, neg_preds, weights = feature_importance_tradic_logistic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
        result = get_results(args, pos_preds, neg_preds, result_key)
        plot_tradic_feature_importance(weights, algorithm, args.dataset_name, result_key, result)
        

                    
def analyze_difference(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # tradic_preds, tradic_results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links,test_pos_links, test_neg_links, path)
    # import ipdb; ipdb.set_trace()
    preds_list, results_list = [], []
    # algorithms = ["homo", "CN"]
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]

    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
        elif algorithm == "global":
            preds, results = best_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
        elif algorithm == "model":
            assert models != None, "No model defined" 
            preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
            # for model in models:
            #     check_key(args, args.dataset_name, model, prefix=None)
            # exit()
            # import ipdb; ipdb.set_trace()
            print()
        else:
            args.algorithm = "CN"
            preds, results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
            args.algorithm = "RA"
            preds, results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
            args.algorithm = "PA"
            preds, results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path)
            exit()
        # import ipdb; ipdb.set_trace()
        for key in preds.keys():
            if isinstance(preds[key][0], torch.Tensor):
                preds[key][0] = preds[key][0].cpu().numpy()
                preds[key][1] = preds[key][1].cpu().numpy()   
            if args.is_old_neg == 0:
                preds[key][1]= np.reshape(preds[key][1], [preds[key][0].shape[0], -1])
            
        preds_list.append(preds)
        results_list.append(results)
    preds1, preds2 = preds_list[0], preds_list[1]
    results1, results2 = results_list[0], results_list[1]
    algo_names1, algo_names2 = list(preds1.keys()), list(preds2.keys())

    result_keys = ["TP", "FP", "TN", "FN"]
    num1, num2 = len(preds1.keys()), len(preds2.keys())
    results_dict = {key: np.zeros([num1, num2]) for key in result_keys}
    for idx1, pred1 in enumerate(preds1.values()):
        pos_preds1, neg_preds1 = pred1[0], pred1[1]
        for idx2, pred2 in enumerate(preds2.values()):
            pos_preds2, neg_preds2 = pred2[0], pred2[1]    
            num_pos = pos_preds2.shape[0]
            # import ipdb; ipdb.set_trace()
            hit_result1, mrr_result1, correct_index1 = get_rank_new(pos_preds1, neg_preds1, args.is_old_neg, K=K)
            hit_result2, mrr_result2, correct_index2 = get_rank_new(pos_preds2, neg_preds2, args.is_old_neg, K=K)
            F1_results = F1_new(correct_index1, correct_index2, num_pos)
            # F1_results = F1(rank1, rank2, num_pos=num_pos, K=K)
            
            for key in result_keys:            
                results_dict[key][idx1, idx2] = F1_results[key]
            
               
    F1_compare(results_dict, algo_names1, algo_names2, args.dataset_name, result_key, algorithms=algorithms)
    
    args.num_bin = 5
    preds_dict = {**preds1, **preds2}
    results_dict = {**results1, **results2}
    # import ipdb; ipdb.set_trace()
    for compare_algorithm_name in preds_dict.keys():
        if compare_algorithm_name in models_list or compare_algorithm_name.find("feat") != -1 or  compare_algorithm_name.find("struct") != -1:
            continue
        
        seperate_results = {}
        # compare_algorithm_name = 'CN'
        pos_compare_preds, neg_compare_preds = preds_dict[compare_algorithm_name][0], preds_dict[compare_algorithm_name][1]
        compare_result = results_dict[compare_algorithm_name]        
        split_values, num_pos_values = equal_split(pos_compare_preds, args.num_bin)
        # import ipdb; ipdb.set_trace()
        _, num_neg_values = count_bin(neg_compare_preds, split_values)
        masks = generate_split_masks(pos_compare_preds, split_values)
        
        # check
        print(f"num edge: {np.sum(num_pos_values)} num edge: {pos_compare_preds.shape[0]}")
        # import ipdb; ipdb.set_trace()
        '''
        for i, num_pos_value in enumerate(num_pos_values):
            try:
                print(f"num: {num_pos_value}")
                print(f"mask: {np.sum(masks[i])}")
            except:
                import ipdb; ipdb.set_trace()
        '''     
        compare_results = seperate_accuracy(pos_compare_preds, neg_compare_preds, masks, args.is_old_neg, K)
        for algorithm_name in preds_dict.keys():
            if algorithm_name == compare_algorithm_name:
                continue
            pos_preds, neg_preds = preds_dict[algorithm_name][0], preds_dict[algorithm_name][1]
            result = results_dict[algorithm_name]
            
            results = seperate_accuracy(pos_preds, neg_preds, masks, args.is_old_neg, K)
            seperate_results[algorithm_name] = results
        
        # import ipdb; ipdb.set_trace()
        plot_compare_performance(seperate_results, compare_results, compare_algorithm_name, results_dict, args.dataset_name, split_values, num_pos_values, num_neg_values, result_key, algorithms)


def model_complementary(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    hard_ratio = 0.2
    num_algo = 2
    
    preds_list, results_list, neg_ranks_list, correct_indexes_list = [], [], [], []
    # algorithms = ["homo", "CN"]
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
            # import ipdb; ipdb.set_trace()
        elif algorithm == "global":
            preds, results = best_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
        elif algorithm == "model":
            assert models != None, "No model defined" 
            preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
            print()
        else:
            args.algorithm = "CN"
            preds, results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
        
        neg_ranks, correct_indexes = {}, {}
        for key in preds.keys():
            if isinstance(preds[key][0], torch.Tensor):
                preds[key][0] = preds[key][0].cpu().numpy()
                preds[key][1] = preds[key][1].cpu().numpy()   
            if args.is_old_neg == 0:
                preds[key][1]= np.reshape(preds[key][1], [preds[key][0].shape[0], -1])
            
            _, _, correct_index = get_rank_new(preds[key][0], preds[key][1], args.is_old_neg, K)
            neg_rank = get_rank_single(args, preds[key][1])
            neg_ranks[key] = neg_rank
            correct_indexes[key] = correct_index
            
        # draw bin and then draw the distribution
        neg_ranks_list.append(neg_ranks)      
        correct_indexes_list.append(correct_indexes)
        preds_list.append(preds)
        results_list.append(results)
    
    preds_dict = {}
    results_dict = {}
    neg_ranks_dict = {}
    correct_indexes_dict = {}
    for preds, results, neg_ranks, correct_indexes in zip(preds_list, results_list, neg_ranks_list, correct_indexes_list):
        preds_dict.update(preds)
        results_dict.update(results)
        neg_ranks_dict.update(neg_ranks)
        correct_indexes_dict.update(correct_indexes)
    
    plot_hard_negative(preds_dict, correct_indexes_dict, neg_ranks_dict, args, args.dataset_name, K, hard_ratio)
    


def model_both_pos_neg(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    hard_ratio = 0.2
    num_algo = 1
    
    if args.dataset_name in ["ogbl-ddi", "ogbl-ppa"]:
        algorithms = ["CN", "global"]
    else:
        algorithms = ["CN", "global", "homo"]
        
    # algorithms = ["homo", "CN"]
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    
    # , ranks_list, selected_mask = [], [], [], []
    
    preds_dict, results_dict, ranks_dict, selected_masks_dict = {}, {}, {}, {}

    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = default_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
        elif algorithm == "global":
            preds, results = default_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
            # print(f"{args.dataset_name}: {results}")
            # exit()
        else:
            args.algorithm = "CN"
            preds, results = default_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
        
        # TODO: here we only select one for each algorithm    
        for key in preds.keys():
            if isinstance(preds[key][0], torch.Tensor):
                preds[key][0] = preds[key][0].cpu().numpy()
                preds[key][1] = preds[key][1].cpu().numpy()   
            if args.is_old_neg == 0:
                preds[key][1]= np.reshape(preds[key][1], [preds[key][0].shape[0], -1])
            
            rank = get_rank_single(args, np.concatenate([preds[key][0], preds[key][1]], axis=-1))
            num_selected = int(rank.shape[0] * hard_ratio)
            selected_mask = (rank <= num_selected)   
            # _, _, correct_index = get_rank_new(preds[key][0], preds[key][1], args.is_old_neg, K)
            # import ipdb; ipdb.set_trace()
            ranks_dict[algorithm] = rank
            selected_masks_dict[algorithm] = selected_mask
            preds_dict[algorithm] = preds
            results_dict[algorithm] = results
        # draw bin and then draw the distribution
        # neg_ranks_list.append(neg_ranks)      
        # correct_indexes_list.append(correct_indexes)
        # preds_list.append(preds)
        # results_list.append(results)
        # import ipdb; ipdb.set_trace()
    selected_masks_dict["local"] = selected_masks_dict.pop("CN")
    ranks_dict["local"] = ranks_dict.pop("CN")
    preds_dict["local"] = preds_dict.pop("CN")

    if args.dataset_name not in ["ogbl-ddi", "ogbl-ppa"]:
        selected_masks_dict["feat"] = selected_masks_dict.pop("homo")
        ranks_dict["feat"] = ranks_dict.pop("homo")
        preds_dict["feat"] = preds_dict.pop("homo")
    
    transfer_dict = {'global': "GSP", 'local': "LSP", 'feat': "FP"}
    keys = list(ranks_dict.keys())
    for original_key in keys:
        print(original_key)
        key = transfer_dict[original_key]
        selected_masks_dict[key] = selected_masks_dict.pop(original_key)
        ranks_dict[key] = ranks_dict.pop(original_key)
        preds_dict[key] = preds_dict.pop(original_key)
    # import ipdb; ipdb.set_trace()
    
    plot_rank_compare(preds_dict, selected_masks_dict, ranks_dict, args, args.dataset_name, num_selected, 4)






def model_both_pos_neg_same_cate(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    hard_ratio = 0.2
    num_algo = 1
    
    if args.dataset_name in ["ogbl-ddi", "ogbl-ppa"]:
        algorithms = ["CN", "global"]
    else:
        algorithms = ["CN", "global", "homo"]
        
    # algorithms = ["homo", "CN"]
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    
    # , ranks_list, selected_mask = [], [], [], []
    preds_dict, results_dict, ranks_dict, selected_masks_dict = {}, {}, {}, {}
    
    homo_preds = inner_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
    tradic_preds = inner_tradic(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
    global_preds = inner_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
    
    # import ipdb; ipdb.set_trace()
    if args.dataset_name not in ["ogbl-ddi", "ogbl-ppa"]:
        names = ["FP", "LSP", "GSP"]
        preds_list = [homo_preds, tradic_preds, global_preds]
    else:
        names = ["LSP", "GSP"]
        preds_list = [tradic_preds, global_preds]
    
    preds_dict_dict, ranks_dict_dict, selected_masks_dict_dict = {}, {}, {}
    for name, preds in zip(names, preds_list):
        preds_dict, ranks_dict, selected_masks_dict = {}, {}, {}
        for key in preds.keys():
            if isinstance(preds[key][0], torch.Tensor):
                preds[key][0] = preds[key][0].cpu().numpy()
                preds[key][1] = preds[key][1].cpu().numpy()   
            if args.is_old_neg == 0:
                preds[key][1]= np.reshape(preds[key][1], [preds[key][0].shape[0], -1])
            
            rank = get_rank_single(args, np.concatenate([preds[key][0], preds[key][1]], axis=-1))
            num_selected = int(rank.shape[0] * hard_ratio)
            selected_mask = (rank <= num_selected)   
            # _, _, correct_index = get_rank_new(preds[key][0], preds[key][1], args.is_old_neg, K)
            # import ipdb; ipdb.set_trace()
            ranks_dict[key] = rank
            selected_masks_dict[key] = selected_mask
            preds_dict[key] = [preds[key][0], preds[key][1]]
        preds_dict_dict[name] = preds_dict
        selected_masks_dict_dict[name] = selected_masks_dict
        ranks_dict_dict[name] = ranks_dict
    plot_rank_compare_same_cate(preds_dict_dict, selected_masks_dict_dict, ranks_dict_dict, args, args.dataset_name, num_selected, 4)
        
        
    # transfer_dict = {'global': "GSP", 'local': "LSP", 'feat': "FP"}
    # keys = list(ranks_dict.keys())
    # for original_key in keys:
    #     print(original_key)
    #     key = transfer_dict[original_key]
    #     selected_masks_dict[key] = selected_masks_dict.pop(original_key)
    #     ranks_dict[key] = ranks_dict.pop(original_key)
    #     preds_dict[key] = preds_dict.pop(original_key)
    # import ipdb; ipdb.set_trace()
    
    # plot_rank_compare_same_cate(preds_dict, selected_masks_dict, ranks_dict, args, args.dataset_name, num_selected, 4)





def model_correctness(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # similar with the negative ones, but analysis what does the correct one in one algorithm
    num_algo = 2
    
    preds_list, results_list, wrong_pos_indexes_list, correct_indexes_list = [], [], [], []
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
            # import ipdb; ipdb.set_trace()
        elif algorithm == "global":
            preds, results = best_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
        elif algorithm == "model":
            assert models != None, "No model defined" 
            preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
            # print()
        else:
            args.algorithm = "CN"
            preds, results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
        
        wrong_pos_indexes, correct_indexes = {}, {}
        for key in preds.keys():
            if isinstance(preds[key][0], torch.Tensor):
                preds[key][0] = preds[key][0].cpu().numpy()
                preds[key][1] = preds[key][1].cpu().numpy()   
            if args.is_old_neg == 0:
                preds[key][1]= np.reshape(preds[key][1], [preds[key][0].shape[0], -1])
            pos_preds, neg_preds = preds[key][0], preds[key][1]
            _, _, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
            num_pos = pos_preds.shape[0]
            # TODO: do not know whether here is right
            wrong_pos_index = mask_to_index(~index_to_mask(correct_index, num_pos))  
            # import ipdb; ipdb.set_trace()
            # neg_rank = get_rank_single(args, preds[key][1])
            wrong_pos_indexes[key] = wrong_pos_index
            correct_indexes[key] = correct_index
        
        # draw bin and then draw the distribution
        wrong_pos_indexes_list.append(wrong_pos_indexes)      
        correct_indexes_list.append(correct_indexes)
        preds_list.append(preds)
        results_list.append(results)
    # import ipdb; ipdb.set_trace()
    
    preds_dict = {}
    results_dict = {}
    wrong_pos_indexes_dict = {}
    correct_indexes_dict = {}
    for preds, results, wrong_pos_indexes, correct_indexes in zip(preds_list, results_list, wrong_pos_indexes_list, correct_indexes_list):
        preds_dict.update(preds)
        results_dict.update(results)
        wrong_pos_indexes_dict.update(wrong_pos_indexes)
        correct_indexes_dict.update(correct_indexes)
    
    plot_where_positive(preds_dict, correct_indexes_dict, wrong_pos_indexes_dict, args, args.dataset_name, K)
    



def struct_feat_compare(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # similar with the negative ones, but analysis what does the correct one in one algorithm
    num_algo = 1
    
    preds_list, results_list, wrong_pos_indexes_list, correct_indexes_list = [], [], [], []
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    correct_masks_dict = {}
    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = default_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
        elif algorithm == "global":
            preds, results = default_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
        elif algorithm == "model":
            assert models != None, "No model defined" 
            preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
            # print()
        else:
            args.algorithm = "CN"
            preds, results = default_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
        
        for key in preds.keys():
            if isinstance(preds[key][0], torch.Tensor):
                preds[key][0] = preds[key][0].cpu().numpy()
                preds[key][1] = preds[key][1].cpu().numpy()   
            if args.is_old_neg == 0:
                preds[key][1]= np.reshape(preds[key][1], [preds[key][0].shape[0], -1])
            pos_preds, neg_preds = preds[key][0], preds[key][1]
            _, _, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
            num_pos = pos_preds.shape[0]
            # TODO: do not know whether here is right
            correct_mask = index_to_mask(correct_index, num_pos)
            # wrong_pos_index = mask_to_index(~)  
            # import ipdb; ipdb.set_trace()
            # neg_rank = get_rank_single(args, preds[key][1])
            # wrong_pos_indexes[key] = wrong_pos_index
            # correct_indexes[key] = correct_index
        
            correct_masks_dict[algorithm] = correct_mask
        # import ipdb; ipdb.set_trace()
    
    correct_masks_dict["local"] = correct_masks_dict.pop("CN")
    correct_masks_dict["feat"] = correct_masks_dict.pop("homo")
    
    global_correct_mask = correct_masks_dict["global"]
    local_correct_mask = correct_masks_dict["local"]
    feat_mask = correct_masks_dict["feat"]
    

    num_global, num_local, num_feat = torch.sum(global_correct_mask).item(), torch.sum(local_correct_mask).item(), torch.sum(feat_mask).item()
    # import ipdb; ipdb.set_trace()
    global_mask = (global_correct_mask & ~feat_mask)
    feat_global_mask = (~global_correct_mask & feat_mask)
    
    local_mask = (local_correct_mask & ~feat_mask)
    feat_local_mask = (~local_correct_mask & feat_mask)
    
    global_num, feat_global_num, local_num, feat_local_num = torch.sum(global_mask).item(), torch.sum(feat_global_mask).item(), torch.sum(local_mask).item(), torch.sum(feat_local_mask).item()
    # num_global_overlap, num_local_overlap = torch.sum(global_mask).item(), torch.sum(local_mask).item()
    # num_global_feat, num_feat_global = num_global_overlap / num_feat, num_global_overlap / num_global
    # num_local_feat, num_feat_global = num_local_overlap / num_feat, num_local_overlap / num_local
    
    global_acc, feat_global_acc, local_acc, feat_local_acc = global_num / num_global, feat_global_num / num_feat, local_num / num_local, feat_local_num / num_feat
    results = {"global_feat": global_acc, "feat_global": feat_global_acc, "local_feat": local_acc, "feat_local": feat_local_acc}
    print(results)
    with open(f"intermedia_result/result_overlap/{args.dataset_name}.txt", "wb") as f:
        pickle.dump(results, f)
        
        


def property_scatter(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # similar with the negative ones, but analysis what does the correct one in one algorithm
    num_algo = 2
    # only load heuristic
    preds_dict, results, ranks = load_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, 
                                algorithms, models=models, num_algo=num_algo)
    # import ipdb; ipdb.set_trace()
    plot_property_scatter(preds_dict, args, args.dataset_name)
    


def model_complementary_all(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # check whether all algorithms together can almost cover most nodes
    hard_ratio = 0.2
    num_algo = 2
    
    preds_list, results_list, hard_neg_masks_list, correct_indexes_list = [], [], [], []
    # algorithms = ["homo", "CN"]
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = default_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
            # preds, results = default_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
        elif algorithm == "global":
            preds, results = default_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
            # preds, results = default_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
        elif algorithm == "model":
            assert models != None, "No model defined" 
            preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
            # import ipdb; ipdb.set_trace()
            print()
        else:
            args.algorithm = "CN"
            preds, results = default_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
            # preds, results = default_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
        
        hard_neg_masks, correct_indexes = {}, {}
        for key in preds.keys():
            if isinstance(preds[key][0], torch.Tensor):
                preds[key][0] = preds[key][0].cpu().numpy()
                preds[key][1] = preds[key][1].cpu().numpy()   
            if args.is_old_neg == 0:
                preds[key][1]= np.reshape(preds[key][1], [preds[key][0].shape[0], -1])
            
            pos_preds, neg_preds = preds[key][0], preds[key][1]
            if args.is_old_neg and args.dataset_name != 'ogbl-citation2':
                num_neg = neg_preds.shape[0]
                num_hard = int(num_neg * hard_ratio)
                _, _, correct_index = get_rank_new(preds[key][0], preds[key][1], args.is_old_neg, K)
                neg_rank = get_rank_single(args, neg_preds) - 1
                hard_indices = np.where(neg_rank < num_hard)
                hard_mask = np.zeros([num_neg], dtype=bool)
                hard_mask[hard_indices] = True
            else:
                num_pos = neg_preds.shape[0]
                num_neg = num_pos
                hard_mask = np.zeros([num_pos], dtype=bool)
                _, _, correct_index = get_rank_new(preds[key][0], preds[key][1], args.is_old_neg, K)
                wrong_index = mask_to_index(~index_to_mask(correct_index, num_pos))  
                hard_mask[wrong_index] = True
            #     num_neg = neg_preds.shape[1]
            #     hard_mask = np.zeros(neg_preds.shape, dtype=bool)
            #     _, _, correct_index = get_rank_new(preds[key][0], preds[key][1], args.is_old_neg, K)
            #     num_hard = int(K * correct_index.shape[0])
            hard_neg_masks[key] = hard_mask            
            correct_indexes[key] = correct_index
        
        hard_neg_masks_list.append(hard_neg_masks)
        correct_indexes_list.append(correct_indexes)
        preds_list.append(preds)
        results_list.append(results)
    
    preds_dict = {}
    results_dict = {}
    hard_neg_masks_dict = {}
    correct_indexes_dict = {}
    for preds, results, hard_neg_masks, correct_indexes in zip(preds_list, results_list, hard_neg_masks_list, correct_indexes_list):
        preds_dict.update(preds)
        results_dict.update(results)
        hard_neg_masks_dict.update(hard_neg_masks)
        correct_indexes_dict.update(correct_indexes)
    
    # print(hard_neg_masks_dict.keys())
    
    algo_names = list(preds_dict.keys())
    num_algo = len(algo_names)
    hard_neg_record = np.zeros([num_algo, num_neg])
    remain_negs = []
    # masks = np.zeros([num_neg], dtype=bool)
    for idx, algo_name in enumerate(algo_names):
        hard_neg_masks = hard_neg_masks_dict[algo_name]
        if idx == 0:
            masks = hard_neg_masks
            num_hard_neg = np.sum(masks).item()
        else:
            masks = np.logical_and(masks, hard_neg_masks)
        remain_neg = np.sum(masks)
        remain_negs.append(remain_neg)
    
    remain_negs = np.array(remain_negs) / num_hard_neg   # hard_neg_record.shape[1]
    
    algo_names = list(hard_neg_masks_dict.keys())
    # plot_hard_negative(preds_dict, correct_indexes_dict, neg_ranks_dict, args, args.dataset_name, K, hard_ratio)
    # import ipdb; ipdb.set_trace()
    with open(f"intermedia_result/hard_negs/{args.dataset_name}_{algorithms[0]}.txt", "wb") as f:
        pickle.dump(remain_negs, f)
    
    # plot_decay(algo_names, remain_negs, args.dataset_name, args)
    

def model_complementary_all_new(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # check whether all algorithms together can almost cover most nodes
    hard_ratio = 0.2
    num_algo = 2
    
    preds_list, results_list, hard_neg_masks_list, correct_indexes_list = [], [], [], []
    # algorithms = ["homo", "CN"]
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    correct_masks_dict = {}
    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = default_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
        elif algorithm == "global":
            preds, results = default_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
        elif algorithm == "model":
            assert models != None, "No model defined" 
            preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
            # import ipdb; ipdb.set_trace()
            print()
        else:
            args.algorithm = "CN"
            preds, results = default_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
        
        for key in preds.keys():
            if isinstance(preds[key][0], torch.Tensor):
                preds[key][0] = preds[key][0].cpu().numpy()
                preds[key][1] = preds[key][1].cpu().numpy()   
            
            pos_preds, neg_preds = preds[key][0], preds[key][1]
            num_pos = pos_preds.shape[0]
            _, _, correct_index = get_rank_new(preds[key][0], preds[key][1], args.is_old_neg, K)
            correct_mask = index_to_mask(correct_index, num_pos)            
            correct_masks_dict[algorithm] = correct_mask
      
    # plot_hard_negative(preds_dict, correct_indexes_dict, neg_ranks_dict, args, args.dataset_name, K, hard_ratio)
    # import ipdb; ipdb.set_trace()
    cumsum_correct_dict = {}

    for idx, algorithm in enumerate(algorithms):
        if idx == 0:
            mask = correct_masks_dict[algorithm]
        else:
            mask = np.logical_or(mask, correct_masks_dict[algorithm])
        cumsum_correct_dict[algorithm] = torch.sum(mask).item() / num_pos
    # import ipdb; ipdb.set_trace()
    with open(f"intermedia_result/result_cumsum/{args.dataset_name}.txt", "wb") as f:
        pickle.dump(cumsum_correct_dict, f)
    
    # plot_decay(algo_names, remain_negs, args.dataset_name, args)




def model_complementary_all_ranks(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # check whether all algorithms together can almost cover most nodes
    hard_ratio = 0.2
    num_algo = 2
    
    preds_list, results_list, hard_neg_masks_list, correct_indexes_list = [], [], [], []
    # algorithms = ["homo", "CN"]
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
        elif algorithm == "global":
            preds, results = best_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
        elif algorithm == "model":
            assert models != None, "No model defined" 
            preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
            # import ipdb; ipdb.set_trace()
            print()
        else:
            args.algorithm = "CN"
            preds, results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
        
        hard_neg_masks, correct_indexes = {}, {}
        for key in preds.keys():
            if isinstance(preds[key][0], torch.Tensor):
                preds[key][0] = preds[key][0].cpu().numpy()
                preds[key][1] = preds[key][1].cpu().numpy()   
            if args.is_old_neg == 0:
                preds[key][1]= np.reshape(preds[key][1], [preds[key][0].shape[0], -1])
            
            pos_preds, neg_preds = preds[key][0], preds[key][1]
            if args.is_old_neg and args.dataset_name != 'ogbl-citation2':
                num_neg = neg_preds.shape[0]
                num_hard = int(num_neg * hard_ratio)
                _, _, correct_index = get_rank_new(preds[key][0], preds[key][1], args.is_old_neg, K)
                neg_rank = get_rank_single(args, neg_preds) - 1
                hard_indices = np.where(neg_rank < num_hard)
                hard_mask = np.zeros([num_neg], dtype=bool)
                hard_mask[hard_indices] = True
            else:
                num_pos = neg_preds.shape[0]
                num_neg = num_pos
                hard_mask = np.zeros([num_pos], dtype=bool)
                _, _, correct_index = get_rank_new(preds[key][0], preds[key][1], args.is_old_neg, K)
                wrong_index = mask_to_index(~index_to_mask(correct_index, num_pos))  
                hard_mask[wrong_index] = True
            hard_neg_masks[key] = hard_mask            
            correct_indexes[key] = correct_index
        
        hard_neg_masks_list.append(hard_neg_masks)
        correct_indexes_list.append(correct_indexes)
        preds_list.append(preds)
        results_list.append(results)
    
    preds_dict = {}
    results_dict = {}
    hard_neg_masks_dict = {}
    correct_indexes_dict = {}
    for preds, results, hard_neg_masks, correct_indexes in zip(preds_list, results_list, hard_neg_masks_list, correct_indexes_list):
        preds_dict.update(preds)
        results_dict.update(results)
        hard_neg_masks_dict.update(hard_neg_masks)
        correct_indexes_dict.update(correct_indexes)
    
    print(hard_neg_masks_dict.keys())
    # import ipdb; ipdb.set_trace()
    algo_names = list(preds_dict.keys())
    num_algo = len(algo_names)
    hard_neg_record = np.zeros([num_algo, num_neg])
    # import ipdb; ipdb.set_trace()

    algo_names = list(hard_neg_masks_dict.keys())
    # plot_hard_negative(preds_dict, correct_indexes_dict, neg_ranks_dict, args, args.dataset_name, K, hard_ratio)
        
    plot_decay_all_rank(algo_names, hard_neg_masks_dict, num_algo, algorithms, args.dataset_name, args)
    


def load_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models, num_algo=1):
    preds_list, results_list, ranks_list = [], [], []
    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
            ranks = None
        elif algorithm == "global":
            preds, results = best_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
            ranks = None
        elif algorithm == "model":
            assert models != None, "No model defined" 
            preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
            # for model in models:
            #     check_key(args, args.dataset_name, model, prefix=None)
            # exit()
            # import ipdb; ipdb.set_trace()
            # print()
        else:
            args.algorithm = "CN"
            preds, results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, num_algo=num_algo)
            ranks = None
        # import ipdb; ipdb.set_trace()
        if algorithm == "model":
            for key in preds.keys():
                if isinstance(ranks[key], torch.Tensor):
                    ranks[key] = ranks[key].cpu().numpy()
        else:
            ranks = {}
            for key in preds.keys():
                if isinstance(preds[key][0], torch.Tensor):
                    preds[key][0] = preds[key][0].cpu().numpy()
                    preds[key][1] = preds[key][1].cpu().numpy()   
                if args.is_old_neg:
                    rank = get_rank_single(args, np.concatenate([preds[key][0], preds[key][1]], axis=0))
                else:          
                    preds[key][1]= np.reshape(preds[key][1], [preds[key][0].shape[0], -1])
                    rank = get_rank_single(args, np.concatenate([np.expand_dims(preds[key][0], -1), preds[key][1]], axis=-1))
                ranks[key] = rank
            
        # DDI dataset do not have any feature
        if algorithm == "homo" and args.dataset_name == "ogbl-ddi":
            continue
        preds_list.append(preds)
        results_list.append(results)
        ranks_list.append(ranks)
        
    predicts_dict, results_dict, ranks_dict = {}, {}, {}
    for preds, results, ranks in zip(preds_list, results_list, ranks_list):
        # import ipdb; ipdb.set_trace()
        predicts_dict.update(preds)
        results_dict.update(results)
        ranks_dict.update(ranks)
    
    return predicts_dict, results_dict, ranks_dict
    
def analysis_overlapping(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path): 
    results_dict = {}
    # result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
    #                    "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    Ks = {"Cora": 100, "Citeseer": 100, "Pubmed": 100, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 100, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    get_ppr(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
    # neighbors, neighbor_weights = get_ppr_matrix(known_links, num_nodes, alpha=0.15, eps=5e-5)
    
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
    # homophily_tradic_compare(results_dict, tradic_algorithms, homo_algorithms, args.dataset_name, result_key)
    F1_compare(results_dict, tradic_algorithms, homo_algorithms, args.dataset_name, result_key)
    
            
    
def analysis_overlapping_performance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path): 
    # There are many overlapping betweetn the second and the first function
    # We turn different hyperparameter seperatelyto enable easy tuning
    args.num_bin = 5
    results_dict = {}
    # result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
    #                    "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                    "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 10, "ogbl-ppa": 100}
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
        '''
        for i, num_pos_value in enumerate(num_pos_values):
            try:
                print(f"num: {num_pos_value}")
                print(f"mask: {np.sum(masks[i])}")
            except:
                import ipdb; ipdb.set_trace()
        '''
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
        neg_preds, pos_preds = torch.squeeze(neg_preds), torch.squeeze(pos_preds)
        neg_preds = torch.reshape(neg_preds, [pos_preds.shape[0], -1])
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
    # result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
    #                    "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
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
            print(f"file {file_name} does not exist")
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
        neg_preds, pos_preds = torch.squeeze(neg_preds), torch.squeeze(pos_preds)
        neg_preds = torch.reshape(neg_preds, [pos_preds.shape[0], -1])
        # import ipdb; ipdb.set_trace()
        # pos_preds = torch.unsqueeze(pos_preds, dim=1) 
        # new_preds = torch.cat([pos_preds, neg_preds], dim=1)
        results = get_metric_score(pos_preds, neg_preds)
    elif args.dataset_name == "ogbl-citation2":
        pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
        results = get_metric_score(pos_preds, neg_preds)
    else:
        pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
        results = get_metric_score_origin(pos_preds, neg_preds)

    # with open(f"{data_path}/homo_result_{args.norm_type}_{args.is_old_neg}.txt", "wb") as f:
    return pos_preds, neg_preds, results
    
    
    
    
    
'''
def tradic_decay_ideal(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, models=None):
    # check whether there is overlapping between algorithms
    # if one algorithm is corrrect, then the result is correct
    
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    
    pos_preds_list, neg_preds_list = tradic_count_logits(known_links, dataset, path, args, test_pos_links, test_neg_links, is_test=1) 
    num_hops = pos_preds_list.shape[1]
    num_pos, num_neg = pos_preds_list.shape[0], neg_preds_list.shape[0]
    correct_record = np.zeros([num_pos]) != 0
    correct_ratios = []
    for i in range(num_hops):
        pos_preds, neg_preds = pos_preds_list[:, i], neg_preds_list[:, i]
        if args.is_old_neg == 0:
            neg_preds = np.reshape(neg_preds, [pos_preds.shape[0], -1])
        
        hit_result, mrr_result, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
        print(f"hit_result: {hit_result}, mrr_result: {mrr_result}")
        correct_mask = index_to_mask(correct_index, num_pos).numpy()
        correct_record = np.logical_or(correct_record, correct_mask)
        # import ipdb; ipdb.set_trace()
        correct_ratios.append(np.sum(correct_record) / num_pos)

    import ipdb; ipdb.set_trace()
    print()
    

def homo_decay_ideal(args, device, datasets, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, models=None):
    # check whether there is overlapping between algorithms
    # if one algorithm is corrrect, then the result is correct
    # add one thing new, we need to first checkwhich 
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
        
    num_hops = [0, 1, 2, 3, 4] 
    dis_func_names = ["l2", "cos", "jaccard"]
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    
    correct_ratios_dict = {}
    for dataset in datasets:
        first = True
        norm_type = "D-1A"  
        results = []
        num_hop = 0
        # We will select distance function via the peformance on the zero hop
        for dis_func_name in dis_func_names:
            args.is_feature = 1
            args.dis_func_name = dis_func_name
            args.num_hops = num_hop
            args.adj_norm = norm_type
            # pos_preds, neg_preds, result = run_single_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
            try:
                pos_preds, neg_preds, result = run_single_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path)       
            except:
                continue
            
            num_pos, num_neg = pos_preds.shape[0], neg_preds.shape[0]
            result = result[result_key]
            results.append(result)
            
        best_idx = np.argmax(results)
        best_dis_func = dis_func_names[best_idx]
        
        correct_record = np.zeros([num_pos]) != 0
        correct_ratios, results = [], []
        
        for num_hop in num_hops:
            args.is_feature = 1
            args.dis_func_name = best_dis_func
            args.num_hops = num_hop
            args.adj_norm = norm_type
            try:
                pos_preds, neg_preds, result = run_single_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path)       
            except:
                continue
            
            hit_result, mrr_result, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
            print(f"hit_result: {hit_result}, mrr_result: {mrr_result}")
            correct_mask = index_to_mask(correct_index, num_pos).cpu().numpy()
            correct_record = np.logical_or(correct_record, correct_mask)
            correct_ratios.append(np.sum(correct_record) / num_pos)
        correct_ratios_dict[dataset] = correct_ratios
    import ipdb; ipdb.set_trace()

    plot_homo_decay_ideal(args, correct_ratio_dict, result_key)
    print()
    '''