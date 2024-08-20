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
from data_analysis.function.heuristics import Katz, PPR_new, PPR_correct, SimRank_correct
from torch_geometric.utils import mask_to_index, index_to_mask
import torch_geometric
from data_analysis.homo_tradic_plugin import default_homo, default_tradic, default_global, best_homo, best_model, best_tradic, best_global, load_model, get_results, run_single_homo



'''
ogbl-collab: we usually include the validation edges in the training graph when doing testing.
ogbl-ddi: It doesn't have node features. There is a weak relationship between the validation and test performance.
ogbl-ppa: The node feature is the 58 dimensional one-hot vector. The MLP has very bad performance.
ogbl-citation2:  In the validation/test,  positive samples have specific negative samples.
'''




def run_model_analyze(args, device):
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
    
    # Katz(known_links, torch.cat([test_pos_links, test_neg_links], dim=0))
    # best_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
    # exit()
    # "homo", "CN" "global" "model" 
    # TODO: model may need provide candidates
    # algorithms = [["model", "homo", "CN", "global"]]

    # ["homo", "global"], ["global", "CN"], ["homo", "CN"] 
    # ["model", "CN"] 
    # "ablation", 
    # TODO: load ablation has some issue
    # seal is not included into analysis
    # "seal",
    
    # "mlp",
    models = ["mlp", "gcn", "sage", "buddy", "neognn", "ncnc"]  # , "neognn"
    # models = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # models = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    
    # algorithms = [["homo", "global"], ["global", "CN"], ["homo", "CN"]]
    # algorithms = [["CN", "homo"]]
    # algorithms = ["homo", "global", "CN"]
    algorithms = ["model"]


    '''
    evaluate_seperete_performance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)
    analyze_pred_correlation(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)
    '''
    # analyze_pairwise_hard(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)      
    analyze_pairwise_hard(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)      
    
    # analyze_triple_hard(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)
    
    # find_majority(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)
    # find_train_majority(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)


    base_models = ['empty', "sage", 'mlp']
    # base_model = "empty" # 'gcn' 'homo' 'local' 'global's
    basis_heuristics = ["homo", 'local', 'global']  
    # for algorithm in algorithm:
    
    '''
    for base_model in base_models:
        for basis_heusistic in basis_heuristics:
            # print(f"{base_model} {basis_heusistic}") 
            if basis_heusistic == "homo" and args.dataset_name == "ogbl-ddi":
                continue
            if base_model == 'mlp' and args.dataset_name == "ogbl-ddi":
                continue
            args.num_bin = 4
            args.base_model, args.basis_heuristic = base_model, basis_heusistic
            print(basis_heusistic)
            analyze_difference_with_base_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)       
    '''
    
    '''
    for base_model in base_models:
        for basis_heusistic1 in basis_heuristics:
            for basis_heusistic2 in basis_heuristics:
                base_model = "empty"
                basis_heusistic1 = "homo"
                basis_heusistic2 = "local"
                
                print(f"{base_model} {basis_heusistic1} {basis_heusistic2}")
                if basis_heusistic1 == basis_heusistic2:
                    continue
                if (basis_heusistic1 == "homo" or basis_heusistic2 == "homo") and args.dataset_name == "ogbl-ddi":
                    continue
                if base_model == 'mlp' and args.dataset_name == "ogbl-ddi":
                    continue
                args.base_model, args.basis_heuristic1, args.basis_heuristic2 = base_model, basis_heusistic1, basis_heusistic2
                args.num_bin = 3
                analyze_double_difference_with_base_model(args, device, dataset, known_links, valid_pos_links, 
                                    valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)       
    '''

def evaluate_seperete_performance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # this evaluation is designed for each dataset
    args.num_bin = 5
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    
    preds, results, ranks = load_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, 
                            algorithms, models=models)
    # results in for overall performance
    
    heuristic_algorithms = ["CN", "homo", "global"]
    heuristic_preds_dict, _, _ = load_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, 
                            heuristic_algorithms, models=models)
    for heuristic_key in heuristic_preds_dict.keys():
        seperate_results_dict = defaultdict(dict)
        heuristic_preds = heuristic_preds_dict[heuristic_key]
        pos_heu_preds, neg_heu_preds = heuristic_preds[0], heuristic_preds[1]
        # print(heuristic_key)
        try:
            split_values, num_pos_values = equal_split(pos_heu_preds, args.num_bin)
        except:
            # heuristic_key = 'l2_0'
            if heuristic_key.find("l2") != -1 and args.dataset_name == 'ogbl-ppa':
                min_value = np.min(pos_heu_preds)
                split_values = [min_value,1] # , min_value + (1-min_value)/3
            else:
                import ipdb; ipdb.set_trace()
        _, num_neg_values = count_bin(neg_heu_preds, split_values)
        masks = generate_split_masks(pos_heu_preds, split_values)
        seperate_results = regional_evaluate(args, preds, masks)
        
        for model_name in seperate_results.keys():
            for group_id in range(len(seperate_results[model_name])):
                if group_id != len(seperate_results[model_name]) - 1:
                    group_range = f"{split_values[group_id]}-{split_values[group_id+1]}"
                else:
                    group_range = f"{split_values[group_id]}-inf"
                # import ipdb; ipdb.set_trace()
                seperate_results_dict[group_range][model_name] = seperate_results[model_name][group_id]

        records = pd.DataFrame(seperate_results_dict)
        records.to_excel(f"output_analyze/seperate_results/{args.dataset_name}_{heuristic_key}.xlsx")
        
    # import ipdb; ipdb.set_trace()
    # print()


def regional_evaluate(args, preds_dict, masks):
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    
    algo_names = preds_dict.keys()
    seperate_results_dict = {}
    
     
    for algo_idx, algo_name in enumerate(algo_names):
        preds = preds_dict[algo_name]
        pos_preds, neg_preds = preds[0], preds[1]
        seperate_result = seperate_accuracy(pos_preds, neg_preds, masks, args.is_old_neg, K)
        seperate_results_dict[algo_name] = seperate_result
    
    return seperate_results_dict
    
def analyze_pred_correlation(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    num_algo = 1
    
    preds_list, results_list, wrong_pos_indexes_list, correct_indexes_list = [], [], [], []
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    algorithms = ["homo", 'CN', 'global'] if args.dataset_name != "ddi" else ['CN', 'global']

    preds_dict, heu_ranks_dict = {}, {}    
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
        key = list(preds.keys())[0]
        preds = preds[key]
        preds_dict[algorithm] = preds
        # print(algorithm)
        if isinstance(preds[0], np.ndarray):
            preds[0] = torch.tensor(preds[0])
            preds[1] = torch.tensor(preds[1])
            
        preds = torch.cat([preds[0], preds[1]], dim=0)
        rank = get_rank_single(args, preds)
        heu_ranks_dict[algorithm] = rank

    model_preds, model_results, model_ranks_dict = best_model(args, args.dataset_name, models, prefix=None)
    plot_models_prediction_correlation(args, heu_ranks_dict, model_ranks_dict, args.dataset_name)
    # import ipdb; ipdb.set_trace()
    # print()
    


    
def analyze_pairwise_hard(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    num_algo = 1
    preds_list, results_list, wrong_pos_indexes_list, correct_indexes_list = [], [], [], []
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    algorithms = ["homo", 'CN', 'global'] if args.dataset_name != "ddi" else ['CN', 'global']
    
    preds_dict = {}    
    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
        elif algorithm == "global":
            preds, results = best_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
        elif algorithm == "model":
            assert models != None, "No model defined" 
            preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
        else:
            args.algorithm = "CN"
            preds, results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
        preds_dict[algorithm] = preds
    
    model_preds, model_results, model_ranks = best_model(args, args.dataset_name, models, prefix=None)

    model_correct_mask_dict = {}
    
    for model_name in model_preds.keys():
        model_pred = model_preds[model_name]
        pos_preds, neg_preds = model_pred[0], model_pred[1]
        num_pos = pos_preds.shape[0]
        # # print(num_pos)
        # import ipdb; ipdb.set_trace()
        _, _, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
        correct_mask = index_to_mask(correct_index, num_pos)
        model_correct_mask_dict[model_name] = correct_mask.cpu()
        
    # import ipdb; ipdb.set_trace()
    # algorithms1 = ["homo"]
    # algorithms2 = ["CN",  "global"]
    
    for idx1 in range(len(algorithms)):
        algorithm1 = algorithms[idx1]
        for idx2 in range(idx1+1, len(algorithms)):
        # for idx2 in range(len(algorithms2)):
            algorithm2 = algorithms[idx2]
            # import ipdb; ipdb.set_trace()
            if algorithm1 == algorithm2:
                continue
            print(f"{algorithm1} {algorithm2}")
            preds_dict1, preds_dict2 = preds_dict[algorithm1], preds_dict[algorithm2]
            algo_names1, algo_names2 = list(preds_dict1.keys()), list(preds_dict2.keys())
            num_algo1, num_algo2 = len(preds_dict1.keys()), len(preds_dict2.keys())
            # import ipdb; ipdb.set_trace()
            results_dict = defaultdict(dict)
            
            correct_wrong_results, wrong_correct_results = {}, {}
            for algo_idx1, algo_name1 in enumerate(preds_dict1.keys()):
                # print(algo_name1)
                for algo_idx2, algo_name2 in enumerate(preds_dict2.keys()):
                    preds1, preds2 = preds_dict1[algo_name1], preds_dict2[algo_name2]
                    pos_preds1, neg_preds1 = preds1[0], preds1[1]
                    if isinstance(pos_preds1, torch.Tensor):
                        if pos_preds1.device == torch.device("cuda"):
                            pos_preds1, neg_preds1 = pos_preds1.cpu(), neg_preds1.cpu()
                    else:
                        pos_preds1, neg_preds1 = torch.tensor(pos_preds1), torch.tensor(neg_preds1)                    
                    
                    pos_preds2, neg_preds2 = preds2[0], preds2[1]
                    if isinstance(pos_preds2, torch.Tensor):
                        if pos_preds2.device == "cuda":
                            pos_preds2, neg_preds2 = pos_preds2.cpu(), neg_preds2.cpu()
                    else:
                        pos_preds2, neg_preds2 = torch.tensor(pos_preds2), torch.tensor(neg_preds2)
                    # import ipdb; ipdb.set_trace()
                    num_pos = pos_preds1.shape[0]
                    
                    _, _, correct_index1 = get_rank_new(pos_preds1, neg_preds1, args.is_old_neg, K)
                    _, _, correct_index2 = get_rank_new(pos_preds2, neg_preds2, args.is_old_neg, K)
                    
                    correct_mask1, correct_mask2 = index_to_mask(correct_index1, num_pos), index_to_mask(correct_index2, num_pos)
                    correct_mask1, correct_mask2 = correct_mask1.cpu(), correct_mask2.cpu()
                    wrong_mask1, wrong_mask2  = ~correct_mask1, ~correct_mask2
                    
                    # print(f"{correct_mask1.device} {wrong_mask2.device} {wrong_mask1.device} {correct_mask2.device}")
                    # import ipdb; ipdb.set_trace()
                    correct_wrong_mask, wrong_correct_mask = correct_mask1 & wrong_mask2, wrong_mask1 & correct_mask2
                    '''
                    print(f"{correct_wrong_mask.sum().item()}   {wrong_correct_mask.sum().item()}")
                    # correct_wrong_index, wrong_correct_index = mask_to_index(correct_wrong_mask), mask_to_index(wrong_correct_mask)
                    name_dict = {"homo": "feat", "CN":"local", "global": "global"}
                    algorithm_tmp1, algorithm_tmp2 = name_dict[algorithm1], name_dict[algorithm2]
                    torch.save(correct_wrong_mask, f"intermedia_result/harry/{args.dataset_name}_{algorithm_tmp1}_{algorithm_tmp2}.pt")
                    torch.save(wrong_correct_mask, f"intermedia_result/harry/{args.dataset_name}_{algorithm_tmp2}_{algorithm_tmp1}.pt")
                    continue
                    '''
                    for model_name in model_preds.keys():
                        model_mask = model_correct_mask_dict[model_name]
                        num_correct_wrong, num_wrong_correct = correct_wrong_mask.sum().item(), wrong_correct_mask.sum().item()
                        model_correct_wrong_mask, model_wrong_correct_mask = correct_wrong_mask & model_mask, wrong_correct_mask & model_mask
                        
                        num_correct_wrong = 1 if num_correct_wrong == 0 else num_correct_wrong 
                        num_wrong_correct = 1 if num_wrong_correct == 0 else num_wrong_correct  
                        correct_wrong_acc, wrong_correct_acc = model_correct_wrong_mask.sum().item() / num_correct_wrong, model_wrong_correct_mask.sum().item() / num_wrong_correct
                        
                        correct_wrong_results[model_name] = correct_wrong_acc
                        wrong_correct_results[model_name] = wrong_correct_acc

                    # results_dict[algo_name1][algo_name2] = {"CW": correct_wrong_results, "WC": wrong_correct_results}    
                    results_dict[algorithm1][algorithm2] = {"CW": correct_wrong_results, "WC": wrong_correct_results}    
            # import ipdb; ipdb.set_trace()
            # plot_pairwise_hard(args, results_dict, algorithm1, algorithm2)
            plot_pairwise_hard_new(args, results_dict)
            print()    
                
# TODO: add experiments to compare on one group





def analyze_pairwise_hard_new(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    num_algo = 1
    
    preds_list, results_list, wrong_pos_indexes_list, correct_indexes_list = [], [], [], []
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    algorithms = ["homo", 'CN', 'global'] if args.dataset_name != "ddi" else ['CN', 'global']

    preds_dict = {}    
    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
            # import ipdb; ipdb.set_trace()
        elif algorithm == "global":
            preds, results = best_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path, num_algo=num_algo)
        elif algorithm == "model":
            assert models != None, "No model defined" 
            preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
            # print()
        else:
            args.algorithm = "CN"
            preds, results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
        preds_dict[algorithm] = preds
    
    model_preds, model_results, model_ranks = best_model(args, args.dataset_name, models, prefix=None)

    model_correct_mask_dict = {}
    
    for model_name in model_preds.keys():
        model_pred = model_preds[model_name]
        pos_preds, neg_preds = model_pred[0], model_pred[1]
        num_pos = pos_preds.shape[0]
        _, _, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
        correct_mask = index_to_mask(correct_index, num_pos)
        model_correct_mask_dict[model_name] = correct_mask.cpu()
        
    results_dict = defaultdict(dict)
    for idx1 in range(len(algorithms)):
        algorithm1 = algorithms[idx1]
        algorithm1 = "homo"
        algorithms2 = ["CN", "global"]
        # for idx2 in range(idx1+1, len(algorithms)):
            # algorithm2 = algorithms[idx2]
        for algorithm2 in algorithms2:
            if algorithm1 == algorithm2:
                continue
            print(f"{algorithm1} {algorithm2}")
            preds_dict1, preds_dict2 = preds_dict[algorithm1], preds_dict[algorithm2]
            algo_names1, algo_names2 = list(preds_dict1.keys()), list(preds_dict2.keys())
            num_algo1, num_algo2 = len(preds_dict1.keys()), len(preds_dict2.keys())
            # import ipdb; ipdb.set_trace()
            
            correct_wrong_results, wrong_correct_results = {}, {}
            for algo_idx1, algo_name1 in enumerate(preds_dict1.keys()):
                # print(algo_name1)
                for algo_idx2, algo_name2 in enumerate(preds_dict2.keys()):
                    preds1, preds2 = preds_dict1[algo_name1], preds_dict2[algo_name2]
                    pos_preds1, neg_preds1 = preds1[0], preds1[1]
                    if isinstance(pos_preds1, torch.Tensor):
                        if pos_preds1.device == torch.device("cuda"):
                            pos_preds1, neg_preds1 = pos_preds1.cpu(), neg_preds1.cpu()
                    else:
                        pos_preds1, neg_preds1 = torch.tensor(pos_preds1), torch.tensor(neg_preds1)                    
                    
                    pos_preds2, neg_preds2 = preds2[0], preds2[1]
                    if isinstance(pos_preds2, torch.Tensor):
                        if pos_preds2.device == "cuda":
                            pos_preds2, neg_preds2 = pos_preds2.cpu(), neg_preds2.cpu()
                    else:
                        pos_preds2, neg_preds2 = torch.tensor(pos_preds2), torch.tensor(neg_preds2)
                    # import ipdb; ipdb.set_trace()
                    num_pos = pos_preds1.shape[0]
                    
                    _, _, correct_index1 = get_rank_new(pos_preds1, neg_preds1, args.is_old_neg, K)
                    _, _, correct_index2 = get_rank_new(pos_preds2, neg_preds2, args.is_old_neg, K)
                    
                    correct_mask1, correct_mask2 = index_to_mask(correct_index1, num_pos), index_to_mask(correct_index2, num_pos)
                    correct_mask1, correct_mask2 = correct_mask1.cpu(), correct_mask2.cpu()
                    wrong_mask1, wrong_mask2  = ~correct_mask1, ~correct_mask2
                    
                    # print(f"{correct_mask1.device} {wrong_mask2.device} {wrong_mask1.device} {correct_mask2.device}")
                    # import ipdb; ipdb.set_trace()
                    correct_wrong_mask, wrong_correct_mask = correct_mask1 & wrong_mask2, wrong_mask1 & correct_mask2
                    # correct_wrong_index, wrong_correct_index = mask_to_index(correct_wrong_mask), mask_to_index(wrong_correct_mask)
                    import ipdb; ipdb.set_trace()
                    
                    for model_name in model_preds.keys():
                        model_mask = model_correct_mask_dict[model_name]
                        num_correct_wrong, num_wrong_correct = correct_wrong_mask.sum().item(), wrong_correct_mask.sum().item()
                        model_correct_wrong_mask, model_wrong_correct_mask = correct_wrong_mask & model_mask, wrong_correct_mask & model_mask
                        
                        num_correct_wrong = 1 if num_correct_wrong == 0 else num_correct_wrong 
                        num_wrong_correct = 1 if num_wrong_correct == 0 else num_wrong_correct  
                        correct_wrong_acc, wrong_correct_acc = model_correct_wrong_mask.sum().item() / num_correct_wrong, model_wrong_correct_mask.sum().item() / num_wrong_correct
                        
                        correct_wrong_results[model_name] = correct_wrong_acc
                        wrong_correct_results[model_name] = wrong_correct_acc

                    # results_dict[algo_name1][algo_name2] = {"CW": correct_wrong_results, "WC": wrong_correct_results}    
                    results_dict[algorithm1][algorithm2] = {"CW": correct_wrong_results, "WC": wrong_correct_results}    
    # import ipdb; ipdb.set_trace()
    plot_pairwise_hard_new_new(args, results_dict)
    print()    


                    
def analyze_triple_hard(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    assert args.dataset_name not in ["ogbl-ddi", "ddi"], "ogbl-ddi does not have homophily as feature"
    num_algo = 1
    
    preds_list, results_list, wrong_pos_indexes_list, correct_indexes_list = [], [], [], []
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    algorithms = ["homo", 'CN', 'global'] if args.dataset_name != "ddi" else ['CN', 'global']

    preds_dict = {}    
    for algorithm in algorithms:
        if algorithm == "homo":
            preds, results = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
            # import ipdb; ipdb.set_trace()
        elif algorithm == "global":
            preds, results = best_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path, num_algo=num_algo)
        elif algorithm == "model":
            assert models != None, "No model defined" 
            preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
            # print()
        else:
            args.algorithm = "CN"
            # preds, results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
            preds, results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path,num_algo=num_algo)
        preds_dict[algorithm] = preds
    
    model_preds, model_results, model_ranks = best_model(args, args.dataset_name, models, prefix=None)

    model_correct_mask_dict = {}
    
    for model_name in model_preds.keys():
        model_pred = model_preds[model_name]
        pos_preds, neg_preds = model_pred[0], model_pred[1]
        num_pos = pos_preds.shape[0]
        _, _, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
        correct_mask = index_to_mask(correct_index, num_pos)
        model_correct_mask_dict[model_name] = correct_mask.cpu()
    
    algo_type_names = list(preds_dict.keys())
    
    heu_types = ["CN", "global", "homo"]
    
    for heu_idx, heu_type in enumerate(heu_types):
        results_dict = defaultdict(dict)
        correct_heu = heu_type
        wrong_heus = [heu_type for heu_type in heu_types if heu_type != correct_heu]
        # import ipdb; ipdb.set_trace()
        correct_preds_dict = preds_dict[correct_heu]
        
        for correct_algo_idx, correct_algo_key in enumerate(correct_preds_dict.keys()):
            correct_preds = correct_preds_dict[correct_algo_key]
            pos_correct_preds, neg_correct_preds = correct_preds[0], correct_preds[1]
            _, _, correct_index = get_rank_new(pos_correct_preds, neg_correct_preds, args.is_old_neg, K)
            correct_mask = index_to_mask(correct_index, num_pos).cpu()
            num_pos = pos_correct_preds.shape[0]
            
            wrong_preds_dict1, wrong_preds_dict2 = preds_dict[wrong_heus[0]], preds_dict[wrong_heus[1]]
            
            for wrong_algo_idx1, wrong_algo_key1 in enumerate(wrong_preds_dict1.keys()):
                wrong_preds1 = wrong_preds_dict1[wrong_algo_key1]
                pos_wrong_preds1, neg_wrong_preds1 = wrong_preds1[0], wrong_preds1[1]
                
                _, _, correct_index1 = get_rank_new(pos_wrong_preds1, neg_wrong_preds1, args.is_old_neg, K)
                correct_mask1 = index_to_mask(correct_index1, num_pos).cpu()
                wrong_mask1 = ~correct_mask1.cpu()
                
                for wrong_algo_idx2, wrong_algo_key2 in enumerate(wrong_preds_dict2.keys()):
                    wrong_algo_key = f"{wrong_algo_key1} {wrong_algo_key2}"
                    wrong_preds2 = wrong_preds_dict2[wrong_algo_key2]
                    pos_wrong_preds2, neg_wrong_preds2 = wrong_preds2[0], wrong_preds2[1]
                
                    _, _, correct_index2 = get_rank_new(pos_wrong_preds2, neg_wrong_preds2, args.is_old_neg, K)
                    correct_mask2 = index_to_mask(correct_index2, num_pos).cpu()
                    wrong_mask2 = ~correct_mask2.cpu()

                    wrong_mask = wrong_mask1 & wrong_mask2
                    
                    correct_wrong_mask = correct_mask & wrong_mask
                    wrong_correct_mask = ~correct_wrong_mask & ~wrong_mask 
                    num_correct_wrong, num_wrong_correct = correct_wrong_mask.sum().item(), wrong_correct_mask.sum().item()
                                    
                    correct_wrong_results, wrong_correct_results = {}, {}
                    for model_name in model_preds.keys():
                        model_mask = model_correct_mask_dict[model_name]
                        model_correct_wrong_mask, model_wrong_correct_mask = correct_wrong_mask & model_mask, wrong_correct_mask & model_mask
                        num_wrong_correct = 1 if num_wrong_correct == 0 else num_wrong_correct
                        num_correct_wrong = 1 if num_correct_wrong == 0 else num_correct_wrong
                        correct_wrong_acc, wrong_correct_acc = model_correct_wrong_mask.sum().item() / num_correct_wrong, model_wrong_correct_mask.sum().item() / num_wrong_correct
                        
                        correct_wrong_results[model_name] = correct_wrong_acc
                        wrong_correct_results[model_name] = wrong_correct_acc
                        
                    results_dict[correct_algo_key][wrong_algo_key] = {"CW": correct_wrong_results, "WC": wrong_correct_results}    
            
        plot_triple_hard(args, results_dict, correct_heu, wrong_heus)
                    


def find_majority(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # For the majority, I would only find one algorith, for CN, we only use the CN zero
    num_algo = 2
    preds_list, results_list, wrong_pos_indexes_list, correct_indexes_list = [], [], [], []
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    algorithms = ["homo", 'CN', 'global'] if args.dataset_name != "ddi" else ['CN', 'global']

    preds_dict = {}    
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
        preds_dict[algorithm] = preds
    
    discrete_names = [f"CN_{i}" for i in range(6)]
    interval_dict = {}
    for algo_type in preds_dict.keys():
        pred_dict = preds_dict[algo_type]
        for algo_name in pred_dict.keys():
            pred = pred_dict[algo_name]
            pos_preds, neg_preds = pred[0], pred[1]
            num_pos = pos_preds.shape[0]
            # if algo_name in discrete_names:
            interval = find_majority_group(pos_preds)
            interval_dict[algo_name] = interval
    
    # for the interval, both left and right are closed
    with open(f"intermedia_result/major_interval/{args.dataset_name}.txt", "wb") as f:
        pickle.dump(interval_dict, f)
    



def analyze_difference_with_base_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # the base model could be GCN, MLP, heuristic method, other algorithms will compare the gap between the base model
    
    # similar with analysis in the last piece
    # base_model = "empty" # 'gcn' 'homo' 'local' 'global's
    # basis_heuristic = "homo" # 'local' 'global'
    base_model = args.base_model # 'gcn' 'homo' 'local' 'global's
    basis_heuristic = args.basis_heuristic # 2'local' 'global'
    
    algorithms = ["model"]
    model_group_dict = {'mlp':'model', "cn": "local", "CN": "local", 'gcn': "model"}
    base_model_group = model_group_dict[base_model] if base_model in model_group_dict.keys() else base_model
    basis_heuristic_group = model_group_dict[basis_heuristic] if basis_heuristic in model_group_dict.keys() else basis_heuristic
    # , basis_heuristic_group = , model_group_dict[basis_heuristic]
    # tradic_preds, tradic_results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links,test_pos_links, test_neg_links, path)
    # import ipdb; ipdb.set_trace()
    # algorithms = ["homo", "CN"]
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]

    preds, results, ranks = load_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, 
                                algorithms, models=models)
    if base_model_group != "empty":
        base_preds, base_results, base_ranks = load_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, 
                                    [base_model_group], models=[base_model], num_algo=1)
    
    basis_heuristic_preds, basis_heuristic_results, basis_heuristic_ranks = load_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, 
                                [basis_heuristic_group], models=[basis_heuristic], num_algo=1)
    
    # import ipdb; ipdb.set_trace()
    heuristic_key = list(basis_heuristic_preds.keys())[0]
    try:
        basic_key = list(base_preds.keys())[0] if base_model_group != "empty" else "empty" 
    except:
        import ipdb; ipdb.set_trace()
    pos_basis_heuristic_preds, neg_basis_heuristic_preds = basis_heuristic_preds[heuristic_key][0], basis_heuristic_preds[heuristic_key][1]
    try:
        split_values, num_pos_values = equal_split(pos_basis_heuristic_preds, args.num_bin)
    except:
        if basis_heuristic == "homo" and args.dataset_name == 'ogbl-ppa':
            min_value = np.min(pos_basis_heuristic_preds)
            split_values = [min_value,1] # , min_value + (1-min_value)/3
            # in this case, there is only two values. 
            # min_value + (1-min_value)/2
        else:
            import ipdb; ipdb.set_trace()
    
    _, num_neg_values = count_bin(neg_basis_heuristic_preds, split_values)
    masks = generate_split_masks(pos_basis_heuristic_preds, split_values)
    # if basis_heuristic == "homo" and args.dataset_name == 'ogbl-ppa':
    #     masks = [masks[0], masks[2]]
    # import ipdb; ipdb.set_trace()
    if base_model_group != "empty":
        base_seperate_result = seperate_accuracy(base_preds[basic_key][0], base_preds[basic_key][1], masks, args.is_old_neg, K)
        base_seperate_result = np.array(base_seperate_result)
    else:
        args.num_bin = len(masks)
        base_seperate_result = np.zeros([args.num_bin])
    model_seperate_results = {}
    for key in preds.keys():
        pos_preds, neg_preds = preds[key][0], preds[key][1]
        model_seperate_result = seperate_accuracy(pos_preds, neg_preds, masks, args.is_old_neg, K)
        model_seperate_results[key] = np.array(model_seperate_result)
        # import ipdb; ipdb.set_trace()
        # print()
    # import ipdb; ipdb.set_trace()
    plot_difference_with_base_model(preds, base_seperate_result, model_seperate_results, split_values, base_model, basis_heuristic, 
                               result_key, args.dataset_name, args, algorithms)
    
    

def analyze_double_difference_with_base_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # the base model could be GCN, MLP, heuristic method, other algorithms will compare the gap between the base model
    args.num_bin = 3
    # similar with analysis in the last piece
    # base_model = "homo" # 'gcn' 'homo' 'local' 'global 'empty'
    # basis_heuristic1 = "homo"
    # basis_heuristic2 = "local"# 'local' 'global'
    base_model = args.base_model # 'gcn' 'homo' 'local' 'global 'empty'
    basis_heuristic1 = args.basis_heuristic1
    basis_heuristic2 = args.basis_heuristic2   # 'local' 'global'
    
    algorithms = ["model"]

    model_group_dict = {'mlp':'model', "cn": "local", "CN": "local", 'gcn': "model"}
    base_model_group = model_group_dict[base_model] if base_model in model_group_dict.keys() else base_model
    basis_heuristic_group1 = model_group_dict[basis_heuristic1] if basis_heuristic1 in model_group_dict.keys() else basis_heuristic1
    basis_heuristic_group2 = model_group_dict[basis_heuristic2] if basis_heuristic2 in model_group_dict.keys() else basis_heuristic2
    # , basis_heuristic_group = , model_group_dict[basis_heuristic]
    # tradic_preds, tradic_results = best_tradic(args, device, dataset, known_links, valid_pos_links, valid_neg_links,test_pos_links, test_neg_links, path)
    # import ipdb; ipdb.set_trace()
    # algorithms = ["homo", "CN"]
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]

    preds, results, ranks = load_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, 
                                algorithms, models=models)
    if base_model_group != "empty":
        base_preds, base_results, base_ranks = load_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, 
                                    [base_model_group], models=[base_model], num_algo=1)
    
    basis_heuristic_preds1, basis_heuristic_results, basis_heuristic_ranks = load_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, 
                                [basis_heuristic_group1], models=[basis_heuristic1], num_algo=1)
    basis_heuristic_preds2, basis_heuristic_results, basis_heuristic_ranks = load_model(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, 
                                [basis_heuristic_group2], models=[basis_heuristic2], num_algo=1)
    
    # import ipdb; ipdb.set_trace()
    heuristic_key1 = list(basis_heuristic_preds1.keys())[0]
    heuristic_key2 = list(basis_heuristic_preds2.keys())[0]
    base_key = list(base_preds.keys())[0] if base_model_group != "empty" else "empty" 
    
    pos_basis_heuristic_preds1, neg_basis_heuristic_preds1 = basis_heuristic_preds1[heuristic_key1][0], basis_heuristic_preds1[heuristic_key1][1]
    try:
       split_values1, num_pos_values1 = equal_split(pos_basis_heuristic_preds1, args.num_bin)
    except:
        if basis_heuristic1 == "homo" and args.dataset_name == 'ogbl-ppa':
            min_value = np.min(pos_basis_heuristic_preds1)
            split_values1 = [min_value,1] # , min_value + (1-min_value)/3
            # min_value + (1-min_value)/2
        else:
            import ipdb; ipdb.set_trace()
    _, num_neg_values1 = count_bin(neg_basis_heuristic_preds1, split_values1)
    masks1 = generate_split_masks(pos_basis_heuristic_preds1, split_values1)
    

    pos_basis_heuristic_preds2, neg_basis_heuristic_preds2 = basis_heuristic_preds2[heuristic_key2][0], basis_heuristic_preds2[heuristic_key2][1]    
    try:
        split_values2, num_pos_values2 = equal_split(pos_basis_heuristic_preds2, args.num_bin)
    except:
        if basis_heuristic2 == "homo" and args.dataset_name == 'ogbl-ppa':
            min_value = np.min(pos_basis_heuristic_preds2)
            split_values2 = [min_value,1] # , min_value + (1-min_value)/3
            # min_value + (1-min_value)/2
        else:
            import ipdb; ipdb.set_trace()
        
    _, num_neg_values2 = count_bin(neg_basis_heuristic_preds2, split_values2)
    masks2 = generate_split_masks(pos_basis_heuristic_preds2, split_values2)
    
    '''
    if base_model_group != "empty":
        base_seperate_result = seperate_accuracy(base_preds[basic_key][0], base_preds[basic_key][1], masks2, args.is_old_neg, K)
        # import ipdb; ipdb.set_trace()
        base_seperate_result = np.array(base_seperate_result)
    else:
        base_seperate_result = np.zeros([args.num_bin])
    '''
    
    model_seperate_results, base_seperate_results = defaultdict(list), defaultdict(list)
    # lack of value calculation
    results = []
    for idx1, mask1 in enumerate(masks1):
        num_edge = mask1.shape[0]
        pos_basis_heuristic_preds1_masked = pos_basis_heuristic_preds1[mask1]
        pos_basis_heuristic_preds2_masked = pos_basis_heuristic_preds2[mask1]
        masks2 = generate_split_masks(pos_basis_heuristic_preds2_masked, split_values2)
        
        # masks2_tmp = []
        for idx2, mask2 in enumerate(masks2):
            index1 = np.where(mask1)[0]
            index2 = index1[mask2]
            mask2 = np.zeros([num_edge], dtype=bool)
            mask2[index2] = True
            masks2[idx2] = mask2
                        
        for key in preds.keys():
            pos_preds, neg_preds = preds[key][0], preds[key][1]
            model_seperate_result = seperate_accuracy(pos_preds, neg_preds, masks2, args.is_old_neg, K)
            model_seperate_results[key].append(np.array(model_seperate_result))
        
        if base_model_group != "empty":
            base_seperate_result = seperate_accuracy(base_preds[base_key][0], base_preds[base_key][1], masks2, args.is_old_neg, K)
            base_seperate_result = np.array(base_seperate_result)
        else:
            args.num_bin = len(masks2)
            base_seperate_result = np.zeros([args.num_bin])
        base_seperate_results[base_key].append(base_seperate_result)
    # import ipdb; ipdb.set_trace()
            
    plot_double_difference_with_base_model(preds, base_seperate_results, model_seperate_results, split_values1, split_values2, base_model, 
                                           basis_heuristic1, basis_heuristic2, result_key, args.dataset_name, args, algorithms)
    # import ipdb; ipdb.set_trace()
    print()
    print()






def find_train_majority(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    num_nodes = dataset.data.x.shape[0]
    num_algo = 1
    # if args.is_load:
    #     with open(f"intermedia_result/train_CN/{args.dataset_name}.txt", "rb") as f:
    #         pickle.dump(dist_results, f)
    # else:
    #     dist_results = target_edge_removal(dataset, path, num_nodes, links=known_links)
    #     with open(f"intermedia_result/train_CN/{args.dataset_name}.txt", "wb") as f:
    #         pickle.dump(dist_results, f)
    
    
    preds_list, results_list, wrong_pos_indexes_list, correct_indexes_list = [], [], [], []
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    models_list = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # notably, homo, global and model are three important metric
    algorithms = ["homo", 'CN', 'global'] if args.dataset_name != "ddi" else ['CN', 'global']

    preds_dict = {}    
    # find the suitable algorithm name on test, then find train
    if not args.is_load:
        for algorithm in algorithms:
            if algorithm == "homo":
                if args.dataset_name == "ogbl-ddi":
                    continue
                preds, results = best_homo(args, device, dataset, known_links, test_pos_links, test_neg_links, path, K=num_algo)
                # import ipdb; ipdb.set_trace()
            elif algorithm == "global":
                preds, results = best_global(args, device, dataset, known_links, test_pos_links, test_neg_links, path)
            elif algorithm == "model":
                assert models != None, "No model defined" 
                preds, results, ranks = best_model(args, args.dataset_name, models, prefix=None)
                # print()
            else:
                continue
            preds_dict[algorithm] = preds
        
        # import ipdb; ipdb.set_trace()
        num_pos_edges = known_links.shape[0]
        if num_pos_edges > 10000000:
            known_links_test = known_links_test[:10000000]
            is_reach_max = True
        else: 
            known_links_test = known_links
            is_reach_max = False
            
        algo_names = []
        preds_list = []
        for algo_type in preds_dict.keys():
            preds = preds_dict[algo_type]
            algo_name = list(preds.keys())[0]
            if algo_type == "homo":
                algo_names.append(f"{algo_type}_{algo_name}")
                datas = algo_name.split("_")
                num_hop = int(datas[-1])
                args.is_feature = 1
                if num_hop == 0:
                    args.dis_func_name = datas[0]
                    args.num_hops = num_hop
                    args.adj_norm = "A"
                else:
                    args.dis_func_name = datas[0]
                    args.adj_norm = datas[1]
                    args.num_hops = num_hop
                pos_preds, neg_preds, result = run_single_homo(args, device, dataset, known_links, known_links_test.T, test_neg_links, path)       
                preds = pos_preds
                preds = preds.cpu().numpy()
                preds_list.append(preds)
            else:
                # import ipdb; ipdb.set_trace()
                if algo_name == "PPR":
                    preds = PPR_correct(known_links, known_links_test.T, is_old_neg=args.is_old_neg)
                else:
                    preds = SimRank_correct(known_links, known_links_test.T, args.dataset_name, is_old_neg=args.is_old_neg) 
                algo_names.append(f"{algo_type}_{algo_name}")
                preds = preds.numpy()
                preds_list.append(preds)
        args.algorithm = "CN"
        algo_names.append("CN_0")
        # import ipdb; ipdb.set_trace()
        pos_preds_list, neg_preds_list = tradic_count_logits(known_links, dataset, path, args, known_links_test, test_neg_links, is_test=1)
        pos_preds_list = pos_preds_list.astype(np.int32)
        preds_list.append(pos_preds_list[:, 0])
    
        # import ipdb; ipdb.set_trace()
        
        with open(f"intermedia_result/train_preds/{args.dataset_name}_preds.txt", "wb") as f:
            pickle.dump(preds_list, f)

        with open(f"intermedia_result/train_preds/{args.dataset_name}_names.txt", "wb") as f:
            pickle.dump(algo_names, f)

    else:
        with open(f"intermedia_result/train_preds/{args.dataset_name}_preds.txt", "rb") as f:
            preds_list = pickle.load(f)

        with open(f"intermedia_result/train_preds/{args.dataset_name}_names.txt", "rb") as f:
            algo_names = pickle.load(f)

        print()
        
    
    
    # discrete_names = [f"CN_{i}" for i in range(6)]
    interval_dict = {}
    for algo_name, preds in zip(algo_names, preds_list):
        interval = find_majority_group(preds)
        interval_dict[algo_name] = interval
    
    # for algo_type in preds_dict.keys():
    #     pred_dict = preds_dict[algo_type]
    #     for algo_name in pred_dict.keys():
    #         pred = pred_dict[algo_name]
    #         pos_preds, neg_preds = pred[0], pred[1]
    #         num_pos = pos_preds.shape[0]
            # if algo_name in discrete_names:
            
    # import ipdb; ipdb.set_trace()
    # for the interval, both left and right are closed
    with open(f"intermedia_result/major_interval/{args.dataset_name}_train.txt", "wb") as f:
        pickle.dump(interval_dict, f)
    
    print()




def find_majority_group(preds, ratio=0.5):
    # the smallest range for building ssuchratio
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    
    num_pos = preds.shape[0]
    # print(preds.type)
    # import ipdb; ipdb.set_trace()
    num_candidiate = int(num_pos * ratio)
    preds = np.sort(preds)
    
    min_range, min_idx = 1000000, 0
    for i in range(num_pos - num_candidiate):
        data_range = np.abs(preds[i+num_candidiate] - preds[i])
        if data_range < min_range:
            min_range = data_range
            min_idx = i
    
    return [preds[min_idx], preds[min_idx+num_candidiate]]
        