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
from data_analysis.function.heuristics import PPR_new, PPR_correct, SimRank_correct
from torch_geometric.utils import mask_to_index, index_to_mask
import torch_geometric
from data_analysis.homo_tradic_plugin import best_homo, best_model, best_tradic, best_global, load_model, get_results, run_single_homo



'''
ogbl-collab: we usually include the validation edges in the training graph when doing testing.
ogbl-ddi: It doesn't have node features. There is a weak relationship between the validation and test performance.
ogbl-ppa: The node feature is the 58 dimensional one-hot vector. The MLP has very bad performance.
ogbl-citation2:  In the validation/test,  positive samples have specific negative samples.
'''


def run_model_analyze_new(args, device):
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
   
    args.is_load = 0
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
    models = ["mlp", "gcn", "sage", "seal", "buddy", "neognn"]  # , "neognn"
    # models = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    # models = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    
    # algorithms = [["homo", "global"], ["global", "CN"], ["homo", "CN"]]
    # algorithms = [["CN", "homo"]]
    # algorithms = ["homo", "global", "CN"]
    algorithms = ["model"]
    # evaluate_seperete_performance(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)
    # analyze_pairwise_hard(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)  
    # analyze_triple_hard(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)
    # find_majority(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)
    analyze_majority(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=models)








def analyze_majority(args, device, dataset, known_links, valid_pos_links, valid_neg_links, test_pos_links, test_neg_links, path, algorithms, models=None):
    # TODO: current homo hete have proble
    assert args.dataset_name not in ["ogbl-ddi", "ddi"], "ogbl-ddi does not have homophily as feature"
    num_algo = 1
    base_model = "empty"
    
    
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
            continue
        preds_dict[algorithm] = preds
        
    args.algorithm = "CN"
    pos_preds_list, neg_preds_list = tradic_count_logits(known_links, dataset, path, args, known_links, test_neg_links, is_test=1)
    preds_dict['CN'] = {"CN_0":[pos_preds_list[:, 0], neg_preds_list[:, 0]]}
    
    model_preds, model_results, model_ranks = best_model(args, args.dataset_name, models, prefix=None)
    
    with open(f"intermedia_result/major_interval/{args.dataset_name}_train.txt", "rb") as f:
        interval_dict = pickle.load(f)

    global_algos = ["PPR", "SimRank"]
    
    new_interval_dict = {}
    for algo_key in interval_dict.keys():
        tmp_datas = algo_key.split("_")
        if algo_key in global_algos:
            new_interval_dict["global"] = interval_dict[algo_key]
        elif tmp_datas[0] == "homo":
            new_interval_dict["homo"] = interval_dict[algo_key]
        elif tmp_datas[0] == "CN_0":
            new_interval_dict["CN"] = interval_dict[algo_key]
    
    major_mask_dict = {}
    
    if base_model != "empty":
        base_preds = model_preds[base_model]
        del model_preds[base_model]
    
    for algo_type in preds_dict.keys():
        interval = new_interval_dict[algo_type]
        preds = preds_dict[algo_type]
        algo_name = list(preds.keys())[0]
        preds = preds[algo_name]
        pos_preds, neg_preds = preds[0], preds[1]
        if isinstance(pos_preds, np.ndarray):
            pos_preds, neg_preds = torch.tensor(pos_preds), torch.tensor(neg_preds)
        pos_preds, neg_preds = pos_preds.cpu(), neg_preds.cpu()
        num_pos = pos_preds.shape[0]
        major_mask = (pos_preds >= interval[0]) * (pos_preds <= interval[1])  
        minor_mask = ~major_mask

        masks = [major_mask, minor_mask]
        args.num_bin = len(masks)
        names = ["major", "minor"]
            
        major_mask_dict[algo_type] = major_mask

        if base_model != "empty":
            base_seperate_result = seperate_accuracy(base_preds[0], base_preds[1], masks, args.is_old_neg, K)
            base_seperate_result = np.array(base_seperate_result)
        else:
            base_seperate_result = np.zeros([args.num_bin])
        
        model_seperate_results = {}
        for key in model_preds.keys():
            pos_preds, neg_preds = model_preds[key][0], model_preds[key][1]
            model_seperate_result = seperate_accuracy(pos_preds, neg_preds, masks, args.is_old_neg, K)
            model_seperate_result -= base_seperate_result
            model_seperate_results[key] = np.array(model_seperate_result)
            
        
        import ipdb; ipdb.set_trace()
        plot_major_minor_compare(args, base_model, names, model_seperate_results, result_key, algo_type)
        
        
        
        print()    

        
    # only select one algorithm from one perspecitive
    


    