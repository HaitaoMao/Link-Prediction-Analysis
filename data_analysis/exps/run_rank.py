from data_analysis.function.read_results import *
from data_analysis.function.rank_analysis import *
import numpy as np
import torch

def run_rank(args, device):
    origin_model_names = ["tradic", "gcn"]
    dataset_names = ["Cora"]

    is_std = 0

    args.is_old_neg = 1  # whether use the new heart negative sampling method
    args.analyze_mode = "test"  # "whole" "valid" "test"
    args.is_flatten = 1  # if use the heart, whether remove the redudant validation and test edge 
    args.is_remove_redudant = 1  # if use the heart, whether remove the redudant validation and test edge 

    datasets = ["Citeseer", "Cora", "Pubmed", "ogbl-citation2",  "ogbl-collab", "ogbl-ddi", "ogbl-ppa"]
    models = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    
    heuristic_models = ["tradic", "path", "motif", "homo", "AA", "RA", "PPR", "SimRank"]
    multiple_models = ['tradic', "path", "motif", "homo"]
    # results have multiple results

    remain_models = generate_remains()

    for dataset_name in dataset_names:
        args.dataset_name = dataset_name
        pos_preds_list, neg_preds_list, results_list, model_names, ranks_list = [], [], [], [], []
        for model_name in origin_model_names:
            if model_name in multiple_models:
                new_pos_preds_list, new_neg_preds_list, new_results, new_model_names = load_heuristic_results(model_name, args, remain_models)
                pos_preds_list.extend(new_pos_preds_list)
                neg_preds_list.extend(new_neg_preds_list)
                results_list.extend(new_results)
                model_names.extend(new_model_names)
                ranks_list.extend(generate_ranks(pos_preds_list, neg_preds_list))
            else:
                pos_preds, neg_preds, rank, results = load_results_with_multiseed(args, dataset_name, model_name, is_std, prefix=None)
                pos_preds_list.append(pos_preds)
                neg_preds_list.append(neg_preds)
                results_list.append(results)
                model_names.append(model_name)
                ranks_list.append(rank)
        # import ipdb; ipdb.set_trace()
        # print()

        for idx1, (ranks1, result1, model_name1) in enumerate(zip(ranks_list, results, model_names)):
            for idx2, (ranks2, result2, model_name2) in enumerate(zip(ranks_list, results, model_names)):
                if idx1 == idx2:
                    continue
                corr_coef, p_value = run_correlation_simple(rank1, rank2, model_name1, model_name2)
                xname = f"{model_name1}_{model_name2}"
                # the name in the x axis
                name = f"{dataset_name}_old" if args.is_old_neg else f"{dataset_name}" 
                plot_models_prediction_correlation(args, ranks1, ranks2, model_name1, model_name2, dataset_name, corr_coef, p_value) 



    # data_name = "Cora"
    # model_name = "gcn"
    
    # load_heuristic_results("tradic", args)

    # for model_name in model_names:
    # pos_preds, neg_preds, results = load_results(args, data_name, model_name)


    
    

    
