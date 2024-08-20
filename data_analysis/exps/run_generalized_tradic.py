from data_analysis.plot_exp import *
from data_analysis.function.functional import *
from data_analysis.generate_data import load_data
from data_analysis.function.heuristics import CN, CN_new
from evaluation_new import * 
import scipy.sparse as sp
import torch
import os
import pathlib

def run_heursitic(args, device):
    args.is_generate_train = 1   # whether preprocess on just training set or entire dataset
    args.is_old_neg = 1            # whether use the new heart negative sampling method
    args.analyze_mode = "test"  # "whole" "valid" "test"
    args.is_flatten = 0  # if use the heart, whether remove the redudant validation and test edge 
    args.is_remove_redudant = 1  # if use the heart, whether remove the redudant validation and test edge 
    args.batch_size = 1000
    args.ratio_per_hop = 1.0
    
    dataset, known_links, eval_pos_links, eval_neg_links, path = load_data(args, device)

    args.is_load = 0
    # "katz"
    algorithms = ["CN", "Jaccard", "RA", "AA", "PA", "ppr"]
    for algorithm in algorithms:
        args.algorithm = algorithm
        run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)


def run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path):
    print(f"known links shape {known_links.shape}")
    num_nodes = torch.max(known_links).item() + 1
    select_xlim = 0
    select_lims_dict = {"Cora": [0, 1], "CiteSeer": [0, 1], "PubMed": [0, 1], 
        "ogbl-citation2": [0, 100], "ogbl-ppa": [0, 100], "ogbl-ddi": [0, 1000], "ogbl-collab": [0, 100]}
    if select_xlim:
        xlim = select_lims_dict[args.dataset_name]
    else:
        xlim=-1
    # if xlim = -1, use the default setting 
    # tradic closure 

    data_path = f"output_analyze/results/{args.dataset_name}"
    folder_path = pathlib.Path(data_path) 
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    # pos_tradic_results, neg_tradic_results = pos_tradic_results.cpu().numpy(), neg_tradic_results.cpu().numpy()

    if not args.is_load:
        pos_tradic_results, neg_tradic_results = tradic_algorithm(known_links, dataset, path, args, eval_pos_links, eval_neg_links)
        with open(f"{data_path}/{args.algorithm}_{args.is_old_neg}.txt", "wb") as f:
            pickle.dump({"pos": pos_tradic_results, "neg": neg_tradic_results}, f)
    else:
        with open(f"{data_path}/{args.algorithm}_{args.is_old_neg}.txt", "rb") as f:
            data_dict = pickle.load(f)
        pos_tradic_results = data_dict["pos"]
        neg_tradic_results = data_dict["neg"]   
    
    num_hops = pos_tradic_results.shape[0] if args.algorithm not in ['katz', 'ppr'] else 1
    
    for num_hop1 in range(num_hops):
        for num_hop2 in range(num_hops):
            pos_tradic_result, neg_tradic_result = pos_tradic_results[num_hop1, num_hop2], neg_tradic_results[num_hop1, num_hop2]
            # import ipdb; ipdb.set_trace()
            # pos_tradic_result = torch.tensor(pos_tradic_result)
            # data = pos_tradic_result != pos_preds
            xname = f"{args.algorithm}_{num_hop1+1}_{num_hop2+1}"
            # the name in the x axis
            name = f"{args.dataset_name}_old" if args.is_old_neg else f"{args.dataset_name}" 
            plot_property(pos_tradic_result, neg_tradic_result, name=name, xname=xname, xlim=xlim)
    
    results_dict = {}
    for num_hop1 in range(num_hops):
        for num_hop2 in range(num_hops):
            pos_preds, neg_preds = pos_tradic_results[num_hop1, num_hop2], neg_tradic_results[num_hop1, num_hop2]
            pos_preds, neg_preds = torch.tensor(pos_preds), torch.tensor(neg_preds)
            if args.is_flatten == 0 and args.is_old_neg == 0:
                num_edges = args.num_pos_test_edge if analyze_mode == "valid" else args.num_pos_val_edge
                neg_preds = torch.reshape(neg_preds, [num_edges, -1])
                # pos_preds = torch.unsqueeze(pos_preds, dim=1) 
                # new_preds = torch.cat([pos_preds, neg_preds], dim=1)
                results = get_metric_score(pos_preds, neg_preds)
            else:
                pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
                results = get_metric_score_origin(pos_preds, neg_preds)
            result_strings = [f"{args.dataset_name}_{args.algorithm}_{num_hop1}_{num_hop2}"]
            for key in results:
                result_strings.append(f"{key}:{results[key]}")
            sep = " "
            result_string = sep.join(result_strings) +"\n"
            with open(f"{data_path}/ALL_{args.algorithm}_result_{args.is_old_neg}.txt", "w") as f:
                f.write(result_string)

            results_dict[f"{num_hop1}_{num_hop2}"] = results
    
    
    with open(f"{data_path}/{args.algorithm}_result_{args.is_old_neg}.txt", "wb") as f:
        pickle.dump(results_dict, f)

    