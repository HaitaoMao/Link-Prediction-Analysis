from data_analysis.function.motif_count import *
from data_analysis.function.functional import *
from data_analysis.plot_exp import *
from data_analysis.generate_data import load_data
import scipy.sparse as sp
import torch
import os
import pathlib
from evaluation_new import * 

def run_motif(args, device):
    args.is_generate_train = 1   # whether preprocess on just training set or entire dataset
    args.is_old_neg = 1            # whether use the new heart negative sampling method
    args.analyze_mode = "test"  # "whole" "valid" "test"
    args.is_flatten = 1  # if use the heart, whether remove the redudant validation and test edge 
    args.is_remove_redudant = 1  # if use the heart, whether remove the redudant validation and test edge 

    args.batch_size = 1000
    args.is_load = 1
    dataset, known_links, eval_pos_links, eval_neg_links, path = load_data(args, device)
    num_nodes = torch.max(dataset.data.edge_index).item() + 1
    data_path = f"output_analyze/results/{args.dataset_name}"
    folder_path = pathlib.Path(data_path) 
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    
    if not args.is_load:
        pos_results, descriptions = motif_count(args, known_links, eval_pos_links)
        neg_results, descriptions = motif_count(args, known_links, eval_pos_links)
        with open(f"{data_path}/motif_{args.is_old_neg}.txt", "wb") as f:
            pickle.dump({"pos": pos_results, "neg": neg_results, "description": descriptions}, f)   
        pos_results, neg_results = pos_results.cpu().numpy(), neg_results.cpu().numpy()
    else:
        with open(f"{data_path}/motif_{args.is_old_neg}.txt", "rb") as f:
            data_dict = pickle.load(f)
        pos_results = data_dict["pos"]
        neg_results = data_dict["neg"]   
        pos_results, neg_results = pos_results.cpu().numpy(), neg_results.cpu().numpy()

        


    # import ipdb; ipdb.set_trace()
    for idx in range(len(descriptions)):
        pos_result, neg_result, description = pos_results[:, idx], neg_results[:, idx], descriptions[idx]
        print(description)
        # import ipdb; ipdb.set_trace()
        xname = f"motif_{idx}"
        # the name in the x axis
        name = f"{args.dataset_name}_old" if args.is_old_neg else f"{args.dataset_name}" 
        plot_property(pos_result, neg_result, name=name, xname=xname)

    results_dict = {}
    for idx in range(len(descriptions)):
        pos_result, neg_result, description = pos_results[:, idx], neg_results[:, idx], descriptions[idx]
        pos_preds, neg_preds = torch.tensor(pos_result), torch.tensor(neg_result)
        if args.is_flatten == 0 and args.is_old_neg == 0:
            num_edges = args.num_pos_test_edge if analyze_mode == "valid" else args.num_pos_val_edge
            neg_preds = torch.reshape(neg_preds, [num_edges, -1])
            # pos_preds = torch.unsqueeze(pos_preds, dim=1) 
            # new_preds = torch.cat([pos_preds, neg_preds], dim=1)
            results = get_metric_score(pos_preds, neg_preds)
        else:
            pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
            pos_preds, neg_preds = pos_preds.cpu(), neg_preds.cpu()
            results = get_metric_score_origin(pos_preds, neg_preds)
        
        results_dict[idx] = results
    
    with open(f"{data_path}/motif_result_{args.is_old_neg}.txt", "wb") as f:
        pickle.dump(results_dict, f)

