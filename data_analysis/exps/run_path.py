from data_analysis.function.path_count import *
from data_analysis.function.functional import *
from evaluation_new import * 
from data_analysis.plot_exp import *
from data_analysis.generate_data import load_data
import scipy.sparse as sp
import torch
import os
import pathlib
import scipy.sparse as sp

def run_path(args, device):
    args.is_generate_train = 1   # whether preprocess on just training set or entire dataset
    args.is_old_neg = 1            # whether use the new heart negative sampling method
    args.analyze_mode = "test"  # "whole" "valid" "test"
    args.is_flatten = 1  # if use the heart, whether remove the redudant validation and test edge 
    args.is_remove_redudant = 1  # if use the heart, whether remove the redudant validation and test edge 

    # args.norm_type = "D2AD2"

    args.batch_size = 1000

    dataset, known_links, eval_pos_links, eval_neg_links, path = load_data(args, device)

    args.is_load = 1

    num_hops = [0, 1, 2, 3, 4]
    norm_types = ["D2AD2", "A", "D-1A"]
    for norm_type in norm_types:
        for num_hop in num_hops:
            args.num_hops = num_hop
            args.norm_type = norm_type
            try:
                run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
            except:
                continue
                
def run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path):
    num_nodes = torch.max(dataset.data.edge_index).item() + 1
    # import ipdb; ipdb.set_trace()

    data_path = f"output_analyze/results/{args.dataset_name}"
    folder_path = pathlib.Path(data_path) 
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    
    if not args.is_load:
        pos_preds = get_path_score2(args, known_links.T, eval_pos_links, predefine_group="D2AD2")
        neg_preds = get_path_score2(args, known_links.T, eval_neg_links, predefine_group="D2AD2")
        with open(f"{data_path}/path_{args.num_hops}_{args.norm_type}_{args.is_old_neg}.txt", "wb") as f:
            pickle.dump({"pos": pos_preds, "neg": neg_preds}, f)
    else:
        try:
            with open(f"{data_path}/path_{args.num_hops}_{args.norm_type}_{args.is_old_neg}.txt", "rb") as f:
                data_dict = pickle.load(f)
            pos_preds = data_dict["pos"]
            neg_preds = data_dict["neg"]  
        except:
            pos_preds = get_path_score2(args, known_links.T, eval_pos_links, predefine_group="D2AD2")
            neg_preds = get_path_score2(args, known_links.T, eval_neg_links, predefine_group="D2AD2")
            with open(f"{data_path}/path_{args.num_hops}_{args.norm_type}_{args.is_old_neg}.txt", "wb") as f:
                pickle.dump({"pos": pos_preds, "neg": neg_preds}, f) 

    xname = f"path_{args.norm_type}"
    # the name in the x axis
    name = f"{args.dataset_name}_old" if args.is_old_neg else f"{args.dataset_name}" 
    pos_preds, neg_preds = pos_preds.cpu().numpy(), neg_preds.cpu().numpy()
    plot_property(pos_preds, neg_preds, name=name, xname=xname)
    

    pos_preds, neg_preds = torch.tensor(pos_preds), torch.tensor(neg_preds)
    if args.is_flatten == 0 and args.is_old_neg == 0:
        num_edges = args.num_pos_test_edge if analyze_mode == "valid" else args.num_pos_val_edge
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
    # import ipdb; ipdb.set_trace()

    result_strings = [f"{args.dataset_name}_path_{args.num_hops}_{args.norm_type}"]
    for key in results:
        result_strings.append(f"{key}:{results[key]}")
    sep = " "
    result_string = sep.join(result_strings)+"\n"

    with open(f"{data_path}/ALL_path_result_{args.is_old_neg}.txt", "w") as f:
        f.write(result_string)

    with open(f"{data_path}/path_result_{args.num_hops}_{args.norm_type}_{args.is_old_neg}.txt", "wb") as f:
        pickle.dump(results, f)

    result_name = f"output_analyze/all_results/{args.dataset_name}_{args.is_old_neg}.xlsx"
    metric_names = ["algorithm", 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@100', "MRR"]
    try:
        results_record = pd.read_excel(result_name)
    except:
        results_record =pd.DataFrame(columns=metric_names)
        # results_record.set_index("algorithm", inplace=True) 
    algorithm_key =  f"path_{args.num_hops}_{args.norm_type}"
    
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

    
    