from data_analysis.plot_exp import *
from data_analysis.function.functional import *
from data_analysis.generate_data import load_data
from data_analysis.function.heuristics import CN, CN_new
from evaluation_new import * 
import scipy.sparse as sp
import torch
import os
import pathlib
import pandas as pd


def run_tradic(args, device):
    args.is_generate_train = 1   # whether preprocess on just training set or entire dataset
    args.is_old_neg = 1            # whether use the new heart negative sampling method
    args.analyze_mode = "test"  # "whole" "valid" "test"
    args.is_flatten = 0  # if use the heart, whether remove the redudant validation and test edge 
    args.is_remove_redudant = 1  # if use the heart, whether remove the redudant validation and test edge 
    args.batch_size = 1000
    args.ratio_per_hop = 1.0
    
    dataset, known_links, eval_pos_links, eval_neg_links, path = load_data(args, device)
    print(f"known links shape {known_links.shape}")
    num_nodes = torch.max(known_links).item() + 1
    select_xlim = 1
    select_lims_dict = {"Cora": [0, 1], "Citeseer": [0, 1], "Pubmed": [0, 1], 
        "ogbl-citation2": [0, 100], "ogbl-ppa": [0, 100], "ogbl-ddi": [0, 1000], "ogbl-collab": [0, 100]}
    if select_xlim:
        xlim = select_lims_dict[args.dataset_name]
    else:
        xlim=-1
    # if xlim = -1, use the default setting 
    # tradic closure 

    args.is_load = 1
    data_path = f"output_analyze/results/{args.dataset_name}"
    folder_path = pathlib.Path(data_path) 
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    # pos_tradic_results, neg_tradic_results = pos_tradic_results.cpu().numpy(), neg_tradic_results.cpu().numpy()

    if not args.is_load:
        pos_tradic_results, neg_tradic_results = tradic_count_nonremoval(known_links, dataset, path, args, eval_pos_links, eval_neg_links)
        with open(f"{data_path}/tradic_{args.is_old_neg}.txt", "wb") as f:
            pickle.dump({"pos": pos_tradic_results, "neg": neg_tradic_results}, f)
    else:
        with open(f"{data_path}/tradic_{args.is_old_neg}.txt", "rb") as f:
            data_dict = pickle.load(f)
        pos_tradic_results = data_dict["pos"]
        neg_tradic_results = data_dict["neg"]   
    num_hops = pos_tradic_results.shape[0]

    for num_hop1 in range(num_hops):
        for num_hop2 in range(num_hops):
            pos_tradic_result, neg_tradic_result = pos_tradic_results[num_hop1, num_hop2], neg_tradic_results[num_hop1, num_hop2]
            # import ipdb; ipdb.set_trace()
            # pos_tradic_result = torch.tensor(pos_tradic_result)
            # data = pos_tradic_result != pos_preds
            xname = f"tradic_{num_hop1+1}_{num_hop2+1}"
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
            elif args.dataset_name == "ogbl-citation2":
                pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
                results = get_metric_score(pos_preds, neg_preds)
            else:
                pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
                results = get_metric_score_origin(pos_preds, neg_preds)
            result_strings = [f"{args.dataset_name}_tradic_{num_hop1}_{num_hop2}"]
            for key in results:
                result_strings.append(f"{key}:{results[key]}")
            sep = " "
            result_string = sep.join(result_strings) +"\n"
            with open(f"{data_path}/ALL_tradic_result_{args.is_old_neg}.txt", "w") as f:
                f.write(result_string)

            results_dict[f"{num_hop1}_{num_hop2}"] = results
    
            # load data
            result_name = f"output_analyze/all_results/{args.dataset_name}_{args.is_old_neg}.xlsx"
            metric_names = ["algorithm", 'Hits@1', 'Hits@3', 'Hits@10', 'Hits@100', "MRR"]
            try:
                results_record = pd.read_excel(result_name)
            except:
                results_record =pd.DataFrame(columns=metric_names)
                # results_record.set_index("algorithm", inplace=True) 
                
            algorithm_key = f"tradic_{num_hop1}_{num_hop2}"
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
                
    with open(f"{data_path}/tradic_result_{args.is_old_neg}.txt", "wb") as f:
        pickle.dump(results_dict, f)

    
    '''        
    # pos_preds, neg_preds = label_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, label_name="kmeans", is_norm=False, is_feature_norm=False)    
    # "kmeans", "GMM", "SC" 
    # Notice, GMM and SC is quite slow, do not know whjether need accurate on other method with GPU
    # pos_preds, neg_preds = feature_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, dis_func_name="cos", is_norm=False, is_feature_norm=False)    
    if args.is_flatten == 0 and args.is_old_neg == 0:
        num_edges = args.num_pos_test_edge if analyze_mode == "valid" else args.num_pos_val_edge
        neg_preds = torch.reshape(neg_preds, [num_edges, -1])
        # pos_preds = torch.unsqueeze(pos_preds, dim=1) 
        # new_preds = torch.cat([pos_preds, neg_preds], dim=1)
    else:
        pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
    pos_preds, neg_preds = pos_preds.cpu(), neg_preds.cpu()
    results = get_metric_score(pos_preds, neg_preds)
    

    motif_count(args, known_links, eval_pos_links)
    num_nodes = torch.max(dataset.data.edge_index).item() + 1
    pos_preds = get_path_score(args, known_links.T, eval_pos_links, predefine_group="D2AD2")
    neg_preds = get_path_score(args, known_links.T, eval_neg_links, predefine_group="D2AD2")
    results_old = get_metric_score(pos_preds, neg_preds)
    
    indices = known_links.cpu().numpy()
    # [num_edges, 2]
    values = np.ones([np.max(indices.shape)])
    
    A = sp.coo_matrix((np.ones([np.max(indices.shape)]), (indices[:, 0], indices[:, 1])), shape=(num_nodes, num_nodes)).tocsr()
    pos_preds, _ = CN(A, eval_pos_links.cpu())
    neg_preds, _ = CN(A, eval_neg_links.cpu())
    # [num_edges] or [num_edges, 1]
    results_old = get_metric_score(pos_preds, neg_preds)

    results_new, pos_preds, neg_preds = generalized_CN(args, num_hop1, num_hop2, batch_size, dataset, path, known_links, eval_pos_links, eval_neg_links)
    if args.is_flatten == 0 and args.is_old_neg == 0:
        num_edges = args.num_pos_test_edge if analyze_mode == "valid" else args.num_pos_val_edge
        neg_preds = torch.reshape(neg_preds, [num_edges, -1])
        # pos_preds = torch.unsqueeze(pos_preds, dim=1) 
        # new_preds = torch.cat([pos_preds, neg_preds], dim=1)
    else:
        pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
    pos_preds, neg_preds = pos_preds.cpu(), neg_preds.cpu()
    results_new = get_metric_score(pos_preds, neg_preds)
    import ipdb; ipdb.set_trace()
    print()
    '''

    
    '''
    indices = known_links.cpu().numpy()
    # [num_edges, 2]
    values = np.ones([np.max(indices.shape)])
    
    A = sp.coo_matrix((np.ones([np.max(indices.shape)]), (indices[:, 0], indices[:, 1])), shape=(num_nodes, num_nodes)).tocsr()
    pos_pred1s, _ = CN(A, eval_pos_links.cpu())
    neg_pred1s, _ = CN(A, eval_neg_links.cpu())
    # [num_edges] or [num_edges, 1]
    # results_old = get_metric_score(pos_preds, neg_preds)

    # results_new, pos_preds, neg_preds = generalized_CN(args, num_hop1, num_hop2, batch_size, dataset, path, known_links, eval_pos_links, eval_neg_links)
    # if args.is_flatten == 0 and args.is_old_neg == 0:
    #     num_edges = args.num_pos_test_edge if analyze_mode == "valid" else args.num_pos_val_edge
    #     neg_preds = torch.reshape(neg_preds, [num_edges, -1])
    #     # pos_preds = torch.unsqueeze(pos_preds, dim=1) 
    #     # new_preds = torch.cat([pos_preds, neg_preds], dim=1)
    # else:
    #     pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
    # pos_preds, neg_preds = pos_preds.cpu(), neg_preds.cpu()
    # results_new = get_metric_score(pos_preds, neg_preds)
    # import ipdb; ipdb.set_trace()
    # print()
    '''
