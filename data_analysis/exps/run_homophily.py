from data_analysis.function.homophily import *
from data_analysis.function.functional import *
from data_analysis.plot_exp import *
from data_analysis.generate_data import load_data
import scipy.sparse as sp
from evaluation_new import * 
import torch
import os
import pathlib

def run_homophily(args, device):
    args.is_generate_train = 1   # whether preprocess on just training set or entire dataset
    args.is_old_neg = 1            # whether use the new heart negative sampling method
    args.analyze_mode = "test"  # "whole" "valid" "test"
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

    args.is_load = 0
    dataset, known_links, eval_pos_links, eval_neg_links, path = load_data(args, device)

    num_hops = [0, 1, 2, 3, 4]
    norm_types = ["D2AD2", "A", "D-1A"]
    dis_func_names = ["l2", "cos", "jaccard", "kmeans"]
    for norm_type in norm_types:
        for num_hop in num_hops:
            for dis_func_name in dis_func_names:
                args.is_feature = 1 if dis_func_names in ["l2", "cos", "jaccard"] else 0
                args.dis_func_name = dis_func_name
                args.num_hops = num_hop
                args.adj_norm = norm_type
                run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
                
                # try:
                #     run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
                # except:
                #     continue
                    

def run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path):
    data_path = f"output_analyze/results/{args.dataset_name}"
    folder_path = pathlib.Path(data_path) 
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    if args.is_feature:
        file_name = f"homo_{args.num_hops}_{args.adj_norm}_{args.dis_func_name}_{args.is_feature}_{args.is_old_neg}"
    else:
        file_name = f"homo_{args.num_hops}_{args.adj_norm}_{args.label_name}_{args.is_feature}_{args.is_old_neg}"

    if not args.is_load:
        if args.is_feature:
            pos_preds, neg_preds = feature_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
        else:
            pos_preds, neg_preds = label_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, label_name=args.label_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
        
        '''
        args.num_hops = 2
        pos_preds2, neg_preds2 = label_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, label_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
        # pos_preds2, neg_preds2 = feature_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
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
            if args.is_feature:
                pos_preds, neg_preds = feature_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, dis_func_name=args.dis_func_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
            else:
                pos_preds, neg_preds = label_homophily_ratio(args, dataset, known_links, eval_pos_links, eval_neg_links, label_name=args.label_name, predefine_group=args.adj_norm, is_norm=args.is_norm, is_feature_norm=args.is_feature_norm)    
            
            with open(f"{data_path}/{file_name}.txt", "wb") as f:
                pickle.dump({"pos": pos_preds, "neg": neg_preds}, f)
        

        # "kmeans", "GMM", "SC" 
        # Notice, GMM and SC is quite slow, do not know whjether need accurate on other method with GPU
        # TODO: check the predict form
    # results = get_metric_score(pos_preds, neg_preds)

    
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
    
    if args.is_feature:
        file_name = f"homo_result_{args.adj_norm}_{args.dis_func_name}_{args.is_feature}_{args.is_old_neg}"
    else:
        file_name = f"homo_result_{args.adj_norm}_{args.label_name}_{args.is_feature}_{args.is_old_neg}"

    if args.is_feature:
        result_strings = [f"{args.dataset_name}_homo_{args.adj_norm}_{args.dis_func_name}_{args.is_feature}"]
    else:
        result_strings = [f"{args.dataset_name}_homo_{args.adj_norm}_{args.label_name}_{args.is_feature}"]


    #  = [f"{args.dataset_name}_homo_{args.num_hops}_{args.norm_type}"]
    for key in results:
        result_strings.append(f"{key}:{results[key]}")
    sep = " "
    result_string = sep.join(result_strings)+"\n"

    if args.is_feature:
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


    