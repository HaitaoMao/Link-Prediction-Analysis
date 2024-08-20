from data_analysis.function.homophily import *
from data_analysis.function.functional import *
from data_analysis.plot_exp import *
from data_analysis.generate_data import load_data
from data_analysis.plot_exp import plot_homo_hop, plot_homo_difference
import scipy.sparse as sp
from evaluation_new import * 
import torch
import os
import pathlib
from data_analysis.function.read_results import generate_rank_single
from data_analysis.function.F1 import F1


'''
This file is an adaptive function of the homophily, we aims to use the analysis result for see whether
homophily can really effected by 
different aggregation function, different hop and different normalized factor
TODO: We still need one hadarmard based result
'''



def run_homophily_hop_analysis(args, device):
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
    args.is_log = 0

    args.is_load = 0
    dataset, known_links, eval_pos_links, eval_neg_links, path = load_data(args, device)
    analysis_norm(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
    # analysis_overlapping(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
    # analysis_dis(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
    
    # analysis_overlapping, analysis_norm, analysis_dis
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
        results_lists, pos_preds_lists, neg_preds_lists, hops_lists = [], [], [], []
        # list of list, [norm_type, num_hop: result]
        for norm_type in norm_types:
            results_list, pos_preds_list, neg_preds_list, hops_list = [], [], [], []
             
            for num_hop in num_hops:
                # print(num_hop)
                args.is_feature = 1
                args.dis_func_name = dis_func_name
                args.num_hops = num_hop
                args.adj_norm = norm_type
                pos_preds, neg_preds, results = run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       

                try:
                    # import ipdb; ipdb.set_trace()
                    pos_preds, neg_preds, results = run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
                    results_list.append(results[result_key])
                    pos_preds_list.append(pos_preds)
                    neg_preds_list.append(neg_preds)
                    hops_list.append(num_hop)
                    # if args.num_hops == 0:
                    #     print(pos_preds[:10].sum())
                    #     print(results[result_key])
                    # import ipdb; ipdb.set_trace()
                except:
                    print(norm_type)
                    continue
            if len(results_list) != len(num_hops):
                print("incomplete results")
            results_lists.append(results_list)
            pos_preds_lists.append(pos_preds_list)
            neg_preds_lists.append(neg_preds_list)
            hops_lists.append(hops_list)
            
            # print(norm_type)
            # print(dis_func_name)
            # print(results_list)
    # import ipdb; ipdb.set_trace()
    plot_homo_hop(results_lists, len(num_hops), norm_types, args.dataset_name, dis_func_name, "norm", result_key)    


def analysis_logits(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path):
    # analysis one will focus on drawing results on different hops
    # we will use the same aggregation function and the same distance function
    num_hops = [0, 1, 2, 3, 4] 
    norm_types = ["D2AD2", "A", "D-1A"]
    
    dis_func_names = ["l2", "cos", "jaccard"]
    result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    
    results_lists, pos_preds_lists, neg_preds_lists, hops_lists = [], [], [], []
    
    for norm_type in norm_types:
        results_list, pos_preds_list, neg_preds_list, hops_list = [], [], [], []         
        for num_hop in num_hops:
            args.is_feature = 1
            args.dis_func_name = dis_func_name
            args.num_hops = num_hop
            args.adj_norm = norm_type
            pos_preds, neg_preds, results = run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
            
            try:
                pos_preds, neg_preds, results = run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
                results_list.append(results[result_key])
                pos_preds_list.append(pos_preds)
                neg_preds_list.append(neg_preds)
                hops_list.append(num_hop)
                # import ipdb; ipdb.set_trace()
            except:
                print(norm_type)
                continue
        results_lists.append(results_list)
        pos_preds_lists.append(pos_preds_list)
        neg_preds_lists.append(neg_preds_list)
        hops_lists.append(hops_list)
            
        if len(results_list) != len(num_hops):
            print("incomplete results")
            
        plot_homo_hop(results_lists, len(num_hops), norm_types, args.dataset_name, None, "logits", result_key)    
        # import ipdb; ipdb.set_trace()

    


def analysis_dis(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path):
    # very similar analysis
    # analysis one will focus on drawing results on different hops
    # we will use the same aggregation function and the same distance function
    num_hops = [0, 1, 2, 3, 4]
    norm_types = ["D2AD2", "A", "D-1A"]
    dis_func_names = ["l2", "cos", "jaccard"]
    result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    
        
    for norm_type in norm_types:
        results_lists, pos_preds_lists, neg_preds_lists, hops_lists = [], [], [], []
        # list of list, [norm_type, num_hop: result]
        for dis_func_name in dis_func_names:
            results_list, pos_preds_list, neg_preds_list, hops_list = [], [], [], []
            for num_hop in num_hops:
                args.is_feature = 1
                args.dis_func_name = dis_func_name
                args.num_hops = num_hop
                args.adj_norm = norm_type
                # pos_preds, neg_preds, results = run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
                try:
                    pos_preds, neg_preds, results = run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
                    results_list.append(results[result_key])
                    pos_preds_list.append(pos_preds)
                    neg_preds_list.append(neg_preds)
                    hops_list.append(num_hop)
                    
                    # import ipdb; ipdb.set_trace()
                except:
                    continue
            if len(results_list) != len(num_hops):
                print("incomplete results")
            results_lists.append(results_list)
            pos_preds_lists.append(pos_preds_list)
            neg_preds_lists.append(neg_preds_list)
            hops_lists.append(hops_list)
            
            plot_homo_hop(results_lists, len(num_hops), dis_func_names, args.dataset_name, norm_type, "dis", result_key)    
          
            # import ipdb; ipdb.set_trace()
        



def analysis_overlapping(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path):
    # very similar analysis
    # analysis one will focus on drawing results on different hops
    # we will use the same aggregation function and the same distance function
    if args.dataset_name == "ogbl-citation2":
        import ipdb; ipdb.set_trace()
        # check the output type, whether it is still a list, how to handle this case
    
    
    num_hops = [0, 1, 2, 3, 4]
    norm_types = ["D2AD2", "A", "D-1A"]
    
    is_dis = 1
    selected_dis_func_name = "cos"
    selected_norm_type = "D-1A"
    dis_func_names = ["l2", "cos", "jaccard"]
    
    result_key_dict = {"Cora": "Hits@100", "Citeseer": "Hits@100", "Pubmed": "Hits@100", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    
    Ks = {"Cora": 100, "Citeseer": 100, "Pubmed": 100, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 100, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    
    candidate_names = norm_types if is_dis else dis_func_names
    
    for candidate_name in candidate_names:
        results_list, pos_preds_list, neg_preds_list, hops_list = [], [], [], []
        for num_hop in num_hops:
            args.is_feature = 1
            if is_dis:
                args.dis_func_name = selected_dis_func_name
                args.adj_norm = candidate_name
            else:
                args.dis_func_name = candidate_name
                args.adj_norm = selected_norm_type
            args.num_hops = num_hop
            # pos_preds, neg_preds, results = run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       

            try:
                pos_preds, neg_preds, results = run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
                results_list.append(results[result_key])
                pos_preds_list.append(pos_preds)
                neg_preds_list.append(neg_preds)
                hops_list.append(num_hop)
                # import ipdb; ipdb.set_trace()
            except:
                continue
        if len(results_list) != len(num_hops):
            print("incomplete results")
        
        differences_list = defaultdict(list)
        for i in range(len(pos_preds_list) - 1):
            pos_preds = pos_preds_list[i]
            num_pos = pos_preds.shape[0]
            neg_preds = neg_preds_list[i]
            pos_preds_next = pos_preds_list[i + 1]
            neg_preds_next = neg_preds_list[i + 1]
            
            # import ipdb; ipdb.set_trace()
            rank1 = generate_rank_single(pos_preds, neg_preds)
            rank2 = generate_rank_single(pos_preds_next, neg_preds_next)

            F1_results = F1(rank1, rank2, num_pos=num_pos, K=K)
            for key in F1_results.keys():
                differences_list[key].append(F1_results[key])
        
        plot_homo_difference(differences_list, len(num_hops), candidate_name, is_dis, args.dataset_name, result_key)    
        # import ipdb; ipdb.set_trace()
        

def run_single(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path):
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

    # import ipdb; ipdb.set_trace()
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
            
    # if args.num_hop == 0:
    #     print(pos_preds[:10])
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
    
    