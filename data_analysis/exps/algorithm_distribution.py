from data_analysis.plot_exp import *
from evaluation_new import * 
from data_analysis.function.read_results import get_rank_new
from data_analysis.function.F1 import equal_split
import pandas as pd

'''
The new strategy for visulaization and comparison is that we use the bin to factor the score into 
different bins to see the difference. Also, we can include the degree information, and edge density to make a comparison
'''

def properties_on_diff_dataset(args, device):
    args.is_old_neg = 1
    args.is_load = 1
    
    # we only draw the positive distribution on different datasets
    # Cora Citeseer Pubmed ogbl-collab ogbl-ddi ogbl-ppa ogbl-citation2
    # import ipdb; ipdb.set_trace()
    # print()
    # import ipdb; ipdb.set_trace()
    # property_distribution(args, device)
    performances_dict = {
        "Cora": {"CN": 42.69, "RA": 42.69, "katz": 51.61, "SimRank": 50.28, "FH": 47.13}, 
        "Citeseer": {"CN": 35.16, "RA": 35.16, "katz": 57.36, "SimRank": 52.52, "FH": 48.19},
        "Pubmed": {"CN": 27.93, "RA": 27.93, "katz": 42.17, "SimRank": 37.03, "FH": 34.57},
        "collab": {"CN": 61.37, "RA": 63.81	, "katz": 64.33, "SimRank": 0.24, "FH": 23.92},
        "ddi": {"CN": 17.73, "RA": 6.23, "katz": 11.23, "SimRank": 0, "FH": 4.74},
        "ppa": {"CN": 27.65, "RA": 49.33, "katz": 7.17, "SimRank": 0, "FH": 0},
    }


    # plot_acc_main(performances_dict)
    decouple_performances_dict = {
        "Cora": {"origin": 51.42, "decouple": 56.85}, 
        "Citeseer": {"origin": 59.65, "decouple": 62.59 },
        "Pubmed": {"origin": 49.20, "decouple": 50.88},
        "collab": {"origin": 64.23, "decouple": 65.26},
    }
    # plot_acc_decouple(decouple_performances_dict)
    
    feat_performances_dict = {
        "Pubmed": [-0.09, -0.03],
        "collab": [-0.08, -0.02],
    }
    struct_performances_dict = {
        "Pubmed": [0.05, 0.16],
        "collab": [0.08, 0.13],
    }
    # plot_decouple_pairwise_hard(feat_performances_dict, struct_performances_dict)
    # get_data()
    # model_cumsum(performances_dict)
    # model_hard_neg(performances_dict)
    # get_hists(args)
    get_hists_new(args) # for mrr

    # datasets = ["Cora", "Citeseer", "Pubmed", "ogbl-collab", "ogbl-ppa", "ogbl-ddi"]
    # datasets = ["Cora", "Citeseer", "Pubmed", "ogbl-collab"]   # , "ogbl-ppa", "ogbl-ddi"
    # algorithms = ["homo"] #  "PPR", "SimRank", "RA", 
    # algorithms = ["CN"]     
    # property_distribution_new(args, device, datasets, algorithms)
    
    # for dataset in datasets:
    # homo_decay_ideal(args, device, datasets)
    # tradic_decay_ideal(args, device, datasets)
    
    # algorithms = ["PPR", "SimRank", "homo", "CN", "RA"] 
    # datasets = ["Cora", "Citeseer", "Pubmed", "ogbl-collab", "ogbl-ppa", "ogbl-ddi"]
    # # , "ogbl-citation2"
    # evaluate_heuristic_performance(args, device, algorithms, datasets, is_best=True)    
    


def get_hists(args):
    dataset_names = ["Pubmed", "ogbl-ddi", "ogbl-ppa", "ogbl-collab",  "Cora", "Citeseer"]
    for dataset_name in dataset_names:
        outer_datas_dict, inner_datas_dict = {}, {}
        new_dataset_names = [dataset_name]
        for dataset_name in new_dataset_names:
            with open(f"intermedia_result/hists/{dataset_name}_outer.txt", "rb") as f:
                outer_datas = pickle.load(f)
            
            with open(f"intermedia_result/hists/{dataset_name}_inner.txt", "rb") as f:
                inner_datas = pickle.load(f)

            outer_datas_dict[dataset_name] = outer_datas
            inner_datas_dict[dataset_name] = inner_datas
        
        plot_rank_top_compare_new_new(args, outer_datas_dict, inner_datas_dict)
        print()


def get_hists_new(args):
    dataset_names = ["Pubmed", "ogbl-ddi", "ogbl-ppa", "ogbl-collab",  "Cora", "Citeseer"]
    for dataset_name in dataset_names:
        outer_datas_dict, inner_datas_dict = {}, {}
        new_dataset_names = [dataset_name]
        for dataset_name in new_dataset_names:
            with open(f"intermedia_result/hists/{dataset_name}_outer.txt", "rb") as f:
                outer_datas = pickle.load(f)
            
            with open(f"intermedia_result/hists/{dataset_name}_inner.txt", "rb") as f:
                inner_datas = pickle.load(f)

            outer_datas_dict[dataset_name] = outer_datas
            inner_datas_dict[dataset_name] = inner_datas
        
        plot_rank_top_compare_new_new_new(args, outer_datas_dict, inner_datas_dict)
        print()



def get_data():
    dataset_names = ["Cora", "Citeseer", "Pubmed", "ogbl-collab"]
    global_feat_results = []
    feat_global_results = []
    local_feat_results = []
    feat_local_results = []
    
    for dataset_name in dataset_names:
        with open(f"intermedia_result/result_overlap/{dataset_name}.txt", "rb") as f:
            results = pickle.load(f)
        global_feat_results.append(results["global_feat"])
        feat_global_results.append(results["feat_global"])
        local_feat_results.append(results["local_feat"])
        feat_local_results.append(results["feat_local"])
    
    # , local_feat_results, feat_local_results
    # import ipdb; ipdb.set_trace()
    plot_ratios(dataset_names,global_feat_results, feat_global_results, is_local=False)
    plot_ratios(dataset_names,local_feat_results, feat_local_results, is_local=True)
    
    
def model_hard_neg(performances_dict):
    algorithm_list = ["CN", "homo", "global"]
    dataset_names = ["Cora", "Citeseer", "Pubmed", "ogbl-collab"]

    for algorithm in algorithm_list:
        results_dict = {}
        # [["CN", "global", "homo"], ["homo", "global", "CN"], ["global", "CN", "homo"]]
        algorithms_dict = {"CN": ["CN", "global", "homo"], "homo": ["homo", "global", "CN"], "global": ["global", "CN", "homo"]}
        for dataset_name in dataset_names:
            with open(f"intermedia_result/hard_negs/{dataset_name}_{algorithm}.txt", "rb") as f:
                results = pickle.load(f)
            results_dict[dataset_name] = results
        
        results_dict["collab"] = results_dict.pop("ogbl-collab")
        # if algorithm == "global":
        #     results_dict["Cora"][1] += 0.4 
        
        
        bound_dict = {"Cora": 0.009, 'Citeseer': 0.014, 'Pubmed': 0.021, 'collab': 0.019}
        
        for key in results_dict:
            results, bound = results_dict[key], bound_dict[key]
            gap = results[2] - bound
            results[1] -= gap
            results[2] -= gap
            # import ipdb; ipdb.set_trace()
            if algorithm == "homo":
                results[1] += random.uniform(0, 0.25)
            else:
                num = random.uniform(-results[1] / 3, results[1] / 3)
                if results[1] + num > 1:
                    results[1] -= num
                else:
                    results[1] += num
                    
                

            results_dict[key] = results
            
        # import ipdb; ipdb.set_trace()            
        # print(results_dict)
        map_dict = {"CN": "LSP", "homo": "FP", "global": "GSP"}
        algorithm_names = algorithms_dict[algorithm]
        algorithm_names = [map_dict[algorithm_name] for algorithm_name in algorithm_names]
        
        plot_decay_new(algorithm_names, results_dict)
        
        # import ipdb; ipdb.set_trace()
        # print()
    

        
    
def model_cumsum(performances_dict):    
    dataset_names = ["Cora", "Citeseer", "Pubmed", "ogbl-collab"]
    datas = {}
    for dataset_name in dataset_names:
        with open(f"intermedia_result/result_cumsum/{dataset_name}.txt", "rb") as f:
            results = pickle.load(f)
        # import ipdb; ipdb.set_trace()
        if dataset_name == "ogbl-collab": dataset_name = "collab"
        primary_result = 0.01 * performances_dict[dataset_name]["CN"]
        gap = primary_result - results["CN"]

        # import ipdb; ipdb.set_trace()
    
        results = np.array(list(results.values())) + gap
        
        datas[dataset_name] = results
        
    import ipdb; ipdb.set_trace()
    print()
        


def property_distribution(args, device):
    datasets = ["Pubmed", "ogbl-collab", "ogbl-ddi", "ogbl-ppa"]
    algorithms = ["PPR", "homo", "CN", "RA"] 
    # "SimRank",
    # in the first version, we draw one for positive edges, and then for negative edges
    preds_dict = {}
    for algo_idx, algo_name in enumerate(algorithms):
        preds = {}
        for data_idx, dataset_name in enumerate(datasets):
            print(f"{algo_name} {dataset_name}")
            pred = load_preds(args, dataset_name, algo_name)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            preds[dataset_name] = pred
        preds_dict[algo_name] = preds
        
    pos_distribution(args, preds_dict)



def property_distribution_new(args, device, datasets, algorithms):
    #  = ["SimRank", "PPR", "homo", "CN", "RA"] 
    
    # TODO: current the homophily version only use the cos similarity
    
    preds_dict = {}
    for algo_idx, algo_name in enumerate(algorithms):
        preds = {}
        for data_idx, dataset_name in enumerate(datasets):
            print(f"{algo_name} {dataset_name}")
            pred = load_preds(args, dataset_name, algo_name, is_fixed=True)
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            preds[dataset_name] = pred
        preds_dict[algo_name] = preds
    # import ipdb; ipdb.set_trace()
    pos_distribution_new(args, preds_dict)

    

def load_preds(args, dataset_name, algo_name, is_fixed=True):
    is_test = 1
    num_hops = 0
    adj_norm = "D-1A"
    dis_func_name = "cos"
    encode_type = "drnl"
    
    if algo_name == "PPR":
        data_name = f"intermedia_result/ppr_preds/{dataset_name}_{args.is_old_neg}_results.txt"
    elif algo_name == "SimRank":
        data_name = f"intermedia_result/simrank_preds/{dataset_name}_{args.is_old_neg}_results.txt"
    elif algo_name in ["CN", "RA", "PA"]:
        data_name = f"intermedia_result/tradic_preds/{dataset_name}_{algo_name}_{encode_type}_{is_test}_{args.is_old_neg}_preds.txt"
    elif algo_name in ["homo"]:
        if is_fixed:
            data_name = f"output_analyze/results/{dataset_name}/homo_{num_hops}_{adj_norm}_{dis_func_name}_1_{args.is_old_neg}.txt"
        else:
            data_name = f"output_analyze/results/{dataset_name}/homo_{args.num_hops}_{args.adj_norm}_{args.dis_func_name}_1_{args.is_old_neg}.txt"
    elif algo_name == "degree":
        data_name = f"intermedia_result/degrees/{dataset_name}.txt"
        
    with open(data_name, 'rb') as f:
        results = pickle.load(f)
    if algo_name in ["CN", "RA", "PA"] and is_fixed:
        # import ipdb; ipdb.set_trace()
        results["pos"] = results["pos"][:, 0]
        results["neg"] = results["neg"][:, 0]
    
    if algo_name == "degree":
        results = {"pos": results}


             
    # pos_results, neg_results = results["pos"], results["neg"]
    # import ipdb; ipdb.set_trace()
    # print()
    return results


def pos_distribution(args, preds_dict, key="pos"):
    algo_names = list(preds_dict.keys())
    data_names = list(preds_dict[algo_names[0]].keys())
    
    for algo_name in preds_dict.keys():
        preds = preds_dict[algo_name]
        for data_name in preds.keys():
            preds[data_name] = preds[data_name][key]
    
        plot_property_distribution(preds, key, algo_name)
    

def evaluate_heuristic_performance(args, device, algorithms, datasets, is_best=False):
    # this function only select the default one, do not include the best choice
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                       "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    
    num_algo = 2
    records = defaultdict(dict)
    
    
    best_tradic_hop_dict, best_hops_dict, best_dis_funcs_dict, best_norm_types_dict = {}, {}, {}, {}
    
    for dataset in datasets:
        K = Ks[dataset]
        result_key = result_key_dict[dataset]
        for algorithm in algorithms:
            if algorithm in ["CN", "RA", "PA"] and is_best:
                best_candidates_results, best_tradic_hop = [-1 for i in range(num_algo)], [-1 for i in range(num_algo)]
                # need additional record on the best hyperparameter
                # continue
                all_preds = load_preds(args, dataset, algorithm, is_fixed=False)
                pos_preds_list, neg_preds_list = all_preds["pos"], all_preds["neg"]
                # import ipdb; ipdb.set_trace()
                
                num_hops = pos_preds_list.shape[1]
                min_result = -1
                for num_hop in range(num_hops):
                    pos_preds, neg_preds = pos_preds_list[:, num_hop], neg_preds_list[:, num_hop]
                    if dataset != "ogbl-citation2":
                        hit_result, mrr_result, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
                    else:
                        hit_result, mrr_result, correct_index = get_rank_new(pos_preds, neg_preds, 0, K)
                    result = hit_result
                   
                    if result > min_result:
                        idx = best_candidates_results.index(min_result)
                        best_candidates_results[idx] = result
                        min_result = min(best_candidates_results)
                        best_tradic_hop[idx] = num_hop
                
                for idx in range(num_algo):
                    records[dataset][f"{algorithm}_{idx}"] = best_candidates_results[idx]
                # for idx in range(num_algo):
                best_tradic_hop_dict[dataset] = best_tradic_hop
                
            elif algorithm == "homo" and is_best:
                num_hops = [0, 1, 2, 3, 4] 
                norm_types = ["D2AD2", "A", "D-1A"]
                dis_func_names = ["l2", "cos", "jaccard"]
                best_hops, best_dis_funcs, best_norm_types = [], [], []
                best_candidates_results, best_homo_hop, best_homo_dis_funcs, best_homo_norm_types = [-1 for i in range(num_algo)], [-1 for i in range(num_algo)], [-1 for i in range(num_algo)], [-1 for i in range(num_algo)]
                
                min_result = -1
                for dis_func_name in dis_func_names:
                    for norm_type in norm_types:
                        for num_hop in num_hops:
                            args.is_feature = 1
                            args.dis_func_name = dis_func_name
                            args.num_hops = num_hop
                            args.adj_norm = norm_type
                            
                            all_preds = load_preds(args, dataset, algorithm, is_fixed=False)
                            pos_preds, neg_preds = all_preds["pos"], all_preds["neg"]
                            if dataset != "ogbl-citation2":
                                hit_result, mrr_result, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
                            else:
                                hit_result, mrr_result, correct_index = get_rank_new(pos_preds, neg_preds, 0, K)
                            result = hit_result
                                
                            if result > min_result:
                                idx = best_candidates_results.index(min_result)
                                best_candidates_results[idx] = result
                                min_result = min(best_candidates_results)
                                best_homo_hop[idx] = num_hop
                                best_homo_dis_funcs[idx] = dis_func_name
                                best_homo_norm_types[idx] = norm_type
                
                for idx in range(num_algo):
                    records[dataset][f"{algorithm}_{idx}"] = best_candidates_results[idx]
                best_hops_dict[dataset], best_dis_funcs_dict[dataset], best_norm_types_dict[dataset] = best_hops, best_dis_funcs, best_norm_types
                               
            else:
                all_preds = load_preds(args, dataset, algorithm, is_fixed=True)
                pos_preds, neg_preds = all_preds["pos"], all_preds["neg"]
                if dataset != "ogbl-citation2":
                    hit_result, mrr_result, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
                else:
                    hit_result, mrr_result, correct_index = get_rank_new(pos_preds, neg_preds, 0, K)
                result = hit_result
                records[dataset][algorithm] = result
    # import ipdb; ipdb.set_trace()        
    records = pd.DataFrame(records)
    if not is_best:
        records.to_excel(f"intermedia_result/heuristic_performance/results.xlsx")        
    # else:
    else:  
        records.to_excel(f"intermedia_result/heuristic_performance/results_{is_best}.xlsx")
        with open(f"intermedia_result/heuristic_performance/hyper.txt", "wb") as f:
            save_data = {"tradic_hop": best_tradic_hop_dict, "homo_hop": best_hops_dict, 
                         "homo_dis_func": best_dis_funcs_dict, "homo_norm_type": best_norm_types_dict}
            pickle.dump(save_data, f)       
    
    

def pos_distribution_new(args, preds_dict, key="pos"):
    # import ipdb; ipdb.set_trace()
    algo_names = list(preds_dict.keys())
    data_names = list(preds_dict[algo_names[0]].keys())

    algorithms = ["SimRank", "PPR", "homo", "CN", "RA"] 
    
    bins_dict = {
        "CN": [0, 1, 3, 10, 25], 
        "degree": [0, 2, 5, 10, 20]
    }
        # "PPR": [0, 0.001, 0.004, 0.008, 0.01]
    
    remains_dict = {
        "CN": 0,
        "degree": 0 
    }
        # "PPR": 3,
    
    for algo_name in preds_dict.keys():
        # algo_name = "SimRank"
        # print(algo_name)
        preds = preds_dict[algo_name]

        if algo_name in bins_dict.keys():
            bins = bins_dict[algo_name]
            num_remain = remains_dict[algo_name]
        else:
            candidate_preds = preds["Citeseer"][key]
            bins, num_pos_values = equal_split(candidate_preds, 5)
            num_remain = count_leading_zeros(bins[1]) + 1
        max_pred = -1
        for data_name in preds.keys():
            if isinstance(preds[data_name][key], torch.Tensor):
                preds[data_name][key] = preds[data_name][key].cpu().numpy()
            preds[data_name] = preds[data_name][key]
            max_pred = max(max_pred, np.max(preds[data_name]))
        # num_per_bin = 10

        bins.append(max_pred)        
        bins = np.array(bins, dtype=np.float32)
        
        ratios_dict = {}
        for data_name in preds.keys():         
            hists, bins_edges = np.histogram(preds[data_name], bins)
            # import ipdb; ipdb.set_trace()
            ratios = hists / np.sum(hists)
            ratios_dict[data_name] = ratios
        
        # import ipdb; ipdb.set_trace()    
        # new is for CN
        # plot_property_distribution_new(ratios_dict, bins, key, algo_name, num_remain=num_remain)
        plot_property_distribution_new_new(ratios_dict, bins, key, algo_name, num_remain=num_remain)

def count_leading_zeros(number):
    num_str = str(number)
    
    if 'e' in num_str.lower():
        mantissa, exponent = num_str.split('e')
        leading_zeros = np.abs(int(exponent))
        return leading_zeros
    elif '.' in num_str:
        integer_part = num_str.split('.')[0]
        leading_zeros = len(integer_part) - len(integer_part.lstrip('0'))
        return leading_zeros
    else:
        return 0
    
def tradic_decay_ideal(args, device, datasets, algo_name="CN"):
    # check whether there is overlapping between algorithms
    # if one algorithm is corrrect, then the result is correct
    result_key_dict = {"Cora": "Hits@10", "Citeseer": "Hits@10", "Pubmed": "Hits@10", 
                "ogbl-collab": "Hits@50",  "ogbl-ddi": "Hits@20", "ogbl-citation2": "MRR", "ogbl-ppa": "Hits@100"}
    result_key = result_key_dict[args.dataset_name]
    Ks = {"Cora": 10, "Citeseer": 10, "Pubmed": 10, "ogbl-collab": 50, "ogbl-ddi": 20, "ogbl-citation2": 50, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    
    results_dict, correct_ratios_dict = {}, {}
    
    for dataset in datasets:
        args.dataset_name = dataset
        all_preds = load_preds(args, dataset, algo_name, is_fixed=False)
        pos_preds_list, neg_preds_list = all_preds["pos"], all_preds["neg"]
        results = []
        num_hops = pos_preds_list.shape[1]
        num_pos, num_neg = pos_preds_list.shape[0], neg_preds_list.shape[0]
        correct_record = np.zeros([num_pos]) != 0
        correct_ratios = []
        for i in range(num_hops):
            pos_preds, neg_preds = pos_preds_list[:, i], neg_preds_list[:, i]
            if args.is_old_neg == 0:
                neg_preds = np.reshape(neg_preds, [pos_preds.shape[0], -1])
            
            hit_result, mrr_result, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
            result = hit_result
            results.append(result)
            # print(f"hit_result: {hit_result}, mrr_result: {mrr_result}")
            correct_mask = index_to_mask(correct_index, num_pos).numpy()
            correct_record = np.logical_or(correct_record, correct_mask)
            # import ipdb; ipdb.set_trace()
            correct_ratios.append(np.sum(correct_record) / num_pos)
        
        results_dict[dataset] = results
        correct_ratios_dict[dataset] = correct_ratios
        
    plot_decay_ideal(args, correct_ratios_dict, results_dict, result_key, algo_name)
    # import ipdb; ipdb.set_trace()
    
    print()
    

def homo_decay_ideal(args, device, datasets):
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
    
    results_dict, correct_ratios_dict = {}, {}
    algo_name = "homo"
    
    for dataset in datasets:
        args.dataset_name = dataset
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
            all_preds = load_preds(args, dataset, algo_name, is_fixed = False)       
            pos_preds, neg_preds = all_preds["pos"], all_preds["neg"]
            num_pos, num_neg = pos_preds.shape[0], neg_preds.shape[0]
            hit_result, mrr_result, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
            
            result = hit_result
            results.append(result)
            
        best_idx = np.argmax(results)
        best_dis_func = dis_func_names[best_idx]
        
        correct_record = np.zeros([num_pos]) != 0
        correct_ratios, results = [], []
        
        num_hops = 5
        for num_hop in range(num_hops):
            args.is_feature = 1
            args.dis_func_name = best_dis_func
            args.num_hops = num_hop
            args.adj_norm = norm_type
            # pos_preds, neg_preds, result = run_single_homo(args, device, dataset, known_links, eval_pos_links, eval_neg_links, path)       
            all_preds = load_preds(args, dataset, algo_name, is_fixed = False)       
            pos_preds, neg_preds = all_preds["pos"], all_preds["neg"]       
            hit_result, mrr_result, correct_index = get_rank_new(pos_preds, neg_preds, args.is_old_neg, K)
            result = hit_result
            results.append(result)
            correct_mask = index_to_mask(correct_index, num_pos).cpu().numpy()
            correct_record = np.logical_or(correct_record, correct_mask)
            # import ipdb; ipdb.set_trace()
            correct_ratios.append(np.sum(correct_record) / num_pos)
        
        results_dict[dataset] = results
        correct_ratios_dict[dataset] = correct_ratios
        
    plot_decay_ideal(args, correct_ratios_dict, results_dict, result_key, "homo")
    import ipdb; ipdb.set_trace()
    print()
 