import pickle
import os
import torch
import numpy as np
from evaluation_new import * 
from collections import defaultdict
import copy
from torch_geometric.utils import mask_to_index

# This function aims to provide load the prediction and performance
def load_heuristic_results(mode_name, args, remains):
    data_path = f"output_analyze/results/{args.dataset_name}"
    # data_path = f"{data_path}_old" if args.is_old_neg else f"{data_path}"
    data_path = f"{data_path}" 
    mode_dict = {1: "tradic", 2: "motif", 3: "path", 4: "homo"}
    if mode_name in mode_dict.keys():
        mode_name = mode_dict[mode_name]
    readed_data = []

    if mode_name == "tradic":
        model_names, pos_preds_list, neg_preds_list, results = [], [], [], []
        pred_path = f"tradic_{args.is_old_neg}.txt"
        result_path = f"tradic_result_{args.is_old_neg}.txt"
        with open(f"{data_path}/{pred_path}", 'rb') as f:
            preds = pickle.load(f)
        with open(f"{data_path}/{result_path}", 'rb') as f:
            result = pickle.load(f)
        pos_preds, neg_preds = preds['pos'], preds['neg']
        for hop1 in range(len(pos_preds)):
            for hop2 in range(len(pos_preds[0])):
                pos_pred, neg_pred = pos_preds[hop1][hop2], neg_preds[hop1][hop2]
                if f"tradic_{hop1}_{hop2}" not in remains:
                    continue
                results.append(result[f"{hop1}_{hop2}"])
                model_names.append(f"tradic_{hop1}_{hop2}")
                pos_preds_list.append(pos_pred)
                neg_preds_list.append(neg_pred)

        return pos_preds_list, neg_preds_list, results, model_names
        
    elif mode_name == "motif":
        model_names, pos_preds_list, neg_preds_list, results, descriptions_list = [], [], [], [], []
        pred_path = f"motif_{args.is_old_neg}.txt"
        result_path = f"motif_result_{args.is_old_neg}.txt"
        with open(f"{data_path}/{pred_path}", 'rb') as f:
            preds = pickle.load(f)
        with open(f"{data_path}/{result_path}", 'rb') as f:
            result = pickle.load(f)
        
        pos_preds, neg_preds, descriptions = preds['pos'], preds['neg'], preds['description']
        for idx in range(len(pos_preds)):
            if f"motif_{idx}" not in remains:
                continue
                
            pos_pred, neg_pred = pos_preds[idx], neg_preds[idx]
            results.append(result[idx])  # f"idx"
            model_names.append(f"motif_{idx}")
            pos_preds_list.append(pos_pred)
            neg_preds_list.append(neg_pred)
            descriptions_list.append(descriptions[idx])

        return pos_preds_list, neg_preds_list, results, model_names
        
    elif mode_name == "path":
        model_names, pos_preds_list, neg_preds_list, results = [], [], [], []
        norm_types = ["D2AD2", "A", "D-1A"]
        num_hops = [0, 1, 2, 3, 4]
        for norm_type in norm_types:
            for num_hop in num_hops:
                pred_path = f"path_{num_hop}_{norm_type}_{args.is_old_neg}.txt"
                result_path = f"path_result_{num_hop}_{norm_type}_{args.is_old_neg}.txt"
                try:
                    if f"path_{norm_type}_{num_hop}" not in remains:
                        continue
                    with open(f"{data_path}/{pred_path}", 'rb') as f:
                        preds = pickle.load(f)
                    with open(f"{data_path}/{result_path}", 'rb') as f:
                        result = pickle.load(f)
                    pos_preds, neg_preds = preds['pos'], preds['neg']
                    model_names.append(result[f"path_{norm_type}_{num_hop}"])
                    pos_preds_list.append(pos_preds)
                    neg_preds_list.append(neg_preds)
                    # preds.append(pred)
                    # results.append(result)
                    # readed_data.append({"norm": norm_type, "num_hop": num_hop})
                except:
                    continue 

        return pos_preds_list, neg_preds_list, results, model_names

    
    elif mode_name == "homo":
        model_names, pos_preds_list, neg_preds_list, results = [], [], [], []
        norm_types = ["D2AD2", "A", "D-1A"]
        num_hops = [0, 1, 2, 3, 4]
        is_feature = 0, 1
        dis_func_names = ["l2", "cos", "jaccard", "kmeans"]
        for norm_type in norm_types:
            for num_hop in num_hops:
                for dis_func_name in dis_func_names:
                    # homo_{args.num_hops}_{args.adj_norm}_{args.dis_func_name}_{args.is_feature}
                    is_feature = 0 if dis_func_name == "kmeans" else 1
                    pred_path = f"homo_{num_hop}_{norm_type}_{dis_func_name}_{is_feature}_{args.is_old_neg}.txt"
                    result_path = f"homo_result_{num_hop}_{norm_type}_{dis_func_name}_{is_feature}_{args.is_old_neg}.txt"
                    try:
                        if f"homo_{norm_type}_{num_hop}_{dis_func_name}" not in remains:
                            continue

                        with open(f"{data_path}/{pred_path}", 'rb') as f:
                            preds = pickle.load(f)
                        with open(f"{data_path}/{result_path}", 'rb') as f:
                            result = pickle.load(f)
                        pos_preds, neg_preds = preds['pos'], preds['neg']
                        model_names.append(result[f"homo_{norm_type}_{num_hop}_{dis_func_name}"])
                        pos_preds_list.append(pos_preds)
                        neg_preds_list.append(neg_preds)
                    except:
                        continue

        return pos_preds_list, neg_preds_list, results, model_names
        # return preds, results, readed_data   

def read_results():
    pass


def generate_remains():
    # for some heurtisc, there are too many variants,
    tradic_remain_list = [f"tradic_{i}_{i}" for i in range(3)]
    motif_remain_list = [f"motif_{i}" for i in range(7, 9)]

    path_remain_list = []
    norm_types = ["D2AD2", "A"]
    num_hops = [0, 2]
    for norm_type in norm_types:
        for num_hop in num_hops:
            path_remain_list.append(f"path_{num_hop}_{norm_type}")
        
    homo_remain_list = []

    norm_types = ["D2AD2", "A"]
    num_hops = [0, 2]
    dis_func_names = ["l2", "kmeans"]
    for norm_type in norm_types:
        for num_hop in num_hops:
            for dis_func_name in dis_func_names:
                homo_remain_list.append(f"homo_{num_hop}_{norm_type}_{dis_func_name}")
    

    remain_list = tradic_remain_list + motif_remain_list + path_remain_list + homo_remain_list

    return remain_list

def load_results(args, data_name, model_name, prefix=None):
    data_dict = {"Citeseer": "citeseer",  "Cora": "cora",  "Pubmed": "pubmed", 
        "ogbl-citation2": "citation2",  "ogbl-collab": "collab",  "ogbl-ddi": "ddi",  "ogbl-ppa": "ppa"}
    if data_name in data_dict.keys():
        data_name = data_dict[data_name]
    models = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    datas = ["citeseer", "cora", "pubmed", "citation2", "collab", "ddi", "ppa"]
    path = f"/egr/research-dselab/shared/benchmark_bindoc_output/existing_setting_ogb" if data_name.find("ogb") != -1 else f"/egr/research-dselab/shared/benchmark_bindoc_output/existing_setting_small"
    path = f"{path}/{data_name}/{model_name}" if prefix == None else f"{path}/{data_name}/{model_name}_{prefix}"
    file_names = get_file_names(path)
    
    file_name = f"{path}/{file_names[0]}"
    data = torch.load(file_name)
    keys = ['pos_valid_score', 'neg_valid_score', 'pos_test_score', 'neg_test_score']
    
    if args.analyze_mode == "valid":
        pos_preds, neg_preds = data['pos_valid_score'], data['neg_valid_score']
    elif args.analyze_mode == "test":
        pos_preds, neg_preds = data['pos_test_score'], data['neg_test_score']

    if args.is_old_neg == 0:
        num_edges = args.num_pos_test_edge if args.analyze_mode == "valid" else args.num_pos_val_edge
        neg_preds = torch.reshape(neg_preds, [num_edges, -1])
        results = get_metric_score(pos_preds, neg_preds)
    else:
        pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
        results = get_metric_score_origin(pos_preds, neg_preds)
    
    # import ipdb; ipdb.set_trace()
    # print()
    
    return pos_preds, neg_preds, results

def load_description():
    pass


def list_folders_in_directory(directory_path):
    folders = [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]
    return folders

def load_ablation(args, data_name, name="seal_drnl", selected_metric="Hits@100", prefix=None):
    assert args.is_old_neg == 1, "Currently, we do not have new results"
    pos_preds, neg_preds, _, _ = load_ablation_inner(args, data_name, is_std=False, name=name, selected_metric=selected_metric, prefix=None)
    _, _, results, ranks = load_ablation_inner(args, data_name, is_std=True, name=name, selected_metric=selected_metric, prefix=None)
    # pos_pred, neg_pred = pos_preds[0], neg_preds[0]
    # two key: feat, struct
    # inner key: different models
    preds_dict, ranks_dict, results_dict = {}, {}, {}
    
    for ablation_name in pos_preds.keys():
        for model_name in pos_preds[ablation_name].keys():
            pos_pred, neg_pred = pos_preds[ablation_name][model_name][0], neg_preds[ablation_name][model_name][0]
            # import ipdb; ipdb.set_trace()
            rank, result = ranks[ablation_name][model_name], results[ablation_name][model_name]
            if isinstance(pos_pred, torch.Tensor):
                pos_pred, neg_pred = pos_pred.cpu().numpy(), neg_pred.cpu().numpy()
            preds_dict[f"{model_name}_{ablation_name}"] = [pos_pred, neg_pred]
            ranks_dict[f"{model_name}_{ablation_name}"] = rank
            results_dict[f"{model_name}_{ablation_name}"] = result
    
    return preds_dict, ranks_dict, results_dict

def load_ablation_inner(args, data_name, is_std, name="seal_drnl", selected_metric="Hits@100", prefix=None):
    # ablation study 里面文件的名字： featno表示没有feature: 所以就是only structure, featonlyfeat: only node feature
    # 然后ablation study我跑了seal和zero one, 这两个差不多了
    
    # ablation aims to 
    # seal backbone aims to find 
    data_dict = {"Citeseer": "citeseer",  "Cora": "cora",  "Pubmed": "pubmed", 
        "ogbl-citation2": "citation2",  "ogbl-collab": "collab",  "ogbl-ddi": "ddi",  "ogbl-ppa": "ppa"}
    if data_name in data_dict.keys():
        data_name = data_dict[data_name]
    
    analysis_dict = {"ablation_study": ["featonlyfeat", "featno"], "seal_backbone": ["nodefeat+structfeat"]}
    path = "/egr/research-dselab/shared/juanhui"
    direct_path = "ablation_study"
    # , "seal_backbone"
    direct_path = f"{path}/{direct_path}"
        
    candidate_model_names = ["GCN", "GIN", "SAGE"]
    candidate_dist_names = ["drnl", "zo"]
    candidate_feature_names = ["featonlyfeat", "featno"]

    exist_data_names = list_folders_in_directory(direct_path)
    assert data_name in exist_data_names, f"{data_name} is not in {exist_data_names}"
    direct_path = f"{direct_path}/{data_name}/{name}"
    
    model_names = list_folders_in_directory(direct_path)
    
    pos_preds, neg_preds, results, ranks = {}, {}, {}, {}
    
    file_names = get_file_names(direct_path)
    # import ipdb; ipdb.set_trace()
    for candidate_feature_name in candidate_feature_names:
        pos_preds[candidate_feature_name] = defaultdict(list)
        neg_preds[candidate_feature_name] = defaultdict(list)
        results[candidate_feature_name] = defaultdict(list)
        ranks[candidate_feature_name] = defaultdict(list)
        
        candidate_file_names = [file_name for file_name in file_names if file_name.find(candidate_feature_name) != -1]
        # import ipdb; ipdb.set_trace()
        for candidate_dist_name in candidate_dist_names:
            candidate_names = [file_name for file_name in candidate_file_names if file_name.find(candidate_dist_name) != -1]
            for candidate_model_name in candidate_model_names:
                candidate_names = [file_name for file_name in candidate_names if file_name.find(candidate_model_name) != -1]
                if len(candidate_names) == 0: continue
                candidate_names = [candidate_names[0]] if not is_std else candidate_names
                key_name = f"{candidate_dist_name}_{candidate_model_name}"
                
                for file_name in candidate_names:
                    file_name = f"{direct_path}/{file_name}"
                    data = torch.load(file_name)
                    keys = ['pos_valid_score', 'neg_valid_score', 'pos_test_score', 'neg_test_score']
                    
                    if args.analyze_mode == "valid":
                        pos_pred, neg_pred = data['pos_valid_score'], data['neg_valid_score']
                    elif args.analyze_mode == "test":
                        pos_pred, neg_pred = data['pos_test_score'], data['neg_test_score']
                    
                    pos_preds[candidate_feature_name][key_name].append(pos_pred.cpu().numpy())
                    neg_preds[candidate_feature_name][key_name].append(neg_pred.cpu().numpy())
                    # Use torch.argsort() to get the indices that would sort the tensor
                    preds = torch.cat([pos_pred, neg_pred], dim=0)
                    sorted_indices = torch.argsort(preds, descending=True)
                    # Get the ranks by assigning the sorted indices
                    rank = torch.empty_like(sorted_indices)
                    rank[sorted_indices] = torch.arange(1, len(rank) + 1)
                    rank = rank.cpu().numpy()
                    ranks[candidate_feature_name][key_name].append(rank)
                    
                    if args.is_old_neg == 0:
                        num_edges = args.num_pos_test_edge if args.analyze_mode == "valid" else args.num_pos_val_edge
                        neg_pred = torch.reshape(neg_pred, [num_edges, -1])
                        result = get_metric_score(pos_pred, neg_pred)
                    else:
                        pos_pred, neg_pred = torch.flatten(pos_pred), torch.flatten(neg_pred)
                        result = get_metric_score_origin(pos_pred, neg_pred)
                    
                    results[candidate_feature_name][key_name].append(result[selected_metric])
    
    for candidate_feature_name in ranks.keys():
        for key_name in ranks[candidate_feature_name].keys():
            # import ipdb; ipdb.set_trace()
            rank = torch.tensor(np.stack(ranks[candidate_feature_name][key_name]))
            rank = torch.mean(rank.to(torch.float32), dim=0, keepdim=False)
            sorted_indices = torch.argsort(rank, descending=False)
            # Get the ranks by assigning the sorted indices
            rank = torch.empty_like(sorted_indices)
            rank[sorted_indices] = torch.arange(1, len(rank) + 1)
            ranks[candidate_feature_name][key_name] = rank.cpu()
            # ranks[candidate_feature_name][key_name] = np.mean(ranks[candidate_feature_name][key_name], axis=0)
            result = results[candidate_feature_name][key_name]
            # import ipdb; ipdb.set_trace()
            if is_std:
                results[candidate_feature_name][key_name] = (np.mean(result), np.std(result))
            else:
                results[candidate_feature_name][key_name] = np.mean(result)            
    
    replaced_dict = {'featonlyfeat': "feat", 'featno': 'struct'}
    keys = copy.deepcopy(list(results.keys()))
    for key in keys:
        results[replaced_dict[key]] = results.pop(key)
        ranks[replaced_dict[key]] = ranks.pop(key)
        pos_preds[replaced_dict[key]] = pos_preds.pop(key)
        neg_preds[replaced_dict[key]] = neg_preds.pop(key)

    return pos_preds, neg_preds, results, ranks            
            
def load_seal_analysis(args, dataset_name, is_std=False, prefix=None):
    pos_preds, neg_preds, results, ranks = load_seal(args, dataset_name, is_std=is_std, prefix=None)

    model_name = "GCN"
    pos_preds = pos_preds[model_name]
    neg_preds = neg_preds[model_name]
    ranks = ranks[model_name]
    results = results[model_name]
    
    return pos_preds, neg_preds, results, ranks
    
def load_seal(args, data_name, is_std, name="nodefeat+structfeat", selected_metric="Hits@100", prefix=None):
    # ablation study 里面文件的名字： featno表示没有feature: 所以就是only structure, featonlyfeat: only node feature
    # 然后ablation study我跑了seal和zero one, 这两个差不多了
    
    # ablation aims to 
    # seal backbone aims to find 
    data_dict = {"Citeseer": "citeseer",  "Cora": "cora",  "Pubmed": "pubmed", 
        "ogbl-citation2": "citation2",  "ogbl-collab": "collab",  "ogbl-ddi": "ddi",  "ogbl-ppa": "ppa"}
    if data_name in data_dict.keys():
        data_name = data_dict[data_name]
    
    analysis_dict = {"ablation_study": ["featonlyfeat", "featno"], "seal_backbone": ["nodefeat+structfeat"]}
    path = "/egr/research-dselab/shared/juanhui"
    direct_path = "seal_backbone"
    # , "seal_backbone"
    direct_path = f"{path}/{direct_path}/{name}"
        
    candidate_model_names = ["GCN", "GIN", "SAGE"]
    candidate_dist_names = ["drnl", "zo"]
    candidate_feature_names = ["featyes"]
    # feature is not controlled here
    
    exist_data_names = list_folders_in_directory(direct_path)
    assert data_name in exist_data_names, f"{data_name} is not in {exist_data_names}"
    direct_path = f"{direct_path}/{data_name}"
    
    model_names = list_folders_in_directory(direct_path)
    
    pos_preds, neg_preds, results, ranks = {}, {}, {}, {}
    
    file_names = get_file_names(direct_path)
    # import ipdb; ipdb.set_trace()
    for candidate_model_name in candidate_model_names:
        pos_preds[candidate_model_name] = defaultdict(list)
        neg_preds[candidate_model_name] = defaultdict(list)
        results[candidate_model_name] = defaultdict(list)
        ranks[candidate_model_name] = defaultdict(list)
        
        candidate_file_names = [file_name for file_name in file_names if file_name.find(candidate_model_name) != -1]
        # import ipdb; ipdb.set_trace()
        for candidate_dist_name in candidate_dist_names:
            candidate_names = [file_name for file_name in candidate_file_names if file_name.find(candidate_dist_name) != -1]
            if len(candidate_names) == 0: continue
            candidate_names = [candidate_names[0]] if not is_std else candidate_names
            key_name = f"{candidate_dist_name}"
            
            for file_name in candidate_names:
                file_name = f"{direct_path}/{file_name}"
                data = torch.load(file_name)
                keys = ['pos_valid_score', 'neg_valid_score', 'pos_test_score', 'neg_test_score']
                
                if args.analyze_mode == "valid":
                    pos_pred, neg_pred = data['pos_valid_score'], data['neg_valid_score']
                elif args.analyze_mode == "test":
                    pos_pred, neg_pred = data['pos_test_score'], data['neg_test_score']
                
                pos_preds[candidate_model_name][key_name].append(pos_pred.cpu().numpy())
                neg_preds[candidate_model_name][key_name].append(neg_pred.cpu().numpy())
                # Use torch.argsort() to get the indices that would sort the tensor
                preds = torch.cat([pos_pred, neg_pred], dim=0)
                sorted_indices = torch.argsort(preds, descending=True)
                # Get the ranks by assigning the sorted indices
                rank = torch.empty_like(sorted_indices)
                rank[sorted_indices] = torch.arange(1, len(rank) + 1)
                rank = rank.cpu().numpy()
                ranks[candidate_model_name][key_name].append(rank)
                    
                if args.is_old_neg == 0:
                    num_edges = args.num_pos_test_edge if args.analyze_mode == "valid" else args.num_pos_val_edge
                    neg_pred = torch.reshape(neg_pred, [num_edges, -1])
                    result = get_metric_score(pos_pred, neg_pred)
                else:
                    pos_pred, neg_pred = torch.flatten(pos_pred), torch.flatten(neg_pred)
                    result = get_metric_score_origin(pos_pred, neg_pred)
                
                results[candidate_model_name][key_name].append(result[selected_metric])
    
    for candidate_model_name in ranks.keys():
        for key_name in ranks[candidate_model_name].keys():
            # import ipdb; ipdb.set_trace()
            rank = torch.tensor(np.stack(ranks[candidate_model_name][key_name]))
            rank = torch.mean(rank.to(torch.float32), dim=0, keepdim=False)
            sorted_indices = torch.argsort(rank, descending=False)
            # Get the ranks by assigning the sorted indices
            rank = torch.empty_like(sorted_indices)
            rank[sorted_indices] = torch.arange(1, len(rank) + 1)
            ranks[candidate_model_name][key_name] = rank.cpu()
            # ranks[candidate_feature_name][key_name] = np.mean(ranks[candidate_feature_name][key_name], axis=0)
            result = results[candidate_model_name][key_name]
            # import ipdb; ipdb.set_trace()
            if is_std:
                results[candidate_model_name][key_name] = (np.mean(result), np.std(result))
            else:
                results[candidate_model_name][key_name] = np.mean(result)            

    return pos_preds, neg_preds, results, ranks            
    

def check_key(args, data_name, model_name, prefix=None):
    # current version only support the original setting
    # The rank can be average on different seeds
    data_dict = {"Citeseer": "citeseer",  "Cora": "cora",  "Pubmed": "pubmed", 
        "ogbl-citation2": "citation2",  "ogbl-collab": "collab",  "ogbl-ddi": "ddi",  "ogbl-ppa": "ppa"}
    path = f"/egr/research-dselab/shared/benchmark_bindoc_output/existing_setting_ogb" if data_name.find("ogb") != -1 else f"/egr/research-dselab/shared/benchmark_bindoc_output/existing_setting_small"
    if data_name in data_dict.keys():
        data_name = data_dict[data_name]
    path = f"{path}/{data_name}/{model_name}" if prefix == None else f"{path}/{data_name}/{model_name}_{prefix}"
    try:
        file_names = get_file_names(path)
        file_name = file_names[0]
        file_name = f"{path}/{file_name}"
        data = torch.load(file_name)
    except:
        return
    if "state_dict_model" in data.keys():
        # import ipdb; ipdb.set_trace()
        print(f"dataset: {data_name}, model: {model_name} keys: {data.keys()}")


def load_results_with_multiseed(args, data_name, model_name, is_single, is_std, prefix=None):
    # current version only support the original setting
    # The rank can be average on different seeds
    data_dict = {"Citeseer": "citeseer",  "Cora": "cora",  "Pubmed": "pubmed", 
        "ogbl-citation2": "citation2",  "ogbl-collab": "collab",  "ogbl-ddi": "ddi",  "ogbl-ppa": "ppa"}
    if args.is_old_neg:
        # /egr/research-dselab/shared/benchmark_bindoc_output/
        if model_name in ["gae", "neognn"] and data_name in ["Cora", "Citeseer", "Pubmed"]:
            path = f"/egr/research-dselab/shared/juanhui/old_results/existing_setting_ogb" if data_name.find("ogb") != -1 else f"/egr/research-dselab/shared/juanhui/old_results/existing_setting_small"
        else:    
            path = f"/egr/research-dselab/shared/benchmark_bindoc_output/existing_setting_ogb" if data_name.find("ogb") != -1 else f"/egr/research-dselab/shared/benchmark_bindoc_output/existing_setting_small"
    else:
        path = f"/egr/research-dselab/shared/juanhui/old_results/heart_ogb" if data_name.find("ogb") != -1 else f"/egr/research-dselab/shared/juanhui/old_results/heart_small"
        
    if data_name in data_dict.keys():
        data_name = data_dict[data_name]
    models = ["buddy", "gae", "gat", "gcn", "MF", "mlp", "n2v", "nbfnet", "ncn", "ncnc", "neognn", "peg", "sage", "seal"]
    datas = ["citeseer", "cora", "pubmed", "citation2", "collab", "ddi", "ppa"]
    if model_name == "sage" and data_name in ["citation2", "collab", "ddi", "ppa"]:
        model_name = "SAGE"

    path = f"{path}/{data_name}/{model_name}" if prefix == None else f"{path}/{data_name}/{model_name}_{prefix}"
    file_names = get_file_names(path)
    ranks, results = [], defaultdict(list)
    pos_preds_list, neg_preds_list = [], []
    
    if is_single: file_names = [file_names[0]]
    
    for file_name in file_names:
        file_name = f"{path}/{file_name}"
        data = torch.load(file_name)
        keys = ['pos_valid_score', 'neg_valid_score', 'pos_test_score', 'neg_test_score']
        
        # print(data.keys())
        if args.analyze_mode == "valid":
            pos_preds, neg_preds = data['pos_valid_score'], data['neg_valid_score']
        elif args.analyze_mode == "test":
            pos_preds, neg_preds = data['pos_test_score'], data['neg_test_score']
        
        preds = torch.cat([pos_preds, neg_preds], dim=0)
        # Use torch.argsort() to get the indices that would sort the tensor
        sorted_indices = torch.argsort(preds, descending=True)
        # Get the ranks by assigning the sorted indices
        rank = torch.empty_like(sorted_indices)
        rank[sorted_indices] = torch.arange(1, len(rank) + 1)
        ranks.append(rank)

        if args.is_old_neg == 0:
            num_edges = args.num_pos_test_edge if args.analyze_mode == "valid" else args.num_pos_val_edge
            neg_preds = torch.reshape(neg_preds, [num_edges, -1])
            result = get_metric_score(pos_preds, neg_preds)
        else:
            pos_preds, neg_preds = torch.flatten(pos_preds), torch.flatten(neg_preds)
            result = get_metric_score_origin(pos_preds, neg_preds)
        for key in result.keys():
            results[key].append(result[key]) 

        pos_preds_list.append(pos_preds.cpu().numpy())
        neg_preds_list.append(neg_preds.cpu().numpy())
    # get the average performance 
    # ranks = torch.mean(data.float(), dim=0, keepdim=False)

    ranks = torch.mean(torch.stack(ranks).to(torch.float32), dim=0, keepdim=False)
    sorted_indices = torch.argsort(ranks, descending=False)
    # Get the ranks by assigning the sorted indices
    rank = torch.empty_like(sorted_indices)
    rank[sorted_indices] = torch.arange(1, len(ranks) + 1)
    rank = rank.cpu().numpy()
    
    final_results = {}
    for key in results.keys():
        if is_std:
            final_results[key] = (np.mean(results[key]), np.std(results[key]))
        else:
            final_results[key] = np.mean(results[key])            

    # import ipdb; ipdb.set_trace()
    
    if is_single:
        return pos_preds_list[0], neg_preds_list[0], rank, final_results
    else:
        return pos_preds_list, neg_preds_list, rank, final_results




def get_file_names(folder_path):
    file_names = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_names.append(file)
    return file_names


def generate_ranks(pos_preds_list, neg_preds_list):
    rank_list = []
    for pos_preds, neg_preds in zip(pos_preds_list, neg_preds_list):
        pos_preds, neg_preds = torch.tensor(pos_preds), torch.tensor(neg_preds)
        preds = torch.cat([pos_preds, neg_preds], dim=0)
        # Use torch.argsort() to get the indices that would sort the tensor
        sorted_indices = torch.argsort(preds, descending=True)
        # Get the ranks by assigning the sorted indices
        rank = torch.empty_like(sorted_indices)
        rank[sorted_indices] = torch.arange(1, rank.shape[0] + 1)
        rank = rank.cpu().numpy()
        rank_list.append(rank)

    return rank_list


def generate_rank_single(pos_preds, neg_preds):
    if isinstance(pos_preds, np.ndarray):
        pos_preds, neg_preds = torch.tensor(pos_preds), torch.tensor(neg_preds)
    device = pos_preds.device
    if len(neg_preds.shape) == 1:
        preds = torch.cat([pos_preds, neg_preds], dim=0)
        # Use torch.argsort() to get the indices that would sort the tensor
        sorted_indices = torch.argsort(preds, descending=True)
        # Get the ranks by assigning the sorted indices
        rank = torch.empty_like(sorted_indices)
        rank[sorted_indices] = torch.arange(1, rank.shape[0] + 1).to(device)
        rank = rank.cpu().numpy()
    else:
        preds = torch.cat([pos_preds, neg_preds], dim=-1)
        sorted_indices = torch.argsort(preds, descending=True, dim=-1)

        for pos_pred, neg_pred in zip(pos_preds, neg_preds):
            # Use torch.argsort() to get the indices that would sort the tensor
            # Get the ranks by assigning the sorted indices
            rank.append(torch.empty_like(sorted_indices))
            rank[-1][sorted_indices] = torch.arange(1, rank[-1].shape[0] + 1).to(device)
            rank[-1] = rank[-1].cpu().numpy()

    return rank


def get_rank_single(args, preds): 
    # import ipdb; ipdb.set_trace()
    if isinstance(preds, np.ndarray):
        preds = torch.tensor(preds)
    device = preds.device
    
    # Use torch.argsort() to get the indices that would sort the tensor
    if args.is_old_neg:
        sorted_indices = torch.argsort(preds,  descending=True)
        # Get the ranks by assigning the sorted indices
        rank = torch.empty_like(sorted_indices)
        rank[sorted_indices] = torch.arange(1, rank.shape[0] + 1).to(device)
    else:
        
        sorted_indices = torch.argsort(preds, dim=-1, descending=True)
        # Get the ranks by assigning the sorted indices
        rank = torch.argsort(sorted_indices, dim=-1, descending=False)
    
    rank = rank.cpu().numpy()
    # import ipdb; ipdb.set_trace()
    return rank

    # a = torch.tensor([7, 9, 1, 5])
    # sorted_indices = torch.argsort(a, dim=-1, descending=True)
    # # Get the ranks by assigning the sorted indices
    # rank = torch.argsort(sorted_indices, dim=-1, descending=False)
        

def get_rank_new(y_pred_pos, y_pred_neg, is_old_neg, K):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''
    if not isinstance(y_pred_pos, torch.Tensor):    
        y_pred_pos, y_pred_neg = torch.tensor(y_pred_pos), torch.tensor(y_pred_neg)
    
    if is_old_neg == 0:
        y_pred_pos = y_pred_pos.view(-1, 1)
        # optimistic rank: "how many negatives have a larger score than the positive?"
        # ~> the positive is ranked first among those with equal score
        optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        # pessimistic rank: "how many negatives have at least the positive score?"
        # ~> the positive is ranked last among those with equal score
        pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hit_result = torch.sum((ranking_list <= K).to(torch.float)).item() / len(y_pred_pos)
        mrr_result = torch.mean(1./ranking_list.to(torch.float))
        correct_index = mask_to_index(ranking_list <= K)
        # import ipdb; ipdb.set_trace()
        return hit_result, mrr_result, correct_index
    else:
        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        hit_result = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
        mrr_result = None
        correct_index = mask_to_index(y_pred_pos > kth_score_in_negative_edges)
        return hit_result, mrr_result, correct_index
        
    


'''
def get_rank_new_detail(y_pred_pos, y_pred_neg, is_old_neg, K):
    y_pred_pos, y_pred_neg = torch.tensor(y_pred_pos), torch.tensor(y_pred_neg)
    
    if is_old_neg == 0:
        y_pred_pos = y_pred_pos.view(-1, 1)
        # optimistic rank: "how many negatives have a larger score than the positive?"
        # ~> the positive is ranked first among those with equal score
        optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        # pessimistic rank: "how many negatives have at least the positive score?"
        # ~> the positive is ranked last among those with equal score
        pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hit_result = torch.sum((ranking_list <= K).to(torch.float)).item() / len(y_pred_pos)
        mrr_result = torch.mean(1./ranking_list.to(torch.float))
        correct_index = (ranking_list <= K)
        return hit_result, mrr_result, correct_index, ranking_list
    else:
        import ipdb; ipdb.set_trace()
        optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=0)
        pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=0)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        
        kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
        hit_result = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
        mrr_result = None
        correct_index = (y_pred_pos > kth_score_in_negative_edges)
        return hit_result, mrr_result, correct_index
'''