from data_analysis.function.read_results import *
import numpy as np
import torch
from torch_geometric.utils import mask_to_index, index_to_mask
import heapq
import pandas as pd
from evaluation_new import evaluate_hits
from ogb.linkproppred import Evaluator


def run_F1(args, device):
    origin_model_names = ["mlp", "gcn"]
    dataset_names = ["Cora"]
    Ks = [1, 3, 10, 100]
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
        num_pos = pos_preds_list[0].shape[0]
        # import ipdb; ipdb.set_trace()
        # print()

        metric_names = ["algorithm1", "algorithm2", "TP", "FP", "FN", "TN"]
        for idx1, (ranks1, result1, model_name1) in enumerate(zip(ranks_list, results, model_names)):
            for idx2, (ranks2, result2, model_name2) in enumerate(zip(ranks_list, results, model_names)):
                if idx1 == idx2:
                    continue
                
                xname = f"{model_name1}_{model_name2}"
                # the name in the x axis
                name = f"{dataset_name}_old" if args.is_old_neg else f"{dataset_name}"
                for K in Ks:
                    compare_results = F1(ranks1, ranks2, num_pos, K=K)
                    result_name = f"output_analyze/F1_results/{args.dataset_name}_{K}.xlsx"
                    try:
                        results_record = pd.read_excel(result_name)
                        # import ipdb; ipdb.set_trace()
                        # print()
                    except:
                        results_record =pd.DataFrame(columns=metric_names)
                        # results_record.set_index("algorithm", inplace=True) 
                            
                        # find_results = resut
                    # import ipdb; ipdb.set_trace()
                    index = (results_record['algorithm1'] == model_name1) & (results_record['algorithm2'] == model_name2)
                    
                    if len(index) == 0:
                        tmp_dict = {"algorithm1": [model_name1], "algorithm2": [model_name2]}                
                        for key in compare_results.keys():
                            tmp_dict[key] = [compare_results[key]]
                        new_row = pd.DataFrame(tmp_dict)

                        results_record = pd.concat([results_record, new_row], ignore_index=True)
                    else:
                        for key in compare_results.keys():
                            results_record[index][key] = compare_results[key]
                        
                    results_record.to_excel(result_name, index=False)

#  pos_preds1, neg_preds1, pos_preds2, neg_preds2, 
def F1(rank1, rank2, num_pos, K=10):
    num_edge = rank1.shape[0]
    num_neg = num_edge - num_pos

    result1 = np.split(rank1, [num_pos])
    
    pos_rank1, neg_rank1 = result1[0], result1[1]
    result2 = np.split(rank2, [num_pos])
    pos_rank2, neg_rank2 = result2[0], result2[1]
    
    pred_correct_index1, pred_correct_mask1 = correct_id(pos_rank1, neg_rank1, K)
    pred_correct_index2, pred_correct_mask2 = correct_id(pos_rank2, neg_rank2, K)

    TP_mask = pred_correct_mask1 & pred_correct_mask2
    FP_mask = ~pred_correct_mask1 & pred_correct_mask2
    FN_mask = pred_correct_mask1 & ~pred_correct_mask2
    TN_mask = ~pred_correct_mask1 & ~pred_correct_mask2
    # import ipdb; ipdb.set_trace()

    results = {}
    results["TP"] = TP_mask.sum() / num_pos
    results["FP"] = FP_mask.sum() / num_pos
    results["FN"] = FN_mask.sum() / num_pos
    results["TN"] = TN_mask.sum() / num_pos
    
    return results


def F1_new(pred_correct_index1, pred_correct_index2, num_pos):
    pred_correct_mask1 = index_to_mask(pred_correct_index1, num_pos)
    pred_correct_mask2 = index_to_mask(pred_correct_index2, num_pos)

    TP_mask = pred_correct_mask1 & pred_correct_mask2
    FP_mask = ~pred_correct_mask1 & pred_correct_mask2
    FN_mask = pred_correct_mask1 & ~pred_correct_mask2
    TN_mask = ~pred_correct_mask1 & ~pred_correct_mask2
    # import ipdb; ipdb.set_trace()

    results = {}
    results["TP"] = TP_mask.sum() / num_pos
    results["FP"] = FP_mask.sum() / num_pos
    results["FN"] = FN_mask.sum() / num_pos
    results["TN"] = TN_mask.sum() / num_pos
    
    return results

    

def correct_id(pos_rank, neg_rank, K):
    # top_ranks, top_indices = torch.topk(neg_rank, K, largest=False, sorted=True)
    # top_indices= np.argpartition(neg_rank, -K)[-K:]
    smallest_k_values = heapq.nsmallest(K, neg_rank)
    top_rank = np.max(smallest_k_values)
    # top_rank = top_rank[0]
    pred_correct_mask = pos_rank < top_rank
    pred_correct_index = np.where(pred_correct_mask)[0] 
    
    return pred_correct_index, pred_correct_mask

    
    
def equal_split(pos_preds, K):
    '''
    There are two important rules for seperate values
    1. node with the same values should be in the same bin
    2. each bin should have some values 
    '''
    if isinstance(pos_preds, torch.Tensor):
        pos_preds = pos_preds.cpu().numpy()
    # values = np.concatenate([pos_preds, neg_preds])
    values = pos_preds
    split_values = [min(values)]
    # split values will record the left bound for each bin, for the last one, it will goes to infinite
    # the set will be close interval on left but open interval on right
    num_values = []
    # the number of instance in each bin, it will not be generally equa
    
    prev_move_idx, move_idx = -1, -1
    num = values.shape[0]
    num_per_bin = num // K
    values = list(np.sort(values))
    too_long_flag = True
    
    # import ipdb; ipdb.set_trace()
    for i in range(K-1):
        # print(i)
        # print(split_values)
        move_idx = prev_move_idx + num_per_bin
        if move_idx >= num - 1:
            break
        value = values[move_idx]
        
        # move to new value 
        
        if too_long_flag and values.count(value) >= num_per_bin:
            # one value wiith too long
            count = values.count(value)
            # too_long_flag = False
            record_move_idx = move_idx
            while move_idx > 0 and values[move_idx] == value:
                move_idx -= 1
            # push back
            previous_value = values[move_idx]
            if previous_value <= split_values[-1] or move_idx - prev_move_idx + 1 <= (num // (K * 10)):
                # no value in current bin or too few values in current bin
                move_idx += 1
                while move_idx < num and values[move_idx] <= value:
                    move_idx += 1
                prev_move_idx = move_idx
                # print(value)
                # print(values[move_idx])
                split_values.append(values[move_idx])
            else:
                move_idx += 1
                prev_move_idx = move_idx
                split_values.append(values[move_idx])
            # print(num_per_bin)
            prev_num_per_bin = num_per_bin
            num_per_bin = (num-move_idx) // (K - i)
            # please do not be too short
            if num_per_bin <= num // (K * 10):
                too_long_flag = False
                num_per_bin = prev_num_per_bin
            # print(num_per_bin)
            continue
            
            # return to the last closest value
            
            # if value == min(values):
            #     too_long_flag = False
            #     num_per_bin = (num-count) // (K - 1)
            # else:
            #     too_long_flag = False
            #     while move_idx >= 0 and values[move_idx] == value:
            #         move_idx -= 1
            #     # return to the last closest value
            #     num_per_bin = (num-np.sum(num_values)) // (K - i - 1)
                # continue

        while move_idx < num and values[move_idx] == value:
            move_idx += 1
        # import ipdb; ipdb.set_trace()
        try:
            split_values.append(values[move_idx])
        except:
            split_values.append(value)
        num_value = move_idx - prev_move_idx
        prev_move_idx = move_idx
        # in the last piece, we check if there is still remain values
    
    # split_values.append(values[-1])   
    # if move_idx > num:
    #     num_values.append(num - prev_move_idx)
    # else:
    #     num_values.append(num - move_idx)
     
    
    # import ipdb; ipdb.set_trace()    
    
    # add count 
    # if values[1] == 
    # import ipdb; ipdb.set_trace()
    split_values, num_values = count_bin(values, split_values)
            
    return split_values, num_values
 
 
def count_bin(values, split_values):
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    elif isinstance(values, list):
        values = np.array(values)
    if len(values.shape) != 1:
        values = values.reshape(-1)

    values = list(np.sort(values))
    num = len(values)
    # import ipdb; ipdb.set_trace()
    num_values = []
    prev_idx, cur_idx = 0, 0
    for i in range(1, len(split_values)):
        split_value = split_values[i]
        while cur_idx < num and values[cur_idx] < split_value:
            cur_idx += 1
        # import ipdb; ipdb.set_trace()
        num_values.append(cur_idx - prev_idx)
        prev_idx = cur_idx
        
    # if num - prev_idx == 0:
    #     split_values.pop()
    # else:
    num_values.append(num - prev_idx)
    
    return split_values, num_values
            
def generate_split_masks(pos_tradic_preds, split_values):
    masks = []
    for i in range(len(split_values)-1):
        mask = (pos_tradic_preds >= split_values[i]) & (pos_tradic_preds < split_values[i+1])
        masks.append(mask)
    mask = (pos_tradic_preds >= split_values[-1])
    masks.append(mask)
    if isinstance(mask, torch.Tensor):
        masks = [mask.cpu().numpy() for mask in masks]    
    return masks


def seperate_accuracy(pos_preds, neg_preds, masks, is_old_neg, K):
    if isinstance(pos_preds, torch.Tensor):
        pos_preds = pos_preds.cpu().numpy()
        neg_preds = neg_preds.cpu().numpy()

    num_pos = pos_preds.shape[0]
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2')
    
    results = []
    if is_old_neg:
        for mask in masks:
            pos_preds_bin = pos_preds[mask]
            result = evaluate_hits(evaluator_hit, pos_preds_bin, neg_preds, [K])
            result = result[f'Hits@{K}']
            results.append(result)
    else:
        for mask in masks:
            pos_preds_bin = pos_preds[mask]
            neg_preds_bin = neg_preds[mask]
            # import ipdb; ipdb.set_trace()
            result = get_metric_score(pos_preds_bin, neg_preds_bin)
            result = result[f'Hits@{K}']
            results.append(result)    
    
    return results


def seperate_accuracy_old(pos_preds, neg_preds, masks, num_values, K):
    if isinstance(pos_preds, torch.Tensor):
        pos_preds = pos_preds.cpu().numpy()
        neg_preds = neg_preds.cpu().numpy()

    num_pos = pos_preds.shape[0]
    rank = generate_rank_single(pos_preds, neg_preds)
    pos_rank, neg_rank = rank[:num_pos], rank[num_pos:]
    pred_correct_index, pred_correct_mask = correct_id(pos_rank, neg_rank, K)

    import ipdb; ipdb.set_trace()
    
    results = []
    for mask, num_value in zip(masks, num_values):
        correct_pred_mask = mask & pred_correct_mask
        result = correct_pred_mask.sum() / mask.sum()
        results.append(result)
    import ipdb; ipdb.set_trace()
    
    return results
    
    

def equal_split_old(pos_preds, neg_preds, K):
    '''
    There are two important rules for seperate values
    1. node with the same values should be in the same bin
    2. each bin should have some values 
    '''
    if isinstance(pos_preds, torch.Tensor):
        pos_preds = pos_preds.cpu().numpy()
        neg_preds = neg_preds.cpu().numpy()
    values = np.concatenate([pos_preds, neg_preds])
    split_values = [0]
    # split values will record the left bound for each bin, for the last one, it will goes to infinite
    # the set will be close interval on left but open interval on right
    num_values = [0]
    # the number of instance in each bin, it will not be generally equa
    
    prev_move_idx, move_idx = -1, -1
    num = values.shape[0]
    num_per_bin = num // K
    values = np.sort(values)
    
    for i in range(K-1):
        move_idx = prev_move_idx + num_per_bin
        if move_idx >= num - 1:
            break
        value = values[move_idx]
        
        # move to new value 
        while move_idx < num and values[move_idx] == value:
            move_idx += 1
        # import ipdb; ipdb.set_trace()
        try:
            split_values.append(values[move_idx])
        except:
            split_values.append(value)
        num_value = move_idx - prev_move_idx
        num_values.append(num_value)
        prev_move_idx = move_idx
        # in the last piece, we check if there is still remain values
    
    split_values.append(values[-1])   
    if move_idx > num:
        num_values.append(num - prev_move_idx)
    else:
        num_values.append(num - move_idx)
        
    # import ipdb; ipdb.set_trace()    
    return split_values, num_values
