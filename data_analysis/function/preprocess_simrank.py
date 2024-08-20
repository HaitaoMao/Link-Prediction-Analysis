from data_analysis.function.homophily import *
from data_analysis.function.functional import *
from data_analysis.plot_exp import *
from data_analysis.generate_data import load_data
from data_analysis.function.read_results import generate_rank_single
import scipy.sparse as sp
from evaluation_new import * 
import torch
import os
import pathlib
from data_analysis.function.F1 import F1, F1_new, equal_split, count_bin, generate_split_masks, seperate_accuracy
from data_analysis.function.read_results import *
from data_analysis.function.heuristics import PPR_new
from torch_geometric.utils import mask_to_index, index_to_mask, remove_self_loops, to_undirected, coalesce
from tqdm import tqdm

def preprocess_simrank(args, device):
    args.is_generate_train = 1   # whether preprocess on just training set or entire dataset
    args.is_old_neg = 1          # whether use the new heart negative sampling method
    args.is_flatten = 0  # if use the heart, whether remove the redudant validation and test edge 
    args.is_remove_redudant = 1  # if use the heart, whether remove the redudant validation and test edge 

    args.is_norm = 0
    args.is_feature_norm = 0
    
    # read_results(args.dataset_name)
    # import ipdb; ipdb.set_trace()
    num_nodes_dict  = {"Cora": 2708, "Citeseer": 3327, "Pubmed": 18717 , "ogbl-collab": 235868 , "ogbl-ddi": 4267 , "ogbl-ppa": 576289 , "ogbl-citation2": 2927963}
    num_nodes = num_nodes_dict[args.dataset_name]
    # TODO: citation2 is still not clear at present
    Ks = {"Cora": 200, "Citeseer": 200, "Pubmed": 200, "ogbl-collab": 50, "ogbl-citation2": 10, "ogbl-ddi": 400, "ogbl-ppa": 100}
    # Ks = {"Cora": 200, "Citeseer": 200, "Pubmed": 200, "ogbl-collab": 50, "ogbl-citation2": 10, "ogbl-ddi": 400, "ogbl-ppa": 100}
    K = Ks[args.dataset_name]
    
    
    # with open(f"/egr/research-dselab/haitaoma/LinkPrediction/subgraph-sketching/src/intermedia_result/simrank_adjs/Cora.txt", "rb") as f:
    #     data = pickle.load(f)
    # import ipdb; ipdb.set_trace()
    read_results(args.dataset_name, num_nodes, K)
    exit()


    args.analyze_mode = "valid"  # "whole" "valid" "test"
    _, _, valid_pos_links, valid_neg_links, _ = load_data(args, device)
    if not args.is_old_neg:
        args.neg_per_valid = valid_neg_links.shape[1]
        valid_neg_links = torch.reshape(valid_neg_links, [-1, 2])
    args.analyze_mode = "test"  # "whole" "valid" "test"
    dataset, known_links, test_pos_links, test_neg_links, path = load_data(args, device)
    if not args.is_old_neg:
        args.neg_per_test = test_neg_links.shape[1]
        test_neg_links = torch.reshape(test_neg_links, [-1, 2])
    
    # print(f"{args.dataset_name}: {num_nodes * K}")
    
    # data = dataset.data
    # num_nodes = data.x.shape[0]
    # train_links = data.edge_index.T
    
    # tmp1, tmp2 = (train_links[:, 0] == valid_pos_links[1][0]), (train_links[:, 1] == valid_pos_links[1][1])
    # print((tmp1 & tmp2).sum())
    known_links, _ = remove_self_loops(to_undirected(known_links.T))
    # import ipdb; ipdb.set_trace()
    print("here")
    known_links = coalesce(known_links).T
    print("here")
    
    known_links = known_links.cpu().numpy()
    num_nodes = np.max(known_links) + 1
    
    folder_path = pathlib.Path(f"/egr/research-dselab/haitaoma/LinkPrediction/UISim2020/data/{args.dataset_name}")
    if not folder_path.exists():
        folder_path.mkdir()
    
    # import ipdb; ipdb.set_trace()
    with open(f"{folder_path}/{args.dataset_name}.edge", "w") as f:
        for i in tqdm(range(known_links.shape[0])):
            f.write(f"{known_links[i][0]}\t{known_links[i][1]}\n")

    with open(f"{folder_path}/{args.dataset_name}.node", "w") as f:
        for i in tqdm(range(num_nodes)):
            f.write(f"{int(i)}\n")
    
    new_folder_path = pathlib.Path(f"{folder_path}/result")
    if not new_folder_path.exists():
        new_folder_path.mkdir()
    
    # read_result
    
def read_results(dataset_name, num_nodes, K):
    num_values = num_nodes * K
    folder_path = f"/egr/research-dselab/haitaoma/LinkPrediction/UISim2020/data/{dataset_name}/result"
    file_names = []
    for file in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file)):
            file_names.append(file)
    file_name = file_names[0]
    
    # name = "fs-AP-k-1_Hindeg_DEP5_STP0.000100_H50_E0.000000"
    pattern = r"[-+]?\d*\.\d+|\d+"
    
    idx = 0
    edge_indices, edge_values = np.zeros([num_values, 2], dtype=np.int32), np.zeros([num_values], dtype=np.float64)
    
    with open(f"{folder_path}/{file_name}", 'r') as f:
        # lines = f.readlines()
        # print(len(lines))
        # import ipdb; ipdb.set_trace()
    
        while True:
            line = f.readline()
            # print(line)
            if not line:
                break
            matches = re.findall(pattern, line)
            numbers = [float(match) for match in matches]
             
            if idx == 0:
                row, col, value = int(numbers[1]), int(numbers[2]), numbers[3]
            else:
                row, col, value = int(numbers[0]), int(numbers[1]), numbers[2]
            # print(f"row: {row}, col: {col}, value: {value}")
            if value != 0:
                edge_indices[idx][0], edge_indices[idx][1] = row, col
                edge_values[idx] = value
                idx += 1
            else:
                break
            # print(idx)
        # lines = f.readlines()
    # import ipdb; ipdb.set_trace()
    edge_indices, edge_values = edge_indices[:idx], edge_values[:idx]
    with open(f"/egr/research-dselab/haitaoma/LinkPrediction/subgraph-sketching/src/intermedia_result/simrank_adjs/{dataset_name}.txt", "wb") as f:
        pickle.dump({"index": edge_indices, "value": edge_values}, f)
    # import ipdb; ipdb.set_trace()
    # f = open(f"/egr/research-dselab/haitaoma/LinkPrediction/subgraph-sketching/src/intermedia_result/simrank_preds/{dataset_name}.txt", "rb")
    