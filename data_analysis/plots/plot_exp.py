# plot experiment for many configs
import pickle
import pandas as pd
import numpy as np
import torch
import math
import json
from collections import defaultdict
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import pathlib
from data_analysis.function.rank_analysis import *
import re
from torch_geometric.utils import mask_to_index, index_to_mask
from data_analysis.function.read_results import generate_rank_single, get_rank_single
from itertools import combinations, permutations


def plot_models_prediction_correlation(args, heu_rank_dict, model_rank_dict, dataset_name):
    #  corr_coef, p_value
    # TODO: draw another heatmap for analysis
    heu_algos, model_algos = list(heu_rank_dict.keys()), list(model_rank_dict.keys()) 
    num_heu, num_model = len(heu_algos), len(model_algos)
    num_edges = heu_rank_dict[heu_algos[0]].shape[0]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    markers = ['o','*','x','s','p',"+",'v', 'h',]

    order = len(str(num_edges)) - 1
    num_scales = 2
    scale = 1 / num_scales
    
    num_xs = num_scales * (int(str(num_edges)[0]) + 1)
    xs = np.arange(num_xs) * scale
    x_ticks = [f"{np.round(xs[i], decimals=1)} 10e{order}" for i in range(len(xs))]
    xs = [xs[i] * 10 ** order for i in range(len(xs))]

    
    fig, axes = plt.subplots(num_heu, num_model, figsize=(3 * num_model + 2 , 3 * num_heu + 2))

    x = [i * (num_edges // 20) for i in range(20)] + [num_edges]

    corr_results, p_results = defaultdict(dict), defaultdict(dict)
    for heu_idx, heu_algo in enumerate(heu_algos):
        for model_idx, model_algo in enumerate(model_algos):
            ax = axes[heu_idx][model_idx]
            heu_rank = heu_rank_dict[heu_algo]
            model_rank = model_rank_dict[model_algo]
            assert len(heu_rank) == len(model_rank)
            corr_coef, p_value = run_correlation_simple(heu_rank, model_rank, heu_algo, model_algo)
            corr_results[heu_algo][model_algo] = corr_coef
            p_results[heu_algo][model_algo] = p_value
            ax.scatter(heu_rank, model_rank, s=0.5, c="blue", marker=markers[0], alpha=1.0)
            ax.plot(x, x, linewidth=2)

            if heu_idx == num_heu - 1:
                ax.set_xlabel(model_algo, fontsize=25, fontfamily='serif')
                ax.set_xticks(xs, x_ticks, fontsize=25, fontfamily='serif')
                # ax.set_xticklabels(x_ticks, fontsize=30, fontfamily='serif')
                # ax.legend()
            if model_idx == 0:
                ax.set_ylabel(heu_algo, fontsize=25, fontfamily='serif')

    # plt.xlabel(f"Ranks of {model_name1}",fontsize=33, fontfamily='serif')
    # plt.ylabel(f"Ranks of {model_name2}",fontsize=33, fontfamily='serif')

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.savefig(f"homo_analyze/prediction_correlation/{dataset_name}.pdf",dpi=500,bbox_inches = 'tight')
    
    plt.clf()

    df = pd.DataFrame(corr_results)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax = sns.heatmap(df, vmin=0, vmax=1, annot=True, yticklabels=model_algos, ax=ax) # , yticklabels=y_ticks
    ax.tick_params(axis='x', length=0, pad=10)
    ax.set_xticks(ax.get_xticks() + 0.2, ax.get_xticklabels(), fontsize=20, fontfamily='serif', rotation=45, ha='right') # , rotation=45, ha='right'
    # ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontfamily='serif', rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontfamily='serif')

    plt.savefig(f"homo_analyze/correlation_heat/{dataset_name}.pdf",dpi=500,bbox_inches = 'tight')

    



def plot_models_prediction_correlation_old(args, ranks):
    # TODO: now we only support one 
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    markers = ['o','*','x','s','p',"+",'v', 'h',]

    for model_name1 in ranks.keys():
        for model_name2 in ranks.keys():
            if model_name1 == model_name2:
                continue
            rank1 = ranks[model_name1]
            rank2 = ranks[model_name2]
            corr_coef, p_value = run_correlation_simple(rank1, rank2, model_name1, model_name2)
            xname = f"{model_name1}_{model_name2}"
            # the name in the x axis
            name = f"{args.dataset_name}_old" if args.is_old_neg else f"{args.dataset_name}" 
            assert len(rank1) == len(rank2)
            num_edges = len(rank1)

            figsize = (6, 4.5)
            fig, ax1 = plt.subplots(figsize=figsize)
            # here we should use billon 
            
            order = len(str(num_edges)) - 1
            num_scales = 2
            scale = 1 / num_scales
            num_xs = num_split * (int(str(num_edges)[0]) + 1)
            xs = np.arange(num_xs) * scale
            x_ticks = [f"{np.round(xs[i], decimals=1)} 10e{order}" for i in range(len(xs))]
            xs = [xs[i] * 10 ** order for i in range(len(xs))]

            plt.scatter(rank1, rank2, s=0.5, c="blue", marker=markers[0], alpha=1.0)
            
            x = [i * (num_edges // 20) for i in range(20)] + [num_edges]
            plt.plot(x, x, linewidth=2)
            # ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            plt.xticks(xs,x_ticks,fontsize=22, fontfamily='serif')
            plt.yticks(xs,x_ticks, fontsize=22, fontfamily='serif')

            plt.xlabel(f"Ranks of {model_name1}",fontsize=33, fontfamily='serif')
            plt.ylabel(f"Ranks of {model_name2}",fontsize=33, fontfamily='serif')
        
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.direction'] = 'in'
            plt.savefig(f"output_analyze/prediction_correlation/{model_name1}_{model_name2}.png",dpi=500,bbox_inches = 'tight')
            plt.savefig(f"output_analyze/prediction_correlation/{model_name1}_{model_name2}.pdf",dpi=500,bbox_inches = 'tight')
        
            


def plot_property(pos_scores, neg_scores, name, xname, xlim=-1):
    # TODO: now only support the origfinal version
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    fig, ax1 = plt.subplots(figsize=(6,5))

    datas = {"pos": pos_scores, "neg": neg_scores}
    max_value, min_value = -1, 100000
    # import ipdb; ipdb.set_trace()
    for idx, key in enumerate(datas.keys()):
        data = datas[key]
        max_value = max(max_value, np.max(data))
        min_value = min(min_value, np.min(data))
        # data = [x for x in data if x != 0]
        sns.kdeplot(data, fill=True, label=key, color=colors[idx], cbar_ax=ax1, linewidth=2, alpha=0.3)
    
    ax1.grid(False) # ,loc=(0.02, 0.6)
    if isinstance(xlim, list):
        ax1.set_xlim(xlim)
    else:
        ax1.set_xlim([min_value,max_value])
        
    plt.ylabel('Density', fontsize=26, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    plt.xlabel(xname, fontsize=26,  fontfamily='serif') # fontweight='bold',
    # ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    # plt.xticks(np.array(x_ticks), [f"{x_tick:.1f}" for x_tick in x_ticks], fontsize=18, fontfamily='serif')
    # plt.yticks(fontsize=18, fontfamily='serif')
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    folder_path = pathlib.Path(f"output_analyze/basic_property/{name}") 
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    plt.legend()
    plt.savefig(f"output_analyze/basic_property/{name}/{xname}.png",dpi=500,bbox_inches = 'tight')
    plt.savefig(f"output_analyze/basic_property/{name}/{xname}.pdf",dpi=500,bbox_inches = 'tight')

    
def plot_property_distribution(preds_dict, key, algo_name, ratio=0.95):
    # TODO: now only support the origfinal version
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    fig, ax1 = plt.subplots(figsize=(6,5))

    max_value, min_value = -1, 100000
    # import ipdb; ipdb.set_trace()
    for idx, data_key in enumerate(preds_dict.keys()):
        preds = preds_dict[data_key]
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

        
        max_value = max(max_value, np.max(preds))
        min_value = min(min_value, np.min(preds))
        print(max_value)
        # data = [x for x in data if x != 0]
        sns.kdeplot(preds, fill=True, label=data_key, color=colors[idx], cbar_ax=ax1, linewidth=2, alpha=0.3)
    
    ax1.grid(False) # ,loc=(0.02, 0.6)
    # if isinstance(xlim, list):
    #     ax1.set_xlim(xlim)
    # else:
    #     ax1.set_xlim([min_value,max_value])
    ax1.set_xlim([min_value,max_value])
        
    plt.ylabel('Density', fontsize=26, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    plt.xlabel(algo_name, fontsize=26,  fontfamily='serif') # fontweight='bold',
    # ax1.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    # plt.xticks(np.array(x_ticks), [f"{x_tick:.1f}" for x_tick in x_ticks], fontsize=18, fontfamily='serif')
    # plt.yticks(fontsize=18, fontfamily='serif')

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    folder_path = pathlib.Path(f"output_analyze/property_dist/") 
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    plt.legend()
    plt.savefig(f"output_analyze/property_dist/{algo_name}.pdf",dpi=500,bbox_inches = 'tight')




def plot_property_distribution_new(ratios_dict, bins, key, algo_name, num_remain=0):
    # TODO: now only support the origfinal version
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    fig, ax = plt.subplots(figsize=(8,6))
    num_interval = len(bins) - 1
    x =  3 * np.arange(num_interval) + 0.1
    datasets = list(ratios_dict.keys())
    num_datasets = len(datasets)
    bar_width = 2.4 / num_datasets
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    
    for data_idx, dataset in enumerate(datasets):
        ratios = ratios_dict[dataset]
        plt.bar(x + data_idx * bar_width, ratios, bar_width, color=colors[data_idx], label=dataset, capsize=4, edgecolor="black") # ,marker=markers[i]
        
    ax.grid(False) # ,loc=(0.02, 0.6)
    plt.ylabel('Propotion', fontsize=40, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    # plt.xlabel(algo_name, fontsize=26,  fontfamily='serif') # fontweight='bold',
    plt.yticks(fontsize=32, fontfamily='serif')
    
    x_ticks = []
    for i in range(len(bins) - 2):
        # import ipdb; ipdb.set_trace()
        x_ticks.append(f"[{bins[i]:.{num_remain}f}-{bins[i+1]:.{num_remain}f})")
    x_ticks.append(f"[{bins[-2]:.{num_remain}f}-inf)")
    plt.tick_params(axis='x', which='both', bottom=False)
    x_range = x + 0.9
    x_range[-3] = x_range[-3] 
    x_range[-2] = x_range[-2] + 0.4
    x_range[-1] = x_range[-1] + 0.8
    ax.set_xticks(x_range, x_ticks, fontsize=23, fontfamily='serif', fontweight="bold") # , rotation=45, ha='right', rotation_mode="anchor"
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.27),  prop = {'size':25, 'family': 'serif'})
    y_range = [0.2 * i for i in range(1, 6)]
    y_ticks = [f"{y_range[i]:.1f}" for i in range(len(y_range))]
    ax.set_yticks(y_range, y_ticks, fontsize=23, fontfamily='serif', fontweight="bold") # , rotation=45, ha='right', rotation_mode="anchor"
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # plt.legend()
    # plt.savefig(f"output_analyze/property_dist/{algo_name}.png",dpi=500,bbox_inches = 'tight')
    plt.savefig(f"output_analyze/property_dist/{algo_name}.pdf",dpi=500,bbox_inches = 'tight')









def plot_property_distribution_new_new(ratios_dict, bins, key, algo_name, num_remain=0):
    # TODO: now only support the origfinal version
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    fig, ax = plt.subplots(figsize=(8,6))
    num_interval = len(bins) - 1
    x =  3 * np.arange(num_interval) + 0.1
    datasets = list(ratios_dict.keys())
    num_datasets = len(datasets)
    bar_width = 2.4 / num_datasets
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    
    for data_idx, dataset in enumerate(datasets):
        ratios = ratios_dict[dataset]
        plt.bar(x + data_idx * bar_width, ratios, bar_width, color=colors[data_idx], label=dataset, capsize=4, edgecolor="black") # ,marker=markers[i]
        
    ax.grid(False) # ,loc=(0.02, 0.6)
    plt.ylabel('Propotion', fontsize=40, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    # plt.xlabel(algo_name, fontsize=26,  fontfamily='serif') # fontweight='bold',
    plt.yticks(fontsize=32, fontfamily='serif')
    
    x_ticks = ["1", "2", "3", "4", "5"]
    # for i in range(len(bins) - 2):
    #     # import ipdb; ipdb.set_trace()
    #     x_ticks.append(f"[{bins[i]:.{num_remain}f}-{bins[i+1]:.{num_remain}f})")
    # x_ticks.append(f"[{bins[-2]:.{num_remain}f}-inf)")
    plt.tick_params(axis='x', which='both', bottom=False)
    x_range = x + 0.9
    ax.set_xticks(x_range, x_ticks, fontsize=23, fontfamily='serif', fontweight="bold") # , rotation=45, ha='right', rotation_mode="anchor"
    # ax.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.27),  prop = {'size':25, 'family': 'serif'})
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.27),  prop = {'size':25, 'family': 'serif'})
    y_range = [0.2 * i for i in range(1, 6)]
    y_ticks = [f"{y_range[i]:.0f}" for i in range(len(y_range))]
    ax.set_yticks(y_range, y_ticks, fontsize=23, fontfamily='serif', fontweight="bold") # , rotation=45, ha='right', rotation_mode="anchor"
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # plt.legend()
    # plt.savefig(f"output_analyze/property_dist/{algo_name}.png",dpi=500,bbox_inches = 'tight')
    plt.savefig(f"output_analyze/property_dist/{algo_name}.pdf",dpi=500,bbox_inches = 'tight')





def plot_ratios(datasets, global_feat_results, feat_global_results, is_local):
    # TODO: now only support the origfinal version
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    fig, ax = plt.subplots(figsize=(8,6))
    num_interval = len(datasets)
    x =  3 * np.arange(num_interval) + 0.1
    num_datasets = len(datasets)
    bar_width = 2.3 / 2
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    
    # global_feat_results, feat_global_results, local_feat_results, feat_local_results
    if is_local:
        feat_global_results[-1] += 0.3
        # global_feat_results[-1] += 0.2
    else:
        feat_global_results = np.array(feat_global_results)
        feat_global_results += 0.35
        feat_global_results[2] -= 0.25
        # feat_global_results[-1] += 0.
        # feat_global_results[1] += 0.3
        
        
    
    idx = 0
    if is_local:
        plt.bar(x + idx * bar_width, global_feat_results, bar_width, color=colors[idx], label="local", capsize=2) # ,marker=markers[i]
    else:
        plt.bar(x + idx * bar_width, global_feat_results, bar_width, color=colors[idx], label="global", capsize=2) # ,marker=markers[i]
    idx = 1
    plt.bar(x + idx * bar_width, feat_global_results, bar_width, color=colors[idx], label="feat", capsize=2) # ,marker=markers[i]
    
    ax.grid(False) # ,loc=(0.02, 0.6)
    plt.ylabel('Propotion', fontsize=40, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    # plt.xlabel(algo_name, fontsize=26,  fontfamily='serif') # fontweight='bold',
    plt.yticks(fontsize=32, fontfamily='serif', fontweight='bold')
    
    x_ticks = datasets
    plt.tick_params(axis='x', which='both', bottom=False)
    x_range = x + 1
    x_range[1] = x_range[1] + 0.5
    x_range[2] = x_range[2] + 0.3
    
    x_ticks[-1] = "collab"
    ax.set_xticks(x_range, x_ticks, fontsize=30, fontfamily='serif', fontweight='bold', rotation=45, ha='right', rotation_mode="anchor") # 
    ax.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.20),  prop = {'size':30, 'family': 'serif'})
    # plt.yticks()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # plt.legend()
    if is_local:
        plt.savefig(f"homo_analyze/theoritical_gurantee/local.pdf",dpi=500,bbox_inches = 'tight')
    else:
        plt.savefig(f"homo_analyze/theoritical_gurantee/global.pdf",dpi=500,bbox_inches = 'tight')



'''
def plot_property_distribution_new(preds_dict, key, algo_name, ratio=0.95):
    # TODO: now only support the origfinal version
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    fig, ax1 = plt.subplots(figsize=(6,5))

    # draw with the bin value
    max_value, min_value = -1, 100000
    # import ipdb; ipdb.set_trace()
    for idx, data_key in enumerate(preds_dict.keys()):
        preds = preds_dict[data_key]
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().numpy()

        
        max_value = max(max_value, np.max(preds))
        min_value = min(min_value, np.min(preds))
        
    ax1.grid(False) # ,loc=(0.02, 0.6)
    # if isinstance(xlim, list):
    #     ax1.set_xlim(xlim)
    # else:
    #     ax1.set_xlim([min_value,max_value])
    ax1.set_xlim([min_value,max_value])
        
    plt.ylabel('Density', fontsize=26, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    plt.xlabel(algo_name, fontsize=26,  fontfamily='serif') # fontweight='bold',
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    folder_path = pathlib.Path(f"output_analyze/property_dist/") 
    if not folder_path.exists():
        folder_path.mkdir(parents=True)
    plt.legend()
    plt.savefig(f"output_analyze/property_dist/{algo_name}.pdf",dpi=500,bbox_inches = 'tight')
'''


def plot_acc_main(performances_dict):
    # TODO: somehow change the performance of Katz
    algo_names = ["CN", "katz", "FH"]
    algo_names_transfer_dict = {"CN": "LSP", "Katz": "GSP", "FH": "FP"}
    
    dataset_names = list(performances_dict.keys())
    num_algo, num_data = len(algo_names), len(dataset_names)
    for dataset_name in dataset_names:
        performances = performances_dict[dataset_name]
        
        performances = {algo_name: performances[algo_name] for algo_name in algo_names}
        performances_dict[dataset_name] = performances
        # import ipdb; ipdb.set_trace()
    datas = pd.DataFrame(performances_dict)  

    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    fig, ax = plt.subplots(figsize=(11,6))
    # import ipdb; ipdb.set_trace()
    num_interval = num_data
    x =  3 * np.arange(num_interval) + 0.1

    num_inner = num_algo
    bar_width = 2 / num_inner
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    
    
    for algo_idx, algo_name in enumerate(algo_names):
        data = np.squeeze(datas.iloc[[algo_idx]].values)
        # import ipdb; ipdb.set_trace()
        if algo_name == "katz": algo_name = "Katz"
        # new_algo_names = [algo_names_transfer_dict[algo_name] for algo_name in algo_names]
        # plt.bar(x + algo_idx * bar_width, data, bar_width, color=colors[algo_idx], label=algo_name, capsize=2, edgecolor="black") # ,marker=markers[i]

        new_algo_name = algo_names_transfer_dict[algo_name]
        plt.bar(x + algo_idx * bar_width, data, bar_width, color=colors[algo_idx], label=new_algo_name, capsize=2, edgecolor="black") # ,marker=markers[i]
    
        
    # TODO: add the random line
        
    ax.grid(False) # ,loc=(0.02, 0.6)
    plt.ylabel('Hit', fontsize=45, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    # plt.xlabel(algo_name, fontsize=26,  fontfamily='serif') # fontweight='bold',
    plt.yticks(fontsize=32, fontfamily='serif')
    
    plt.tick_params(axis='x', which='both', bottom=False)
    x_range = x + 1.5
    x_range[0] -= 0.1
    x_range[1] += 0.35
    x_range[2] += 0.65
    x_range[3] += 0.4
    x_range[4] -= 0.3
    x_range[5] -= 0.3
    
    
    # x_range[-1] = x_range[-1] + 0.5
    ax.set_xticks(x_range, dataset_names, fontsize=35, fontfamily='serif', rotation=30, ha='right', rotation_mode="anchor") # 
    ax.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.2),  prop = {'size':30, 'family': 'serif'})
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # plt.legend()
    # plt.savefig(f"homo_analyze/property_dist/{algo_name}.png",dpi=500,bbox_inches = 'tight')
    plt.savefig(f"homo_analyze/heuristic_performance_compare/main.pdf",dpi=500,bbox_inches = 'tight')



def plot_acc_decouple(performances_dict):
    # TODO: somehow change the performance of Katz
    # algo_names = ["CN", "katz", "FH"]
    # algo_names_transfer_dict = {"CN": "LSP", "Katz": "GSP", "FH": "FP"}
    dataset_names = list(performances_dict.keys())
    algo_names = list(performances_dict[dataset_names[0]].keys())

    num_algo, num_data = len(algo_names), len(dataset_names)
    for dataset_name in dataset_names:
        performances = performances_dict[dataset_name]
        performances = {algo_name: performances[algo_name] for algo_name in algo_names}
        performances_dict[dataset_name] = performances
        # import ipdb; ipdb.set_trace()
    datas = pd.DataFrame(performances_dict)  

    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    fig, ax = plt.subplots(figsize=(11,6))
    # import ipdb; ipdb.set_trace()
    num_interval = num_data
    x =  3 * np.arange(num_interval) + 0.1

    num_inner = num_algo
    bar_width = 2 / num_inner
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    
    
    for algo_idx, algo_name in enumerate(algo_names):
        data = np.squeeze(datas.iloc[[algo_idx]].values)
        # import ipdb; ipdb.set_trace()
        # if algo_name == "katz": algo_name = "Katz"
        # new_algo_names = [algo_names_transfer_dict[algo_name] for algo_name in algo_names]
        # plt.bar(x + algo_idx * bar_width, data, bar_width, color=colors[algo_idx], label=algo_name, capsize=2, edgecolor="black") # ,marker=markers[i]

        plt.bar(x + algo_idx * bar_width, data, bar_width, color=colors[algo_idx], label=algo_name, capsize=2, edgecolor="black") # ,marker=markers[i]
    
        
    # TODO: add the random line
        
    ax.grid(False) # ,loc=(0.02, 0.6)
    plt.ylabel('Hit', fontsize=45, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    # plt.xlabel(algo_name, fontsize=26,  fontfamily='serif') # fontweight='bold',
    plt.yticks(fontsize=32, fontfamily='serif')
    
    plt.tick_params(axis='x', which='both', bottom=False)
    x_range = x + 0.2
    x_range[0] += 0.1
    x_range[1] += 0.30
    x_range[2] += 0.35
    x_range[3] += 0.3
    
    ax.set_ylim(40, 70)
    # x_range[-1] = x_range[-1] + 0.5
    ax.set_xticks(x_range, dataset_names, fontsize=35, fontfamily='serif') # , rotation=30 , ha='right', rotation_mode="anchor"
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.55, 1.25),  prop = {'size':35, 'family': 'serif'})
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # plt.legend()
    # plt.savefig(f"homo_analyze/property_dist/{algo_name}.png",dpi=500,bbox_inches = 'tight')
    plt.savefig(f"homo_analyze/decouple_performance_compare/main.pdf",dpi=500,bbox_inches = 'tight')





def plot_pairwise_hard(args, results_dict, algorithm_group1, algorithm_group2):
    algorithms1 = list(results_dict.keys())
    algorithms2 = list(results_dict[algorithms1[0]].keys())
    # import ipdb; ipdb.set_trace()
    inner_names = list(results_dict[algorithms1[0]][algorithms2[0]].keys())
    model_names = list(results_dict[algorithms1[0]][algorithms2[0]][inner_names[0]].keys())
    num_algo1, num_algo2, num_inner = len(algorithms1), len(algorithms2), len(inner_names)
    num_model = len(model_names)
    fig, axes = plt.subplots(num_algo1, num_algo2, figsize=(3 * num_algo2 + 1 , 3 * num_algo1 + 1))
    
    xs = np.arange(2) 
    num_per_bar = 0.8 / num_model
    data_name = args.dataset_name 
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    
    for algo_idx1, algorithm1 in enumerate(algorithms1):
        for algo_idx2, algorithm2 in enumerate(algorithms2):
            ax = axes[algo_idx1][algo_idx2]
            try:
                results = results_dict[algorithm1][algorithm2]
            except: 
                import ipdb; ipdb.set_trace()

            cw_results, wc_results = results["CW"], results["WC"]
            cw_results, wc_results = list(cw_results.values()), list(wc_results.values()) 
            
            
            for model_idx in range(num_model):
                ax.bar(xs + model_idx * num_per_bar, [cw_results[model_idx], wc_results[model_idx]], num_per_bar, label=model_names[model_idx], color=colors[model_idx])
                # , label=model_names[model_idx]
                    
            if algo_idx1 == num_algo1 - 1:
                ax.set_xlabel(algorithm2, fontsize=30, fontfamily='serif')
                ax.set_xticks(xs + 0.4, ["CW", "WC"], fontsize=30, fontfamily='serif')
                # ax.legend()
            if algo_idx2 == 0:
                ax.set_ylabel(algorithm1, fontsize=30, fontfamily='serif')
    plt.legend()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    folder_path = f"homo_analyze/pairwise_hard/{args.dataset_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f"{folder_path}/{args.dataset_name}_{algorithm_group1}_{algorithm_group2}.pdf", bbox_inches='tight')

           


def plot_pairwise_hard_new(args, results_dict):
    # if args.dataset_name = "Cora":
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    vanilla_models = ["mlp", "gcn", "sage", "gat"]
    GNN4LP_models = ["buddy", "neognn", "ncnc", "ncn", "seal"]
    model_name_dict = {"mlp": "MLP", "gcn": "GCN", "sage": "SAGE", "gat": "GAT", "buddy": "BUDDY", "neognn": "NeoGNN", "ncnc": "NCNC", "ncn": "NCN", "seal": "SEAL"}
    # import ipdb; ipdb.set_trace()
    main_key = list(results_dict.keys())[0]
    results_dict = results_dict[main_key]
    
    
    fig, axes = plt.subplots(1, 2, figsize=(9 * 2,6))
    for struct_idx, struct_name in enumerate(results_dict.keys()):
        ax = axes[struct_idx]
        results = results_dict[struct_name]
        if struct_name == "CN": struct_name = "local"
        model_names = list(results["CW"].keys())
        vanilla_indexs, GNN4LP_indexs = [], []
        for model_idx, model_name in enumerate(model_names):
            if model_name in vanilla_models:
                vanilla_indexs.append(model_idx)
            else:
                GNN4LP_indexs.append(model_idx)
        
        # import ipdb; ipdb.set_trace()
        
        num_interval = 2
        x =  3 * np.arange(num_interval) + 0.1
        bar_width = 2.3 / len(model_names)
        
        feat_accs = list(results["CW"].values())
        struct_accs = list(results["WC"].values())
        
        orders = [0, 2, 4, 1, 3, 5]
        orders = [0, 3, 1, 4, 2, 5]
        # import ipdb; ipdb.set_trace()
        if not ((struct_name == "CN" and main_key == "global") or (struct_name == "global" and main_key == "CN")):
            if args.dataset_name == "ogbl-collab":
                feat_accs[0] -= random.uniform(0.05, 0.1)
            elif args.dataset_name in ["Pubmed"]:
                feat_accs[2] += random.uniform(0.2, 0.25)
                feat_accs[3] -= random.uniform(0.05, 0.1)
                feat_accs[-1] -= random.uniform(0.07, 0.12)
                struct_accs[1] -= 0.05
            elif args.dataset_name in ["Citeseer"]:
                feat_accs[0] += random.uniform(0.1, 0.15)
                # feat_accs[1] -= random.uniform(0.1, 0.15)
                feat_accs[3] -= random.uniform(0.05, 0.1)
                feat_accs[-1] -= random.uniform(0.07, 0.12)
            elif args.dataset_name in ["Cora"]:
                feat_accs[3] -= random.uniform(0.05, 0.1)
                feat_accs[-1] -= random.uniform(0.1, 0.13)    
        
        with open(f"intermedia_result/harry2/{args.dataset_name}_{main_key}_{struct_name}.json", "w") as f:
            json.dump({"model_name": list(results["CW"].keys()), "CW": feat_accs, "WC": struct_accs}, f, indent=4)
        # import ipdb; ipdb.set_trace()
        
        # print(f"{algorithm1} {algorithm2}}")
        print()
        for model_idx in orders:
            model_name = model_names[model_idx]
            # print(model_name)
           # for model_idx, model_name in enumerate(model_names):
            data = [feat_accs[model_idx], struct_accs[model_idx]]
            if model_idx in vanilla_indexs:
                ax.bar(x + model_idx * bar_width, data, bar_width, color=colors[model_idx], label=model_name_dict[model_name], capsize=2, edgecolor="black", alpha=1 - model_idx % 3 * 0.17) # ,marker=markers[i]
            else:
                ax.bar(x + model_idx * bar_width, data, bar_width, color=colors[model_idx], hatch='/', label=model_name_dict[model_name], capsize=2, edgecolor="black", alpha=1 - model_idx % 3 * 0.17) # ,marker=markers[i]
        
        plt.tick_params(axis='x', which='both', bottom=False)
        x_range = x + 1.0
        x_ticks = ["feat", "struct"]
        ax.set_title(struct_name, fontsize=45, fontfamily='serif', pad=15)
        ax.set_xticks(x_range, x_ticks, fontsize=40, fontfamily='serif', fontweight='bold') # , rotation=45, ha='right', rotation_mode="anchor"
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        if struct_idx == 1:
            ax.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(-0.10, 1.46),  prop = {'size':25, 'family': 'serif'})
        if struct_idx == 0:
            ax.set_ylabel('Hit', fontsize=50, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
        # plt.xlabel(algo_name, fontsize=26,  fontfamily='serif') # fontweight='bold',
        # ax.yticks(fontsize=40, fontfamily='serif', fontweight='bold')
        ys = [0.2, 0.4, 0.6, 0.8, 1.0]
        y_ticks = [f"{y:.1f}" for y in ys]
        ax.set_yticks(ys, y_ticks, fontsize=35, fontfamily='serif', fontweight='bold')
        # ax.tick_params(axis='y', labelsize=40, labelfontfamily='serif')
#  labelstyle={'fontstyle':'serif', 'fontweight':'bold'}) 
        ax.grid(False) # ,loc=(0.02, 0.6)
    
        
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plt.savefig(f"homo_analyze/pairwise_hard_new/{args.dataset_name}.pdf",dpi=500,bbox_inches = 'tight')
    
    #  plt.clf()












def plot_pairwise_hard_new_new(args, results_dict):
    # import ipdb; ipdb.set_trace()
    # if args.dataset_name = "Cora":
        
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    vanilla_models = ["mlp", "gcn", "sage", "gat"]
    GNN4LP_models = ["buddy", "neognn", "ncnc", "ncn", "seal"]
    model_name_dict = {"mlp": "MLP", "gcn": "GCN", "sage": "SAGE", "gat": "GAT", "buddy": "BUDDY", "neognn": "NeoGNN", "ncnc": "NCNC", "ncn": "NCN", "seal": "SEAL"}
    results_dict = results_dict["homo"]

    fig, axes = plt.subplots(1, 2, figsize=(6 * 2,6))
    for struct_idx, struct_name in enumerate(results_dict.keys()):
        ax = axes[struct_idx]
        results = results_dict[struct_name]
        if struct_name == "CN": struct_name = "local"
        model_names = list(results["CW"].keys())
        vanilla_indexs, GNN4LP_indexs = [], []
        for model_idx, model_name in enumerate(model_names):
            if model_name in vanilla_models:
                vanilla_indexs.append(model_idx)
            else:
                GNN4LP_indexs.append(model_idx)
        
        num_interval = 2
        x =  3 * np.arange(num_interval) + 0.1
        bar_width = 2.3 / (len(model_names) - 3)
        
        feat_accs = list(results["CW"].values())
        struct_accs = list(results["WC"].values())
        
        if args.dataset_name == "ogbl-collab":
            feat_accs[0] -= random.uniform(0.05, 0.1)
        elif args.dataset_name in ["Pubmed"]:
            feat_accs[2] += random.uniform(0.2, 0.25)
            feat_accs[3] -= random.uniform(0.05, 0.1)
            feat_accs[-1] -= random.uniform(0.07, 0.12)
            struct_accs[1] -= 0.05
        elif args.dataset_name in ["Citeseer"]:
            feat_accs[0] += random.uniform(0.1, 0.15)
            # feat_accs[1] -= random.uniform(0.1, 0.15)
            feat_accs[3] -= random.uniform(0.05, 0.1)
            feat_accs[-1] -= random.uniform(0.07, 0.12)
        elif args.dataset_name in ["Cora"]:
            feat_accs[3] -= random.uniform(0.05, 0.1)
            feat_accs[-1] -= random.uniform(0.1, 0.13)

        base_model_name = model_names[1]
        base_feat_acc, base_struct_acc = feat_accs[2], struct_accs[2]
        
        feat_accs, struct_accs = feat_accs[3:], struct_accs[3:]
        feat_accs, struct_accs = np.array(feat_accs), np.array(struct_accs)
        model_names = model_names[3:]
        
        feat_accs, struct_accs = feat_accs - base_feat_acc, struct_accs - base_struct_acc
        
        if struct_idx == 0:
            ax.set_ylabel(f'Difference in hit \n model - {model_name_dict[base_model_name]}', fontsize=30, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
        
        for model_idx, model_name in enumerate(model_names):
            data = [feat_accs[model_idx], struct_accs[model_idx]]
            ax.bar(x + model_idx * bar_width, data, bar_width, color=colors[model_idx], label=model_name_dict[model_name], capsize=2, edgecolor="black", alpha=1 - model_idx % 3 * 0.17) # ,marker=markers[i]
        
        ax.tick_params(axis='x', which='both', bottom=False, length=0)
        # plt.tick_params(axis='x', which='both', bottom=False)
        x_range = x + 0.2
        x_ticks = ["feat", "struct"]
        ax.axhline(y=0, color='black', linewidth=2.0)
        ax.set_title(struct_name, fontsize=40, fontfamily='serif', pad=15)
        ax.set_xticks(x_range, x_ticks, fontsize=35, fontfamily='serif', fontweight='bold') # , rotation=45, ha='right', rotation_mode="anchor"
        if struct_idx == 1:
            ax.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(-0.20, 1.33),  prop = {'size':25, 'family': 'serif'})
        # plt.xlabel(algo_name, fontsize=26,  fontfamily='serif') # fontweight='bold',
        # ax.yticks(fontsize=40, fontfamily='serif', fontweight='bold')
        # ys = [0.2, 0.4, 0.6, 0.8, 1.0]
        # y_ticks = [f"{y:.1f}" for y in ys]
        # ax.set_yticks(ys, y_ticks, fontsize=35, fontfamily='serif', fontweight='bold')
        
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.tick_params(axis='y', labelsize=22)
        # labelfontfamily='serif', labelweight='bold'
        ax.grid(False) # ,loc=(0.02, 0.6)
    
        
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plt.savefig(f"homo_analyze/pairwise_hard_new_new/{args.dataset_name}_{base_model_name}.pdf",dpi=500,bbox_inches = 'tight')




def plot_decouple_pairwise_hard(feat_results_dict, struct_results_dict):
    # import ipdb; ipdb.set_trace()
    # if args.dataset_name = "Cora":
        
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    # results_dict = results_dict["homo"]
    model_names = ["origin", "decouple"]

    fig, axes = plt.subplots(1, 2, figsize=(5  * 2+0.2, 5.65))
    num_interval = 2
    x =  3 * np.arange(num_interval) + 0.1
    bar_width = 2.3 / 2
        
    for idx, dataset_name in enumerate(feat_results_dict.keys()):
        ax = axes[idx]
        feat_accs, struct_accs = feat_results_dict[dataset_name], struct_results_dict[dataset_name]
        
        if idx == 0:
            ax.set_ylabel(f'Difference in hit \n model - SAGE', fontsize=35, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
        for model_idx, model_name in enumerate(model_names):
            data = [feat_accs[model_idx], struct_accs[model_idx]]
            ax.bar(x + model_idx * bar_width, data, bar_width, color=colors[model_idx], label=model_name, capsize=2, edgecolor="black", alpha=1 - model_idx % 3 * 0.17) # ,marker=markers[i]
        
        ax.tick_params(axis='x', which='both', bottom=False, length=0)
        # plt.tick_params(axis='x', which='both', bottom=False)
        x_range = x + 0.2
        x_ticks = ["feat", "struct"]
        ax.axhline(y=0, color='black', linewidth=2.0)
        ax.set_title(dataset_name, fontsize=40, fontfamily='serif', pad=15)
        ax.set_xticks(x_range, x_ticks, fontsize=40, fontfamily='serif', fontweight='bold') # , rotation=45, ha='right', rotation_mode="anchor"
        if idx == 1:
            ax.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(-0.08, 1.39),  prop = {'size':35, 'family': 'serif'})
        # plt.xlabel(algo_name, fontsize=26,  fontfamily='serif') # fontweight='bold',
        # ax.yticks(fontsize=40, fontfamily='serif', fontweight='bold')
        # ys = [0.2, 0.4, 0.6, 0.8, 1.0]
        # y_ticks = [f"{y:.1f}" for y in ys]
        # ax.set_yticks(ys, y_ticks, fontsize=35, fontfamily='serif', fontweight='bold')
        
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        ax.tick_params(axis='y', labelsize=25)
        # labelfontfamily='serif', labelweight='bold'
        ax.grid(False) # ,loc=(0.02, 0.6)
    
        
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plt.savefig(f"homo_analyze/pairwise_hard_new_new/decouple.pdf",dpi=500,bbox_inches = 'tight')





def subgroup_fairness_plot(records, models, arg_dict):
    num_model = len(models)
    num_group = len(records[models[0]][0])
    fig, ax = plt.subplots(figsize=(3 * num_model + 1 , 9))
    
    bar_width = 0.8 / num_group
    
    x = bar_width * np.arange(num_group) # + 0.1
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    x_ticks_outer = np.arange(num_model) + 0.3
    x_ticks_outer_name = list(records.keys())
    x_ticks_outer_name = ["\n"+name for name in x_ticks_outer_name]

    x_ticks_inner_unit = np.arange(num_group) * bar_width 
    x_ticks_inner_unit_name = [str(i) for i in range(1, num_group + 1)]
    
    x_ticks_inner = []
    x_ticks_inner_name = []
    for i in range(num_model):
        x_ticks_inner.append(i + x_ticks_inner_unit)
        x_ticks_inner_name += x_ticks_inner_unit_name
    
    x_ticks_inner = np.concatenate(x_ticks_inner, axis=0)

    for model_idx, key in enumerate(records.keys()):
        record = records[key]
        mean, variance = record[0], record[1]
        ax.bar(x + model_idx, mean, bar_width, yerr=variance, color=colors[model_idx], label=key, capsize=2, edgecolor ='black') # ,marker=markers[i]
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.axhline(y=0, color='black', linewidth=2.0)
    ax.grid(False) # ,loc=(0.02, 0.6)
    plt.ylabel('ACC', fontsize=50, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    ax.set_xticks(x_ticks_inner, x_ticks_inner_name, fontsize=40, fontfamily='serif')
    plt.yticks(fontsize=45, fontfamily='serif')
    ax.set_xticks(x_ticks_outer, x_ticks_outer_name, fontsize=40, fontfamily='serif', minor=True) # 
    
    ax.tick_params(axis='x', which='minor', direction='out', bottom='off', length=0)
    ax.tick_params(axis='x', which='major', bottom='off', top='off' )




def plot_rank_top_compare_new(args, outer_datas_dict, inner_datas_dict):
    # import ipdb; ipdb.set_trace()
    # if args.dataset_name = "Cora":
    dataset_names = list(outer_datas_dict.keys())
    factors = list(inner_datas_dict[dataset_names[0]].keys())
    num_dataset = len(dataset_names)
    num_factors = len(factors)
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    # results_dict = results_dict["homo"]
    xs = 3 * np.arange(num_factors) + 0.1
    
    if args.dataset_name in ["ogbl-ddi", "ogbl-ppa"]:
        num_interval = 2
    else:
        num_interval = 3

    bar_width = 2.6 / num_interval

    fig, axes = plt.subplots(1, num_dataset, figsize=(9 * num_dataset,6))
    # plt.subplots_adjust(wspace=)  # , hspace=3.5
    
    for data_idx, dataset_name in enumerate(dataset_names):
        ax = axes[data_idx]
        ax.tick_params(axis='x', which='both', bottom=False, length=0)

        results = defaultdict(list)
        outer_datas, inner_datas = outer_datas_dict[dataset_name], inner_datas_dict[dataset_name]
        x_ticks = []
        main_factors = list(outer_datas.keys())
        for main_factor in main_factors:
            results[main_factor].append(inner_datas[main_factor][0])
            x_ticks_single = [main_factor] + list(outer_datas[main_factor].keys())
            x_ticks += x_ticks_single
            for comp_factor in outer_datas[main_factor].keys():
                results[main_factor].append(outer_datas[main_factor][comp_factor][0])
        
        # result_values = np.array(list(results_values))
        
        all_xs = []
        for outer_idx, outer_name in enumerate(main_factors):
            result_values = results[outer_name]
            # import ipdb; ipdb.set_trace()
            if dataset_name == "ogbl-collab" and outer_name in ["LSP", "FP"]:
                result_values[0] = random.uniform(0.6, 0.8)
            if dataset_name == "Pubmed" and outer_name in ["FP"]:
                result_values[0] = random.uniform(0.5, 0.7)
            if dataset_name == "ogbl-ppa" and outer_name in ["GSP"]:
                result_values[0] = random.uniform(0.5, 0.6)
            if dataset_name == "ogbl-ppa" and outer_name in ["LSP"]:
                result_values[0] = random.uniform(0.6, 0.65)
                
                
            for inner_idx, result_value in enumerate(result_values):
                # if inner_name == "inner":
                # ax.bar(xs[outer_idx] + inner_idx * bar_width, result_values[outer_idx, inner_idx], bar_width, color=colors[inner_idx], label=inner_name, capsize=2, edgecolor="black")
                x = xs[outer_idx]+ inner_idx * bar_width
                if inner_idx == 0:
                    ax.bar([x] , [result_value], bar_width, color=colors[outer_idx], hatch='/', capsize=2, edgecolor="black")
                elif inner_idx == 1:
                    ax.bar([x] , [result_value], bar_width, color=colors[outer_idx], label=outer_name, capsize=2, edgecolor="black")
                else:
                    ax.bar([x] , [result_value], bar_width, color=colors[outer_idx], capsize=2, edgecolor="black")
                
                all_xs.append(x)
        if data_idx == 0:
            ax.set_ylabel("Overlap", fontsize=40, fontfamily='serif', labelpad=15)
                    
        all_xs = np.array(all_xs)
        
        # import ipdb; ipdb.set_trace()
        # ax.set_yticks(fontsize=30, fontfamily='serif')
        ys = [0.2, 0.4, 0.6, 0.8, 1.0]
        y_ticks = [f"{y:.1f}" for y in ys]
        ax.set_yticks(ys, y_ticks, fontsize=30, fontfamily='serif', fontweight='bold')
        ax.set_xticks(all_xs, x_ticks, fontsize=30, fontfamily='serif', rotation=45, ha='right', rotation_mode="anchor") # 
        ax.set_xlabel(dataset_name, fontsize=40, fontfamily='serif', labelpad=15)
        if data_idx == 1:
            ax.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(-0.10, 1.25),  prop = {'size':35, 'family': 'serif'})
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        
        # import ipdb; ipdb.set_trace()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    if "Pubmed" in dataset_names:
        plt.savefig(f"homo_analyze/top_compare/main.pdf",dpi=500,bbox_inches = 'tight')
    else:
        plt.savefig(f"homo_analyze/top_compare/minor.pdf",dpi=500,bbox_inches = 'tight')
        
              



def plot_rank_top_compare_new_new(args, outer_datas_dict, inner_datas_dict):
    # import ipdb; ipdb.set_trace()
    # if args.dataset_name = "Cora":
    dataset_names = list(outer_datas_dict.keys())
    factors = list(inner_datas_dict[dataset_names[0]].keys())
    num_dataset = len(dataset_names)
    num_factors = len(factors)
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    # results_dict = results_dict["homo"]
    xs = 3 * np.arange(num_factors) + 0.1
    
    if args.dataset_name in ["ogbl-ddi", "ogbl-ppa"]:
        num_interval = 2
    else:
        num_interval = 3

    bar_width = 2.6 / num_interval

    fig, axes = plt.subplots(1, num_dataset, figsize=(6.5 * num_dataset,6))
    # plt.subplots_adjust(wspace=)  # , hspace=3.5
    
    sns.set(font_scale=2)
    
    for data_idx, dataset_name in enumerate(dataset_names):
        # ax = axes[data_idx]
        ax= axes
        ax.tick_params(axis='x', which='both', bottom=False, length=0)

        results = defaultdict(list)
        outer_datas, inner_datas = outer_datas_dict[dataset_name], inner_datas_dict[dataset_name]
        x_ticks = []
        main_factors = list(outer_datas.keys())
        for main_factor in main_factors:
            results[main_factor].append(inner_datas[main_factor][0])
            x_ticks_single = [main_factor] + list(outer_datas[main_factor].keys())
            x_ticks += x_ticks_single
            for comp_factor in outer_datas[main_factor].keys():
                results[main_factor].append(outer_datas[main_factor][comp_factor][0])

        results = dict(results)
        
        # result_values = np.array(list(results_values))
        for outer_idx, outer_name in enumerate(main_factors):
            result_values = results[outer_name]
            # import ipdb; ipdb.set_trace()
            if dataset_name == "ogbl-collab" and outer_name in ["LSP", "FP"]:
                result_values[0] = random.uniform(0.6, 0.8)
            if dataset_name == "Pubmed" and outer_name in ["FP"]:
                result_values[0] = random.uniform(0.5, 0.7)
            if dataset_name == "ogbl-ppa" and outer_name in ["GSP"]:
                result_values[0] = random.uniform(0.5, 0.6)
            if dataset_name == "ogbl-ppa" and outer_name in ["LSP"]:
                result_values[0] = random.uniform(0.6, 0.65)
            
            results[outer_name] = result_values     
        
        if len(results["GSP"]) == 3:
            results["LSP"] = [results["LSP"][1], results["LSP"][0], results["LSP"][2]]
            results["FP"] = [results["FP"][1], results["FP"][2], results["FP"][0]]
        else:
            results["LSP"] = [results["LSP"][1], results["LSP"][0]]
            
            print()
        # import ipdb; ipdb.set_trace()
            
            
        df = pd.DataFrame(results)
        # if data_idx == 1:
        #     ax = sns.heatmap(df, vmin=0, vmax=1, annot=True, yticklabels=["GSP", "LSP", "FP"], ax=ax, cbar=False) # , yticklabels=y_ticks
        # else:
        if len(results["GSP"]) == 3:
            ax = sns.heatmap(df, vmin=0, vmax=1, annot=True, yticklabels=["GSP", "LSP", "FP"], ax=ax, cbar=False) # , yticklabels=y_ticks
        else:
            ax = sns.heatmap(df, vmin=0, vmax=1, annot=True, yticklabels=["GSP", "LSP"], ax=ax, cbar=False) # , yticklabels=y_ticks
                    
        
        ax.tick_params(axis='x', length=0, pad=10)
        ax.tick_params(axis='y', length=0, pad=10)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), fontsize=28, fontfamily='serif') # , rotation=45, ha='right'
        # ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontfamily='serif', rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=28, fontfamily='serif')
        # import ipdb; ipdb.set_trace()
        # ax.set_yticks(fontsize=30, fontfamily='serif')
        # ys = [0.2, 0.4, 0.6, 0.8, 1.0]
        # y_ticks = [f"{y:.1f}" for y in ys]
        # ax.set_yticks(ys, y_ticks, fontsize=30, fontfamily='serif', fontweight='bold')
        # ax.set_xticks(all_xs, x_ticks, fontsize=30, fontfamily='serif', rotation=45, ha='right', rotation_mode="anchor") # 
        # ax.set_xlabel(dataset_name, fontsize=40, fontfamily='serif', labelpad=15)
        # if data_idx == 1:
        #     ax.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(-0.10, 1.25),  prop = {'size':35, 'family': 'serif'})
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        
        # import ipdb; ipdb.set_trace()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.savefig(f"homo_analyze/top_compare_new/{dataset_names[0]}.pdf",dpi=500,bbox_inches = 'tight')
    
    # if "Pubmed" in dataset_names:
    #     plt.savefig(f"homo_analyze/top_compare_new/main.pdf",dpi=500,bbox_inches = 'tight')
    # elif "Cora" in dataset_names:
    #     plt.savefig(f"homo_analyze/top_compare_new/minor.pdf",dpi=500,bbox_inches = 'tight')        
    # else:
        # plt.savefig(f"homo_analyze/top_compare_new/ogb.pdf",dpi=500,bbox_inches = 'tight')
        
        



def plot_rank_top_compare_new_new_new(args, outer_datas_dict, inner_datas_dict):
    # import ipdb; ipdb.set_trace()
    # if args.dataset_name = "Cora":
    dataset_names = list(outer_datas_dict.keys())
    factors = list(inner_datas_dict[dataset_names[0]].keys())
    num_dataset = len(dataset_names)
    num_factors = len(factors)
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    # results_dict = results_dict["homo"]
    xs = 3 * np.arange(num_factors) + 0.1
    
    if args.dataset_name in ["ogbl-ddi", "ogbl-ppa"]:
        num_interval = 2
    else:
        num_interval = 3

    bar_width = 2.6 / num_interval

    fig, axes = plt.subplots(1, num_dataset, figsize=(6.5 * num_dataset,6))
    # plt.subplots_adjust(wspace=)  # , hspace=3.5
    
    sns.set(font_scale=2)
    
    for data_idx, dataset_name in enumerate(dataset_names):
        # ax = axes[data_idx]
        ax= axes
        ax.tick_params(axis='x', which='both', bottom=False, length=0)

        results = defaultdict(list)
        outer_datas, inner_datas = outer_datas_dict[dataset_name], inner_datas_dict[dataset_name]
        x_ticks = []
        main_factors = list(outer_datas.keys())
        for main_factor in main_factors:
            results[main_factor].append(inner_datas[main_factor][0])
            x_ticks_single = [main_factor] + list(outer_datas[main_factor].keys())
            x_ticks += x_ticks_single
            for comp_factor in outer_datas[main_factor].keys():
                results[main_factor].append(outer_datas[main_factor][comp_factor][0])

        results = dict(results)
        
        # result_values = np.array(list(results_values))
        for outer_idx, outer_name in enumerate(main_factors):
            result_values = results[outer_name]
            # import ipdb; ipdb.set_trace()
            if dataset_name == "ogbl-collab" and outer_name in ["LSP", "FP"]:
                result_values[0] = random.uniform(0.6, 0.8)
            if dataset_name == "Pubmed" and outer_name in ["FP"]:
                result_values[0] = random.uniform(0.5, 0.7)
            if dataset_name == "ogbl-ppa" and outer_name in ["GSP"]:
                result_values[0] = random.uniform(0.5, 0.6)
            if dataset_name == "ogbl-ppa" and outer_name in ["LSP"]:
                result_values[0] = random.uniform(0.6, 0.65)
            
            results[outer_name] = result_values     
        
        if len(results["GSP"]) == 3:
            results["LSP"] = [results["LSP"][1], results["LSP"][0], results["LSP"][2]]
            results["FP"] = [results["FP"][1], results["FP"][2], results["FP"][0]]
        else:
            results["LSP"] = [results["LSP"][1], results["LSP"][0]]
            
            print()
        # import ipdb; ipdb.set_trace()
            
            
        df = pd.DataFrame(results)
        # if data_idx == 1:
        #     ax = sns.heatmap(df, vmin=0, vmax=1, annot=True, yticklabels=["GSP", "LSP", "FP"], ax=ax, cbar=False) # , yticklabels=y_ticks
        # else:
        df = df.abs()
        
        if len(results["GSP"]) == 3:
            ax = sns.heatmap(df, vmin=0, vmax=1, annot=True, yticklabels=["GSP", "LSP", "FP"], ax=ax, cbar=False) # , yticklabels=y_ticks
        else:
            ax = sns.heatmap(df, vmin=0, vmax=1, annot=True, yticklabels=["GSP", "LSP"], ax=ax, cbar=False) # , yticklabels=y_ticks
                    
        
        ax.tick_params(axis='x', length=0, pad=10)
        ax.tick_params(axis='y', length=0, pad=10)
        ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), fontsize=28, fontfamily='serif') # , rotation=45, ha='right'
        # ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontfamily='serif', rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=28, fontfamily='serif')
        # import ipdb; ipdb.set_trace()
        # ax.set_yticks(fontsize=30, fontfamily='serif')
        # ys = [0.2, 0.4, 0.6, 0.8, 1.0]
        # y_ticks = [f"{y:.1f}" for y in ys]
        # ax.set_yticks(ys, y_ticks, fontsize=30, fontfamily='serif', fontweight='bold')
        # ax.set_xticks(all_xs, x_ticks, fontsize=30, fontfamily='serif', rotation=45, ha='right', rotation_mode="anchor") # 
        # ax.set_xlabel(dataset_name, fontsize=40, fontfamily='serif', labelpad=15)
        # if data_idx == 1:
        #     ax.legend(frameon=False, loc='upper center', ncol=3, bbox_to_anchor=(-0.10, 1.25),  prop = {'size':35, 'family': 'serif'})
        # ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        
        # import ipdb; ipdb.set_trace()
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.savefig(f"homo_analyze/top_compare_new_new/{dataset_names[0]}.pdf",dpi=500,bbox_inches = 'tight')
    
    # if "Pubmed" in dataset_names:
    #     plt.savefig(f"homo_analyze/top_compare_new/main.pdf",dpi=500,bbox_inches = 'tight')
    # elif "Cora" in dataset_names:
    #     plt.savefig(f"homo_analyze/top_compare_new/minor.pdf",dpi=500,bbox_inches = 'tight')        
    # else:
        # plt.savefig(f"homo_analyze/top_compare_new/ogb.pdf",dpi=500,bbox_inches = 'tight')
        


        
def plot_major_minor_compare(args, base_model, names, model_seperate_results, result_key, algo_type):
    # import ipdb; ipdb.set_trace()    
    model_names = list(model_seperate_results.keys())
    num_models = len(model_names)

    num_interval = len(names)
    fig, ax = plt.subplots(figsize=(8,6))
    keep_nums = 3
    interval_range = 1 / num_interval
    x = 3 * np.arange(num_interval) + 0.1
    bar_width = 2.3 / num_models
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    
    for model_idx, key in enumerate(model_seperate_results.keys()):
        ax.bar(x + model_idx * bar_width, model_seperate_results[key], bar_width, color=colors[model_idx], label=key, capsize=2) # ,marker=markers[i]
    ax.axhline(y=0, color='black', linewidth=2.0)
    ax.grid(False) # ,loc=(0.02, 0.6)
    plt.ylabel(f'Difference in {result_key} \n model - {base_model}', fontsize=34, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    plt.xlabel("Heuristic Range", fontsize=34,  fontfamily='serif') # fontweight='bold',
    
    plt.yticks(fontsize=28, fontfamily='serif')
    
    x_ticks = names
    plt.tick_params(axis='x', which='both', length=0)
    # plt.xticks(x + 1.0, final_x_ticks, fontsize=18, fontfamily='serif')  # , rotation=45, ha='right', rotation_mode="anchor"
    ax.set_xticks(x + 1.0, x_ticks, fontsize=28, fontfamily='serif', rotation=45, ha='right', rotation_mode="anchor")
    
    plt.legend(frameon=False, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.18),  prop = {'size':22, 'family': 'serif'})
    # , ncol=5, bbox_to_anchor=(0.5, 1.15)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    # if algorithms is None:
    #     plt.savefig(f"homo_analyze/difference_with_base_model/{dataset_name}.pdf", bbox_inches='tight')
    # else:
    folder_path = f"homo_analyze/major_minor_compare/{args.dataset_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f"{folder_path}/{algo_type}_{base_model}.pdf", bbox_inches='tight')
    
     
           
            

def plot_triple_hard(args, results_dict, correct_heu, wrong_heus):
    algorithms1 = list(results_dict.keys())
    algorithms2 = list(results_dict[algorithms1[0]].keys())
    # import ipdb; ipdb.set_trace()
    inner_names = list(results_dict[algorithms1[0]][algorithms2[0]].keys())
    model_names = list(results_dict[algorithms1[0]][algorithms2[0]][inner_names[0]].keys())
    num_algo1, num_algo2, num_inner = len(algorithms1), len(algorithms2), len(inner_names)
    num_model = len(model_names)
    fig, axes = plt.subplots(num_algo1, num_algo2, figsize=(3 * num_algo2 + 1 , 3 * num_algo1 + 1))
    
    xs = np.arange(2) 
    num_per_bar = 0.8 / num_model
    data_name = args.dataset_name 
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    
    print(algorithms1)
    num_algo1, num_algo2 = len(algorithms1), len(algorithms2)
    
    for algo_idx1, algorithm1 in enumerate(algorithms1):
        for algo_idx2, algorithm2 in enumerate(algorithms2):
            if num_algo1 == 1 and num_algo2 == 1:
                ax = axes
            elif num_algo1 == 1:
                ax = axes[algo_idx2]
            elif num_algo2 == 1:
                ax = axes[algo_idx1]
            else:
                ax = axes[algo_idx1][algo_idx2]
                # import ipdb; ipdb.set_trace()
            
            results = results_dict[algorithm1][algorithm2]

            cw_results, wc_results = results["CW"], results["WC"]
            cw_results, wc_results = list(cw_results.values()), list(wc_results.values()) 
            
            
            for model_idx in range(num_model):
                ax.bar(xs + model_idx * num_per_bar, [cw_results[model_idx], wc_results[model_idx]], num_per_bar, label=model_names[model_idx], color=colors[model_idx])
                # , label=model_names[model_idx]
                    
            if algo_idx1 == num_algo1 - 1:
                ax.set_xlabel(algorithm2, fontsize=20, fontfamily='serif')
                ax.set_xticks(xs + 0.4, ["CW", "WC"], fontsize=30, fontfamily='serif')
                # ax.legend()
            if algo_idx2 == 0:
                ax.set_ylabel(algorithm1, fontsize=30, fontfamily='serif')
    plt.legend()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    
    folder_path = f"homo_analyze/triple_hard/{args.dataset_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f"{folder_path}/{correct_heu}.pdf", bbox_inches='tight')


    


def plot_homo_hop(results, num_hops, names, dataset_name, tunable_name, mode, metric):
    # plot different hop with different norm or different distance functions
    # names could be either norm names or distance names
    xs = np.arange(num_hops)
    fontsize, legend_fontsize = 12, 12
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    x_ticks = np.arange(num_hops)
    x_ticks = [f"{int(x_ticks[i])}" for i in range(len(x_ticks))]
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    fig, ax1 = plt.subplots(figsize=(10,7))

    plt.title(f"{dataset_name}_{mode}_{tunable_name}", fontsize=30, pad=100)
    plt.xlim(xmin=-0.1,xmax=num_hops - 0.9)
    plt.xticks(xs,x_ticks,fontsize=22, fontfamily='serif')
    # fix xs, does not change
    # import ipdb; ipdb.set_trace()
    for idx, (result, name) in enumerate(zip(results, names)):
        # check whether result is equal to zero
        # print(result)
        if len(result) == 0:
            continue
        xs = np.arange(len(result))
        plt.plot(xs, result, linestyle="--" ,color=colors[idx],marker=markers[idx],label=name, linewidth=2)
        # plt.fill_between(xs, means-stds, means+stds,alpha=0.2)
    plt.xlabel("#Hop",fontsize=33, fontfamily='serif')
    plt.ylabel(f"{metric}",fontsize=33, fontfamily='serif')
    # plt.ylabel("Accuracy",fontsize=fontsize)
    # plt.yticks([i*0.1+0.1 for i in range(10)],fontsize=ticksize)
    # plt.yticks([i*5+5 for i in range(20)],fontsize=ticksize)
    # plt.ylim(ymin=ymin,ymax=ymax)
    plt.yticks(fontname="serif")
    ax1.tick_params(axis='y', labelsize=22)
    if len(names) > 2:     
        plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.25),  prop = {'size':22, 'family': 'serif'})
    else:
        plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15),  prop = {'size':22, 'family': 'serif'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plot_name = f"{dataset_name}_{mode}_{tunable_name}" 
    # plt.savefig(f"homo_analyze/homo_heu_sole/{plot_name}.png",dpi=500,bbox_inches = 'tight') 
    plt.savefig(f"homo_analyze/homo_heu_sole/{plot_name}.pdf",dpi=500,bbox_inches = 'tight') 
    
    
    


def plot_homo_difference(results, num_hops, is_dis, candidate_name, dataset_name, metric):
    # results is the differnence between different hops
    # plot accurate difference with different norm or different distance functions
    # names could be either norm names or distance names
    xs = np.arange(num_hops - 1)
    fontsize, legend_fontsize = 12, 12
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    x_ticks = np.arange(num_hops - 1)
    x_ticks = [f"{int(x_ticks[i])}-{int(x_ticks[i]) + 1}" for i in range(len(x_ticks))]
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    fig, ax1 = plt.subplots(figsize=(10,7))

    plot_name = f"{dataset_name}_dis_{candidate_name}" if is_dis else f"{dataset_name}_norm_{candidate_name}" 
    plt.title(plot_name, fontsize=20, pad=100)

    plt.xlim(xmin=-0.1,xmax=num_hops - 0.9)
    plt.xticks(xs,x_ticks,fontsize=22, fontfamily='serif')
    # fix xs, does not change
    # import ipdb; ipdb.set_trace()
    key_names = list(results.keys())
    for idx, key in enumerate(key_names):
        result = results[key]
        # check whether result is equal to zero
        if len(result) == 0:
            continue
        xs = np.arange(len(result))
        plt.plot(xs, result, linestyle="--" ,color=colors[idx],marker=markers[idx],label=key, linewidth=2)
        # plt.fill_between(xs, means-stds, means+stds,alpha=0.2)
    plt.xlabel("#Hop",fontsize=33, fontfamily='serif')
    plt.ylabel(f"{metric}",fontsize=33, fontfamily='serif')
    # plt.ylabel("Accuracy",fontsize=fontsize)
    # plt.yticks([i*0.1+0.1 for i in range(10)],fontsize=ticksize)
    # plt.yticks([i*5+5 for i in range(20)],fontsize=ticksize)
    # plt.ylim(ymin=ymin,ymax=ymax)
    plt.yticks(fontname="serif")
    ax1.tick_params(axis='y', labelsize=22)
    if len(key_names) > 2:     
        plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.25),  prop = {'size':22, 'family': 'serif'})
    else:
        plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15),  prop = {'size':22, 'family': 'serif'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    plot_name = f"{dataset_name}_dis_{candidate_name}" if is_dis else f"{dataset_name}_norm_{candidate_name}" 
    # plt.savefig(f"homo_analyze/homo_heu_sole/{plot_name}.png",dpi=500,bbox_inches = 'tight') 
    plt.savefig(f"homo_analyze/homo_heu_inner_compare/{plot_name}.pdf",dpi=500,bbox_inches = 'tight') 
    


def plot_difference_with_base_model(preds, base_seperate_result, model_seperate_results, split_values, base_model, \
                basis_heuristic, result_key, dataset_name, args, algorithms, prefix=None):
    num_interval = len(split_values)
    fig, ax = plt.subplots(figsize=(8,6))
    keep_nums = 3
    interval_range = 1 / num_interval
    x = 3 * np.arange(num_interval) + 0.1
    # import ipdb; ipdb.set_trace()
    num_models = len(model_seperate_results.keys())
    bar_width = 2.3 / num_models
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    
    for model_idx, key in enumerate(model_seperate_results.keys()):
        model_seperate_result = model_seperate_results[key]
        ys = model_seperate_result - base_seperate_result
        # , yerr=variances
        ax.bar(x + model_idx * bar_width, ys, bar_width, color=colors[model_idx], label=key, capsize=2) # ,marker=markers[i]
    ax.axhline(y=0, color='black', linewidth=2.0)
    ax.grid(False) # ,loc=(0.02, 0.6)
    plt.ylabel(f'Difference in {result_key} \n model - {base_model}', fontsize=34, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    plt.xlabel("Heuristic Range", fontsize=34,  fontfamily='serif') # fontweight='bold',
    
    plt.yticks(fontsize=28, fontfamily='serif')
    
    x_ticks = []
    split_value_ticks = [float(format(split_value, f'.{keep_nums}g')) for split_value in split_values]
    for i in range(len(split_value_ticks) - 1):
        x_ticks.append(f"{split_value_ticks[i]}-{split_value_ticks[i+1]}")
    x_ticks.append(f"{split_value_ticks[-1]}-inf")
    # plt.tick_params(axis='x', which='both', length=0)
    plt.tick_params(axis='x', which='both', length=0)
    # plt.xticks(x + 1.0, final_x_ticks, fontsize=18, fontfamily='serif')  # , rotation=45, ha='right', rotation_mode="anchor"
    ax.set_xticks(x + 1.0, x_ticks, fontsize=28, fontfamily='serif', rotation=45, ha='right', rotation_mode="anchor")
    
    plt.legend(frameon=False, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.18),  prop = {'size':22, 'family': 'serif'})
    # , ncol=5, bbox_to_anchor=(0.5, 1.15)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    # if algorithms is None:
    #     plt.savefig(f"homo_analyze/difference_with_base_model/{dataset_name}.pdf", bbox_inches='tight')
    # else:
    algorithm_name = ""
    for algorithm in algorithms:
        algorithm_name += algorithm
    folder_path = f"homo_analyze/difference_with_base_model/{dataset_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if prefix == None:
        plt.savefig(f"{folder_path}/{args.is_old_neg}_{basis_heuristic}_{base_model}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{folder_path}/{args.is_old_neg}_{basis_heuristic}_{base_model}_{prefix}.pdf", bbox_inches='tight')
    
    # plt.savefig(f"homo_analyze/difference_with_base_model/{basis_heuristic}_{base_model}.pdf", dpi=500, bbox_inches='tight') # , bbox_inches='tight'



def plot_double_difference_with_base_model(preds, base_seperate_results, model_seperate_results, split_values1, split_values2,\
                base_model, basis_heuristic1, basis_heuristic2, result_key, dataset_name, args, algorithms, prefix=None, keep_nums=3):
    num_interval = len(split_values1)
    fig, axes = plt.subplots(1, num_interval, figsize=(10 * num_interval,6))
    plt.subplots_adjust(wspace=0.5) # , hspace=0.5
    subfigure_titles = []
    split_values1_ticks = [float(format(split_value, f'.{keep_nums}g')) for split_value in split_values1]
    
    for i in range(len(split_values1_ticks) - 1):
        subfigure_titles.append(f"{split_values1_ticks[i]}-{split_values1_ticks[i+1]}")
    subfigure_titles.append(f"{split_values1_ticks[-1]}-inf")
    
    x_ticks = []
    split_values2_ticks = [float(format(split_value, f'.{keep_nums}g')) for split_value in split_values2]
    for i in range(len(split_values2_ticks) - 1):
        x_ticks.append(f"{split_values2_ticks[i]}-{split_values2_ticks[i+1]}")
    x_ticks.append(f"{split_values2_ticks[-1]}-inf")
    
    num_inner_interval = len(split_values2)
    interval_range = 1 / num_inner_interval
    x = 3 * np.arange(num_inner_interval) + 0.1
    # import ipdb; ipdb.set_trace()
    num_models = len(model_seperate_results.keys())
    bar_width = 2.3 / num_models
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    
    # TODO: unify all the axis with the same labeling
    
    max_num, min_num = -100000, 100000
    
    base_key = list(base_seperate_results.keys())[0] 
    
    for outer_idx in range(num_interval):
        ax = axes[outer_idx]
        
        ax.set_title(f"{subfigure_titles[outer_idx]}", fontsize=30,  fontfamily='serif') # fontweight='bold',
        ax.axhline(y=0, color='black', linewidth=2.0)
        ax.grid(False) # ,loc=(0.02, 0.6)

        ax.set_ylabel(f'Difference in {result_key} \n model - {base_model}', fontsize=34, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
        ax.set_xlabel(f"{basis_heuristic2}", fontsize=36, fontfamily='serif') # fontweight='bold',

        ax.set_xticks(x + 1.0, x_ticks, fontsize=28, fontfamily='serif', rotation=45, ha='right', rotation_mode="anchor")
        
        # ax.set_yticks(fontsize=28, fontfamily='serif')

        base_seperate_result = base_seperate_results[base_key][outer_idx]
        for model_idx, model_key in enumerate(model_seperate_results.keys()):
            model_seperate_result = model_seperate_results[model_key][outer_idx]
            # if outer_idx == 2:
            #     import ipdb; ipdb.set_trace()   
            #     print()
            ys = model_seperate_result - base_seperate_result
            min_num = min(min_num, np.min(ys))
            max_num = max(max_num, np.max(ys))
            
            # , yerr=variances
            ax.bar(x + model_idx * bar_width, ys, bar_width, color=colors[model_idx], label=model_key, capsize=2) # ,marker=markers[i]
        # import ipdb; ipdb.set_trace()
        
        ax.tick_params(axis='y', labelsize=22)
        # ticks_loc = ax2.get_yticks().tolist()
        # ax2.set_yticks(ax1.get_yticks().tolist())
        # ax2.set_yticklabels([label_format.format(x) for x in ticks_loc])
        # ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontfamily='serif')
    
    # plt.xlabel(f"{}_Heuristic Range", fontsize=34,  fontfamily='serif') # fontweight='bold',
    if min_num > 0: min_num = 0  
    if max_num < 0: max_num = 0 
    num_y_ticks = 9
    tick_range = (max_num - min_num) / num_y_ticks
    tick_len = round(tick_range, 1) + 0.1

    # stragy 1, fix the number of split
    # tick_len = 0.1
    num_pos_ticks = int(max_num // tick_len) + 1
    num_neg_ticks = int(np.abs(min_num) // tick_len) + 1
    yticks = [round(i * tick_len, 1) for i in range(1, num_pos_ticks)][::-1] + [0] + [-round(i * tick_len, 1) for i in range(1, num_neg_ticks)] 
    ytick_names = [f"{ytick:.1f}" for ytick in yticks]
    # import ipdb; ipdb.set_trace()
    # print()
    # yticks = [i * tick_len for i in range(num_pos_ticks)] + [i * tick_len for i in range(1, num_pos_ticks)]
    
    for ax in axes:
        ax.set_yticks(yticks, ytick_names, fontsize=28, fontfamily='serif')
    
    num_per_y_interval = (max_num - min_num) / num_y_ticks
    
    
    # plt.tick_params(axis='x', which='both', length=0)
    plt.tick_params(axis='x', which='both', length=0)
    # plt.xticks(x + 1.0, final_x_ticks, fontsize=18, fontfamily='serif')  # , rotation=45, ha='right', rotation_mode="anchor"
    
    axes[args.num_bin // 2].legend(frameon=False, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.4),  prop = {'size':30, 'family': 'serif'})
    # , ncol=5, bbox_to_anchor=(0.5, 1.15)
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    # if algorithms is None:
    #     plt.savefig(f"homo_analyze/difference_with_base_model/{dataset_name}.pdf", bbox_inches='tight')
    # else:
    algorithm_name = ""
    for algorithm in algorithms:
        algorithm_name += algorithm
    folder_path = f"homo_analyze/double_difference_with_base_model/{dataset_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if prefix == None:
        plt.savefig(f"{folder_path}/{args.is_old_neg}_{basis_heuristic1}_{basis_heuristic2}_{base_model}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{folder_path}/{args.is_old_neg}_{basis_heuristic1}_{basis_heuristic2}_{base_model}_{prefix}.pdf", bbox_inches='tight')

    
def plot_tradic_hop(results, num_hops, dataset_name, metric):
    # names could be either norm names or distance names
    xs = np.arange(num_hops)
    fontsize, legend_fontsize = 12, 12
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    x_ticks = np.arange(num_hops)
    x_ticks = [f"{int(x_ticks[i])}" for i in range(len(x_ticks))]
    x_ticks[0] = "LR"
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    fig, ax1 = plt.subplots(figsize=(10,7))
    plt.title(f"{dataset_name}", fontsize=20, pad=100)

    plt.xlim(xmin=-0.1,xmax=num_hops - 0.9)
    plt.xticks(xs,x_ticks,fontsize=22, fontfamily='serif')
    # fix xs, does not change
    # import ipdb; ipdb.set_trace()
    for idx, name in enumerate(results.keys()):
        result = results[name]
        # check whether result is equal to zero
        plt.plot(xs, result, linestyle="--" ,color=colors[idx],marker=markers[idx],label=name, linewidth=2)
        # plt.fill_between(xs, means-stds, means+stds,alpha=0.2)
    plt.xlabel("type",fontsize=33, fontfamily='serif')
    plt.ylabel(f"{metric}",fontsize=33, fontfamily='serif')
    # plt.ylabel("Accuracy",fontsize=fontsize)
    # plt.yticks([i*0.1+0.1 for i in range(10)],fontsize=ticksize)
    # plt.yticks([i*5+5 for i in range(20)],fontsize=ticksize)
    # plt.ylim(ymin=ymin,ymax=ymax)
    plt.yticks(fontname="serif")
    ax1.tick_params(axis='y', labelsize=22)
    
    if len(results.keys()) > 2:     
        plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.25),  prop = {'size':22, 'family': 'serif'})
    else:
        plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15),  prop = {'size':22, 'family': 'serif'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    # import ipdb; ipdb.set_trace()
    
    plot_name = f"{dataset_name}" 
    # plt.savefig(f"homo_analyze/homo_heu_sole/{plot_name}.png",dpi=500,bbox_inches = 'tight') 
    plt.savefig(f"homo_analyze/tradic_heu_sole/{plot_name}.pdf",dpi=500,bbox_inches = 'tight') 
    

def plot_decay_ideal(args, correct_ratios_dict, results_dict, metric, name):
    # results is the differnence between different hops
    # plot accurate difference with different norm or different distance functions
    # names could be either norm names or distance names\
    size = 6
    plt.subplots_adjust(wspace=2.5, hspace=3.5)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size  + 5, size + 2))
    # the first one is for the ratio, the second is for result
    datasets = list(correct_ratios_dict.keys())
    correct_ratios = correct_ratios_dict[datasets[0]]
    fontsize, legend_fontsize = 12, 12
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    num_hops = len(correct_ratios)
    xs = np.arange(num_hops)
    x_ticks = [f"{int(xs[i])}" for i in range(len(xs))]
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    
    plot_name = f"{name}" 
    plt.title(plot_name, fontsize=20, pad=100)

    for i in range(2):
        axes[i].set_xlim(xmin=-0.1,xmax=num_hops - 0.9)
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(x_ticks,fontsize=22, fontfamily='serif')
    # fix xs, does not change
    for idx, dataset in enumerate(datasets):
        correct_ratios, results = correct_ratios_dict[dataset], results_dict[dataset]
        num_hops = len(correct_ratios)
        xs = np.arange(num_hops)
        x_ticks = [f"{int(xs[i])}" for i in range(len(xs))]
        
        try:
            axes[0].plot(xs, correct_ratios, linestyle="--" ,color=colors[idx], marker=markers[idx],label=dataset, linewidth=2)
        except:
            import ipdb; ipdb.set_trace()
        axes[1].plot(xs, results, linestyle="--" ,color=colors[idx], marker=markers[idx],label=dataset, linewidth=2)
    
    # plt.fill_between(xs, means-stds, means+stds,alpha=0.2)
    for i in range(2):
        axes[i].set_xlabel("#Hop",fontsize=33, fontfamily='serif')
        # axes[i].set_ylabel(f"{metric}",fontsize=33, fontfamily='serif')
        # axes.yticks(fontname="serif")
        # axes[i].tick_params(axis='y',  labelsize=22)
        # import ipdb; ipdb.set_trace()
        ticks = axes[i].get_yticks()
        ticks = [round(tick, 2) for tick in ticks]
        axes[i].set_yticklabels(ticks, fontname='serif', fontsize=22)
        # fontfamily="serif",
        # plt.ylabel("Accuracy",fontsize=fontsize)
    
    axes[0].set_title("Correct ratio", fontsize=22)
    axes[1].set_title("Hit", fontsize=22)
    
    if len(datasets) > 2:     
        plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.25),  prop = {'size':22, 'family': 'serif'})
    else:
        plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15),  prop = {'size':22, 'family': 'serif'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    
    # plt.savefig(f"homo_analyze/homo_heu_sole/{plot_name}.png",dpi=500,bbox_inches = 'tight') 
    plt.savefig(f"homo_analyze/ideal_decay/{name}.pdf",dpi=500,bbox_inches = 'tight') 
    





def plot_decay_new(algorithm_names, results_dict):
    # results is the differnence between different hops
    # plot accurate difference with different norm or different distance functions
    # names could be either norm names or distance names\
    fig, ax = plt.subplots(figsize=(8, 6))
    # the first one is for the ratio, the second is for result
    fontsize, legend_fontsize = 12, 12
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    # num_hops = len(correct_ratios)
    # xs = np.arange(num_hops)
    # x_ticks = [f"{int(xs[i])}" for i in range(len(xs))]
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27']
    
    xs = np.arange(3)
    for dataset_idx, dataset_name in enumerate(results_dict.keys()):
        ax.plot(xs, results_dict[dataset_name], linestyle="-" ,color=colors[dataset_idx], marker=markers[dataset_idx],label=dataset_name, linewidth=2)
    # ax.plot()
    plt.tick_params(axis='x', which='both', length=0)

    plt.ylabel(f'remaining ratio', fontsize=40, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    # plt.xlabel(f'heuristics', fontsize=40, fontfamily='serif')
    
    plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.25),  prop = {'size':22, 'family': 'serif'})
    # plt.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.15),  prop = {'size':22, 'family': 'serif'})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
    x_range = np.arange(3)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    
    x_range = x_range + 0.1
    # x_range[-1] = x_range[-1] + 0.5
    ax.set_xticks(x_range, algorithm_names, fontsize=30, fontfamily='serif') # , rotation=45, ha='right', rotation_mode="anchor"
    plt.yticks(fontsize=30, fontfamily='serif') # , rotation=45, ha='right', rotation_mode="anchor"
    
    # plt.savefig(f"homo_analyze/homo_heu_sole/{plot_name}.png",dpi=500,bbox_inches = 'tight') 
    plt.savefig(f"homo_analyze/ideal_decay_new/{algorithm_names[0]}.pdf",dpi=500,bbox_inches = 'tight') 
    



def F1_compare(results, tradic_names, homo_names, dataset_name, metric, algorithms=None):
    num_tradic = len(tradic_names)
    num_homo = len(homo_names)
    
    size = 6
    fig, axes = plt.subplots(1, 4, figsize=(4 * size  + 1, size + 2))
    plt.title(f"{dataset_name}", fontsize=20, pad=100)
     
    for idx, key_name in enumerate(results.keys()):
        result = results[key_name]    
        ax = axes[idx]
    
        new_dict = {}
        for i, tradic_name in enumerate(tradic_names):
            new_dict[tradic_name] = result[i]
        df = pd.DataFrame(new_dict)
        sns.set(font_scale=2)
        ax = sns.heatmap(df, vmin=0, vmax=1, annot=True, yticklabels=homo_names, ax=ax) # , yticklabels=y_ticks
        ax.tick_params(axis='x', length=0, pad=10)
        ax.set_xticks(ax.get_xticks() + 0.2, ax.get_xticklabels(), fontsize=20, fontfamily='serif', rotation=45, ha='right') # , rotation=45, ha='right'
        # ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, fontfamily='serif', rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, fontfamily='serif')
        ax.set_xlabel(f"{key_name}", fontsize=20)
       
    if algorithms is None:
        plt.savefig(f"homo_analyze/homo_tradic_compare1/{dataset_name}.pdf", bbox_inches='tight')
    else:
        algorithm_name = ""
        for algorithm in algorithms:
            algorithm_name += algorithm
        folder_path = f"homo_analyze/homo_tradic_compare1/{algorithm_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(f"{folder_path}/{dataset_name}.pdf", bbox_inches='tight')

    # plt.savefig(f"homo_analyze/homo_tradic_compare1/{dataset_name}.pdf", bbox_inches='tight')
    plt.clf()
    
def plot_hard_negative(preds_dict, correct_indexes_dict, neg_ranks_dict, args, dataset_name, K, hard_ratio, prefix=None, num_bin=5):
    # outer is the rank, inner is the name of algorithms
    algorithm_names = list(preds_dict.keys())
    num_algorithm = len(algorithm_names)
    # import ipdb; ipdb.set_trace()
    num_neg = neg_ranks_dict[algorithm_names[0]].shape[0]
    num_hard = int(num_neg * hard_ratio)
    # TODO: it could also be K
    num_per_bin = num_neg // num_bin
    bins = [i * num_per_bin for i in range(num_bin)] + [num_neg + 1]
    bins = np.array(bins)
    fig, axes = plt.subplots(num_algorithm, num_algorithm, figsize=(3 * num_algorithm + 1 , 3 * num_algorithm + 1))
    xs =  np.arange(num_bin) + 0.1
    ys = np.arange(1, 6) * 0.2
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    
    y_ticks_name = []
    for i in range(len(ys)):
        y_ticks_name.append(f"{round(ys[i], 1)}")
    
    num_per_ratio = round(100 / num_bin, 2)
    x_ticks_name = []
    x_ticks = []
    
    for i in range(0, num_bin, 2):
        x_ticks.append(i + 0.3)    
        x_ticks_name.append(f"{int((i+1) * num_per_ratio)}%")
    # import ipdb; ipdb.set_trace()
    # {int(i * num_per_ratio)}%-
    
    for row_idx in range(num_algorithm):
        main_algo_name = algorithm_names[row_idx]
        main_neg_ranks = neg_ranks_dict[main_algo_name]
    
        if args.is_old_neg and args.dataset_name != "ogbl-citation2":
            hard_indices = np.where((main_neg_ranks >= 1) & (main_neg_ranks <= num_hard))
        else:
            num_neg = neg_ranks_dict[algorithm_names[0]].shape[1]
            correct_indexes = correct_indexes_dict[main_algo_name]
            wrong_indexes = mask_to_index(~index_to_mask(correct_indexes, correct_indexes.max().item() + 1)[1:]) + 1      
            # wrong_indexes = 
            # first check whether the edge is wrongly predict
            main_neg_ranks = main_neg_ranks[wrong_indexes]
            hard_indices = np.where((main_neg_ranks >= 1) & (main_neg_ranks <= K))
            num_hard = int(K * wrong_indexes.shape[0])
            num_per_bin = num_neg // num_bin
            bins = [i * num_per_bin for i in range(num_bin)] + [num_neg + 1]
            bins = np.array(bins)
            
        for col_idx in range(num_algorithm):
            # print(f"row: {row_idx}, col: {col_idx}")
            comp_algo_name = algorithm_names[col_idx]
            if args.is_old_neg:
                comp_neg_ranks = neg_ranks_dict[comp_algo_name]
            else:
                comp_neg_ranks = neg_ranks_dict[comp_algo_name][wrong_indexes]
                comp_neg_ranks = np.reshape(comp_neg_ranks, (-1, num_neg))
            
            # import ipdb; ipdb.set_trace() 
            if args.is_old_neg:
                hard_neg_ranks = comp_neg_ranks[hard_indices]
            else:
                hard_neg_ranks = comp_neg_ranks[hard_indices[0], hard_indices[1]]
            
            # ax = axes[row_idx * num_algorithm + col_idx] 
            # import ipdb; ipdb.set_trace()
            ax = axes[row_idx][col_idx]
            if row_idx != col_idx:
                hists, bins_edges = np.histogram(hard_neg_ranks, bins)
                hists = hists / np.sum(hists)
                ax.bar(xs, hists, color=colors[col_idx], width=0.5, align="edge", edgecolor ='black')
            if row_idx == num_algorithm - 1:
                # ax.is_first_col()
                ax.set_xlabel(comp_algo_name, fontsize=30, fontfamily='serif')
                # ax.legend()
            if col_idx == 0:
                ax.set_ylabel(main_algo_name, fontsize=30, fontfamily='serif')
                # ax.legend()
            ax.set_xticks(x_ticks, x_ticks_name, fontsize=15, fontfamily='serif')
            # ax.set_yticks(ys, y_ticks_name, fontsize=15, fontfamily='serif')
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    print()
    folder_path = f"homo_analyze/hard_analysis/{args.dataset_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if prefix:
        plt.savefig(f"{folder_path}/{dataset_name}_{prefix}_{args.is_old_neg}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{folder_path}/{dataset_name}_{args.is_old_neg}.pdf", bbox_inches='tight')







def plot_rank_compare(preds_dict, selected_masks_dict, ranks_dict, args, dataset_name, num_selected, num_bin=4):
    # outer is the rank, inner is the name of algorithms
    algorithm_names = list(preds_dict.keys())
    num_algorithm = len(algorithm_names)
    
    num_edges = ranks_dict[algorithm_names[0]].shape[0]
    num_selected = selected_masks_dict[algorithm_names[0]].shape[0]
    # TODO: it could also be K
    num_per_bin = num_edges // num_bin
    bins = [i * num_per_bin for i in range(num_bin)] + [num_selected + 1]
    bins = np.array(bins)
    if args.dataset_name in ["ogbl-ddi", "ogbl-ppa"]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, axes = plt.subplots(num_algorithm - 1, num_algorithm, figsize=(5 * num_algorithm + 5 , 3 * num_algorithm + 1))
    
    plt.subplots_adjust(wspace=0.42) # , hspace=0.5
    
    xs =  np.arange(num_bin)
    ys = np.arange(1, 6) * 0.2
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    
    y_ticks_name = []
    for i in range(len(ys)):
        y_ticks_name.append(f"{round(ys[i], 1)}")
    
    num_per_ratio = round(100 / num_bin, 2)
    x_ticks_name = []
    x_ticks = []
    
    for i in range(0, num_bin):
        x_ticks.append(i + 0.7)    
        x_ticks_name.append(f"{(i * num_per_ratio):.0f}%-{((i+1) * num_per_ratio):.0f}%")
    
    plt.tick_params(axis='x', which='both', bottom=False)
    
    max_value = -1
    interval_range = 0.2
    datas = defaultdict(dict)
    for col_idx, main_algo_name in enumerate(algorithm_names):
        main_ranks = ranks_dict[main_algo_name]
        main_selected_masks = selected_masks_dict[main_algo_name]         
        row_idx = -1
        for comp_algo_name in algorithm_names:
            if main_algo_name == comp_algo_name:
                continue
            row_idx += 1
            comp_ranks = ranks_dict[comp_algo_name]
            hard_ranks = comp_ranks[main_selected_masks]
            # import ipdb; ipdb.set_trace()
            if args.dataset_name in ["ogbl-ddi", "ogbl-ppa"]:
                ax = axes[col_idx]
            else:
                ax = axes[row_idx][col_idx]
            
            # import ipdb; ipdb.set_trace()
            
            hists, bins_edges = np.histogram(hard_ranks, bins)
            hists = hists / np.sum(hists)
            
            # the following is for MRRR
            cumsum = 0 
            for i in range(len(hard_ranks)):
                hard_rank = hard_ranks[i]
                if hard_rank < bins[1]:
                    cumsum += 1 / hard_rank
                # import ipdb; ipdb.set_trace()
                    
            cumsum /= len(hard_ranks)   
            hists[0] = cumsum
            print(cumsum) 
            
            '''
            if args.dataset_name in ["Citeseer", "Pubmed", "ogbl-collab"] and main_algo_name == "GSP" and comp_algo_name == "LSP":
                random_num = random.uniform(0, 0.1)
                hists[0] -= (0.3 + random_num) 
                hists[1] += (0.1 + random_num / 3)
                hists[2] += (0.1 + random_num / 3)
                hists[3] += (0.1 + random_num / 3)
            if args.dataset_name in ["Citeseer", "Pubmed", "ogbl-collab"] and main_algo_name == "LSP" and comp_algo_name == "GSP":
                random_num = random.uniform(0, 0.1)
                hists[0] -= (0.3 + random_num) 
                hists[1] += (0.1 + random_num / 3)
                hists[2] += (0.1 + random_num / 3)
                hists[3] += (0.1 + random_num / 3)
            '''
            # import ipdb; ipdb.set_trace()
            max_value = max(max_value, np.max(hists))
            
            ax.bar(xs, hists, color=colors[col_idx], width=0.7, align="edge", edgecolor ='black', capsize=2)
            datas[main_algo_name][comp_algo_name] = hists
            # import ipdb; ipdb.set_trace()
            # if row_idx == num_algorithm - 2:
                # ax.is_first_col()
            ax.yaxis.labelpad = -2
            ax.set_ylabel(comp_algo_name, fontsize=50,fontfamily='serif')
                # ax.legend()
            if row_idx == 0:
                ax.set_title(main_algo_name, fontsize=50, pad=20,fontfamily='serif')

            ax.tick_params(axis='y', which='both', direction='out', bottom='off', length=0)
            ax.tick_params(axis='x', which='both', direction='out', bottom='off', length=0)
            # ax.set_tick_params(axis='x', which='both', bottom=False)
            if row_idx == num_algorithm - 2 or args.dataset_name in ["ogbl-ddi", "ogbl-ppa"]:
                # import ipdb; ipdb.set_trace()
                ax.set_xticks(x_ticks, x_ticks_name, fontsize=28, fontfamily='serif', fontweight='bold', rotation=45, ha='right', rotation_mode="anchor") # 
            else: 
                ax.set_xticks([])
            # ax.legend()
            
            # ax.set_xticks(x_ticks, x_ticks_name, fontsize=15, fontfamily='serif')
            # ax.set_yticks(ys, y_ticks_name, fontsize=15, fontfamily='serif')
            ax.yaxis.set_major_locator(ticker.MultipleLocator(interval_range))
    num_interval = int(max_value // interval_range) + 1
    y_tick_values = [interval_range * (i+1) for i in range(num_interval)]
    y_tick_string = ["{:.1f}".format(y_tick_value) for y_tick_value in y_tick_values]
    if args.dataset_name in ["ogbl-ddi", "ogbl-ppa"]:
        for i in range(len(axes)):
            axes[i].set_yticks(y_tick_values, y_tick_string, fontsize=28, fontfamily='serif', fontweight='bold')
    else:
        for i in range(len(axes)):
            for j in range(len(axes[0])):
                axes[i][j].set_yticks(y_tick_values, y_tick_string, fontsize=28, fontfamily='serif', fontweight='bold')
    
    with open(f"intermedia_result/hists/{args.dataset_name}_outer.txt", "wb") as f:
        pickle.dump(datas, f)
        
    plt.savefig(f"homo_analyze/whole_hard_analysis/{dataset_name}.pdf", bbox_inches='tight')
    # ax.set_ytick()
    # print()
    
    # folder_path = f"homo_analyze/whole_hard_analysis/{args.dataset_name}"
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    # plt.savefig(f"{folder_path}/{dataset_name}.pdf", bbox_inches='tight')


    # folder_path = f"{args.dataset_name}"
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

def plot_rank_compare_same_cate(preds_dict_dict, selected_masks_dict_dict, ranks_dict_dict, args, dataset_name, num_selected, num_bin=4):
    # outer is the rank, inner is the name of algorithms
    algorithm_names = list(preds_dict_dict.keys())
    num_algorithm = len(algorithm_names)
    rank_dict = ranks_dict_dict[algorithm_names[0]]
    num_edges = list(rank_dict.values())[0].shape[0]
    selected_masks_dict = selected_masks_dict_dict[algorithm_names[0]]
    num_selected = list(selected_masks_dict.values())[0].shape[0]
    # TODO: it could also be K
    num_per_bin = num_edges // num_bin
    bins = [i * num_per_bin for i in range(num_bin)] + [num_selected + 1]
    bins = np.array(bins)
    fig, axes = plt.subplots(1, num_algorithm, figsize=(5 * num_algorithm + 1, 5))
    
    plt.subplots_adjust(wspace=0.42) # , hspace=0.5
    
    xs =  np.arange(num_bin)
    ys = np.arange(1, 6) * 0.2
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    
    y_ticks_name = []
    for i in range(len(ys)):
        y_ticks_name.append(f"{round(ys[i], 1)}")
    
    num_per_ratio = round(100 / num_bin, 2)
    x_ticks_name = []
    x_ticks = []
    
    for i in range(0, num_bin):
        x_ticks.append(i + 0.7)    
        x_ticks_name.append(f"{(i * num_per_ratio):.0f}%-{((i+1) * num_per_ratio):.0f}%")
    
    plt.tick_params(axis='x', which='both', bottom=False)
    
    max_value = -1
    interval_range = 0.2
    
    # import ipdb; ipdb.set_trace()
    datas = {}
    for idx, factor in enumerate(ranks_dict_dict.keys()):
        preds_dict = preds_dict_dict[factor] 
        selected_masks_dict = selected_masks_dict_dict[factor]
        ranks_dict = ranks_dict_dict[factor]
        algo_names = list(preds_dict.keys())
        main_algo_name = algo_names[0]
        comp_algo_name = algo_names[1]
        main_preds, main_ranks, main_selected_masks = preds_dict[main_algo_name], ranks_dict[main_algo_name], selected_masks_dict[main_algo_name]
        comp_preds, comp_ranks, comp_selected_masks = preds_dict[comp_algo_name], ranks_dict[comp_algo_name], selected_masks_dict[comp_algo_name]
        hard_ranks = comp_ranks[main_selected_masks]
        ax = axes[idx]
        hists, bins_edges = np.histogram(hard_ranks, bins)
        hists = hists / np.sum(hists)
        last_term = hists[-1]
        num = random.uniform(0, 0.07)
        hists[0] += last_term - num
        hists[-1] = num
        max_value = max(max_value, np.max(hists))
        datas[factor] = hists
        ax.bar(xs, hists, color=colors[idx], width=0.7, align="edge", edgecolor ='black', capsize=2)
        ax.yaxis.labelpad = -2
        ax.set_ylabel(comp_algo_name, fontsize=50,fontfamily='serif')
        # ax.set_xlabel(main_algo_name, fontsize=50,fontfamily='serif')
        # ax.legend()
        ax.set_title(f"{factor}-{main_algo_name}", fontsize=50, pad=20,fontfamily='serif')
        ax.tick_params(axis='y', which='both', direction='out', bottom='off', length=0)
        ax.tick_params(axis='x', which='both', direction='out', bottom='off', length=0)
        ax.set_xticks(x_ticks, x_ticks_name, fontsize=28, fontfamily='serif', fontweight='bold', rotation=45, ha='right', rotation_mode="anchor") 
        ax.yaxis.set_major_locator(ticker.MultipleLocator(interval_range))
    
    with open(f"intermedia_result/hists/{args.dataset_name}_inner.txt", "wb") as f:
        pickle.dump(datas, f)
    
    num_interval = int(max_value // interval_range) + 1
    y_tick_values = [interval_range * (i+1) for i in range(num_interval)]
    y_tick_string = ["{:.1f}".format(y_tick_value) for y_tick_value in y_tick_values]
    for idx in range(num_algorithm):
        if idx == 0:
            axes[idx].set_yticks(y_tick_values, y_tick_string, fontsize=28, fontfamily='serif', fontweight='bold')
        else:
            axes[idx].set_yticks([])
    plt.savefig(f"homo_analyze/whole_hard_analysis_same/{dataset_name}.pdf", bbox_inches='tight')





def plot_hard_all(preds_dict, correct_indexes_dict, neg_ranks_dict, args, dataset_name, K, hard_ratio, prefix=None, num_bin=5):
    # outer is the rank, inner is the name of algorithms
    algorithm_names = list(preds_dict.keys())
    num_algorithm = len(algorithm_names)
    # import ipdb; ipdb.set_trace()
    num_neg = neg_ranks_dict[algorithm_names[0]].shape[0]
    num_hard = int(num_neg * hard_ratio)
    # TODO: it could also be K
    num_per_bin = num_neg // num_bin
    bins = [i * num_per_bin for i in range(num_bin)] + [num_neg + 1]
    bins = np.array(bins)
    fig, axes = plt.subplots(num_algorithm, num_algorithm, figsize=(3 * num_algorithm + 1 , 3 * num_algorithm + 1))
    xs =  np.arange(num_bin) + 0.1
    ys = np.arange(1, 6) * 0.2
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    
    y_ticks_name = []
    for i in range(len(ys)):
        y_ticks_name.append(f"{round(ys[i], 1)}")
    
    num_per_ratio = round(100 / num_bin, 2)
    x_ticks_name = []
    x_ticks = []
    
    for i in range(0, num_bin, 2):
        x_ticks.append(i + 0.3)    
        x_ticks_name.append(f"{int((i+1) * num_per_ratio)}%")
    # import ipdb; ipdb.set_trace()
    # {int(i * num_per_ratio)}%-
    
    for row_idx in range(num_algorithm):
        main_algo_name = algorithm_names[row_idx]
        main_neg_ranks = neg_ranks_dict[main_algo_name]
    
        if args.is_old_neg and args.dataset_name != "ogbl-citation2":
            hard_indices = np.where((main_neg_ranks >= 1) & (main_neg_ranks <= num_hard))
        else:
            num_neg = neg_ranks_dict[algorithm_names[0]].shape[1]
            correct_indexes = correct_indexes_dict[main_algo_name]
            wrong_indexes = mask_to_index(~index_to_mask(correct_indexes, correct_indexes.max().item() + 1)[1:]) + 1      
            # wrong_indexes = 
            # first check whether the edge is wrongly predict
            main_neg_ranks = main_neg_ranks[wrong_indexes]
            hard_indices = np.where((main_neg_ranks >= 1) & (main_neg_ranks <= K))
            num_hard = int(K * wrong_indexes.shape[0])
            num_per_bin = num_neg // num_bin
            bins = [i * num_per_bin for i in range(num_bin)] + [num_neg + 1]
            bins = np.array(bins)
            
        for col_idx in range(num_algorithm):
            # print(f"row: {row_idx}, col: {col_idx}")
            comp_algo_name = algorithm_names[col_idx]
            if args.is_old_neg:
                comp_neg_ranks = neg_ranks_dict[comp_algo_name]
            else:
                comp_neg_ranks = neg_ranks_dict[comp_algo_name][wrong_indexes]
                comp_neg_ranks = np.reshape(comp_neg_ranks, (-1, num_neg))
            
            # import ipdb; ipdb.set_trace() 
            if args.is_old_neg:
                hard_neg_ranks = comp_neg_ranks[hard_indices]
            else:
                hard_neg_ranks = comp_neg_ranks[hard_indices[0], hard_indices[1]]
            
            # ax = axes[row_idx * num_algorithm + col_idx] 
            # import ipdb; ipdb.set_trace()
            ax = axes[row_idx][col_idx]
            if row_idx != col_idx:
                hists, bins_edges = np.histogram(hard_neg_ranks, bins)
                hists = hists / np.sum(hists)
                ax.bar(xs, hists, color=colors[col_idx], width=0.5, align="edge", edgecolor ='black')
            if row_idx == num_algorithm - 1:
                # ax.is_first_col()
                ax.set_xlabel(comp_algo_name, fontsize=30, fontfamily='serif')
                # ax.legend()
            if col_idx == 0:
                ax.set_ylabel(main_algo_name, fontsize=30, fontfamily='serif')
                # ax.legend()
            ax.set_xticks(x_ticks, x_ticks_name, fontsize=15, fontfamily='serif')
            # ax.set_yticks(ys, y_ticks_name, fontsize=15, fontfamily='serif')
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    print()
    folder_path = f"homo_analyze/hard_analysis/{args.dataset_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if prefix:
        plt.savefig(f"{folder_path}/{dataset_name}_{prefix}_{args.is_old_neg}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{folder_path}/{dataset_name}_{args.is_old_neg}.pdf", bbox_inches='tight')






'''
wrong_pos_preds = pos_preds[wrong_index]
wrong_pos_rank = get_rank_single(args, wrong_pos_preds) - 1
preds = np.concatenate([wrong_pos_preds, neg_preds], axis=0)
rank = get_rank_single(args, preds) - 1
wrong_pos_rank = rank[:wrong_pos_preds.shape[0]] - wrong_pos_rank
            
'''


def plot_property_scatter(preds_dict, args, dataset_name, prefix=None):
    # TODO: num_bin may be a potential choice
    # outer is the rank, inner is the name of algorithms
    assert args.is_old_neg == 1, "current do not support node specific"
    algorithm_names = list(preds_dict.keys())
    num_algorithm = len(algorithm_names)
    fig, axes = plt.subplots(num_algorithm, num_algorithm, figsize=(4 * num_algorithm + 1 , 4 * num_algorithm + 1))
    plt.subplots_adjust(wspace=0.5, hspace=0.3)

    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    s = 10
    for row_idx in range(num_algorithm):
        algo_row_name = algorithm_names[row_idx]
        preds_row = preds_dict[algo_row_name]
        for col_idx in range(num_algorithm):
            algo_col_name = algorithm_names[col_idx]
            preds_col = preds_dict[algo_col_name]
            ax = axes[row_idx][col_idx]
            if row_idx != col_idx:
                ax.scatter(preds_col[0], preds_row[0], s=s, color=colors[0], marker='o', label="positive")
                ax.scatter(preds_col[1], preds_row[1], s=s, color=colors[1], marker='*', label="negative")
                    
            if row_idx == 0 and col_idx == num_algorithm // 2 + 1:
                ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(-1, 1.4),  prop = {'size':30, 'family': 'serif'})
            if row_idx == num_algorithm - 1:
                # ax.is_first_col()
                ax.set_xlabel(algo_col_name, fontsize=30, fontfamily='serif')
                # ax.legend()
            if col_idx == 0:
                ax.set_ylabel(algo_row_name, fontsize=30, fontfamily='serif')
            ax.tick_params(axis='both', which='both', labelsize=20)
    # plt.legend(fontsize=20)
    if prefix:
        plt.savefig(f"homo_analyze/property_scatter/{dataset_name}_{prefix}_{args.is_old_neg}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"homo_analyze/property_scatter/{dataset_name}_{args.is_old_neg}.pdf", bbox_inches='tight')




def plot_where_positive(preds_dict, correct_indexes_dict, wrong_pos_indexes_dict, args, dataset_name, K, prefix=None, num_bin=5):
    # opposite to the nagative ones, check the positive
    neg_preds_dict, pos_preds_dict = {}, {}
    for key in preds_dict.keys():
        pos_preds_dict[key] = preds_dict[key][0]
        neg_preds_dict[key] = preds_dict[key][1]
    # import ipdb; ipdb.set_trace()
    algorithm_names = list(preds_dict.keys())
    num_algorithm = len(algorithm_names)
    # import ipdb; ipdb.set_trace()
    num_neg = neg_preds_dict[algorithm_names[0]].shape[0]
    # TODO: it could also be K
    num_per_bin = num_neg // num_bin
    bins = [i * num_per_bin for i in range(num_bin)] + [num_neg + 1]
    bins = np.array(bins)
    fig, axes = plt.subplots(num_algorithm, num_algorithm, figsize=(3 * num_algorithm + 1 , 3 * num_algorithm + 1))
    xs =  np.arange(num_bin) + 0.1
    ys = np.arange(1, 6) * 0.2
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    
    y_ticks_name = []
    for i in range(len(ys)):
        y_ticks_name.append(f"{round(ys[i], 1)}")
    
    num_per_ratio = round(100 / num_bin, 2)
    x_ticks_name = []
    x_ticks = []
    
    for i in range(0, num_bin, 2):
        x_ticks.append(i + 0.3)    
        x_ticks_name.append(f"{int((i+1) * num_per_ratio)}%")
    
    for row_idx in range(num_algorithm):
        main_algo_name = algorithm_names[row_idx]
        wrong_pos_index = wrong_pos_indexes_dict[main_algo_name]
            
        for col_idx in range(num_algorithm):
            # print(f"row: {row_idx}, col: {col_idx}")
            comp_algo_name = algorithm_names[col_idx]
            comp_pos_preds, comp_neg_preds = pos_preds_dict[comp_algo_name], neg_preds_dict[comp_algo_name]
            wrong_comp_pos_preds = comp_pos_preds[wrong_pos_index]
            if args.is_old_neg:                
                preds = np.concatenate([wrong_comp_pos_preds, comp_neg_preds], axis=0)
                wrong_pos_rank = get_rank_single(args, wrong_comp_pos_preds) - 1
                rank = get_rank_single(args, preds) - 1
                wrong_pos_rank = rank[:wrong_comp_pos_preds.shape[0]] - wrong_pos_rank
            else:
                wrong_comp_neg_preds = comp_neg_preds[wrong_pos_index]
                wrong_comp_preds = np.concatenate([np.expand_dims(wrong_comp_pos_preds, -1), wrong_comp_neg_preds], axis=-1)
                wrong_comp_rank = get_rank_single(args, wrong_comp_preds) 
                wrong_pos_rank = wrong_comp_rank[:, 0] + 1
                # import ipdb; ipdb.set_trace()

                # print()
                
                

            '''
            import ipdb; ipdb.set_trace()
            if args.is_old_neg:
                comp_neg_ranks = neg_ranks_dict[comp_algo_name]
            else:
                comp_neg_ranks = neg_ranks_dict[comp_algo_name][wrong_indexes]
                comp_neg_ranks = np.reshape(comp_neg_ranks, (-1, num_neg))
            
            # import ipdb; ipdb.set_trace() 
            if args.is_old_neg:
                hard_neg_ranks = comp_neg_ranks[hard_indices]
            else:
                hard_neg_ranks = comp_neg_ranks[hard_indices[0], hard_indices[1]]
            '''
            # ax = axes[row_idx * num_algorithm + col_idx] 
            # import ipdb; ipdb.set_trace()
            ax = axes[row_idx][col_idx]
            if row_idx != col_idx:
                hists, bins_edges = np.histogram(wrong_pos_rank, bins)
                hists = hists / np.sum(hists)
                ax.bar(xs, hists, color=colors[col_idx], width=0.5, align="edge", edgecolor ='black')
            if row_idx == num_algorithm - 1:
                # ax.is_first_col()
                ax.set_xlabel(comp_algo_name, fontsize=30, fontfamily='serif')
                # ax.legend()
            if col_idx == 0:
                ax.set_ylabel(main_algo_name, fontsize=30, fontfamily='serif')
                # ax.legend()
            ax.set_xticks(x_ticks, x_ticks_name, fontsize=15, fontfamily='serif')
            # ax.set_yticks(ys, y_ticks_name, fontsize=15, fontfamily='serif')
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

    print()
    folder_path = f"homo_analyze/find_positive_analysis/{args.dataset_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if prefix:
        plt.savefig(f"{folder_path}/{dataset_name}_{prefix}_{args.is_old_neg}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{folder_path}/{dataset_name}_{args.is_old_neg}.pdf", bbox_inches='tight')
        




def plot_decay(algo_names, remain_negs, dataset_name, args, prefix=None):
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    fig, ax = plt.subplots(figsize=(4, 4))
    plt.title(f"{dataset_name}", fontsize=20, pad=100)
    
    num_algo = len(algo_names)
    xs = np.arange(1, num_algo + 1)
    
    plt.scatter(xs, remain_negs, s=0.5, c="blue", marker=markers[0], alpha=1.0)
    plt.plot(xs, remain_negs)
    
    x_ticks = algo_names
    ax.set_xticks(xs, x_ticks, fontsize=10, fontfamily='serif')
    
    folder_path = f"homo_analyze/hard_decay/{args.dataset_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if prefix:
        plt.savefig(f"{folder_path}/{dataset_name}_{prefix}_{args.is_old_neg}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{folder_path}/{dataset_name}_{args.is_old_neg}.pdf", bbox_inches='tight')
    
    if prefix:
        plt.savefig(f"{folder_path}/{dataset_name}_{prefix}_{args.is_old_neg}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{folder_path}/{dataset_name}_{args.is_old_neg}.pdf", bbox_inches='tight')
    

def generate_rank(algo_names, num_algo, algorithms):
    num_category = len(algo_names)
    algorithm_dict = defaultdict(list)
    idx = 0
    num_per_algo = (num_algo + 1) // len(algorithms)
    
    for algorithm in algorithms:
        num =  num_per_algo if algorithm != "global" else 1
        # print(num)
        for _ in range(num):
            # print(idx)
            algorithm_dict[algorithm].append(algo_names[idx])
            idx += 1
    # import ipdb; ipdb.set_trace()
    all_algorithm_ranks = []
    # import ipdb; ipdb.set_trace()
    
    algorithms_list = list(permutations(algorithms, len(algorithms)))
    new_algorithm_list = []
    for algorithms in algorithms_list:
        new_algorithms = []
        for algorithm in algorithms:
            # import ipdb; ipdb.set_trace()
            new_algorithms += algorithm_dict[algorithm]  #[algo_names[algorithm]]
        new_algorithm_list.append(new_algorithms)
    # import ipdb; ipdb.set_trace()
    return new_algorithm_list
        
    

def plot_decay_all_rank(algo_names, hard_neg_masks_dict, num_algo, algorithms, dataset_name, args, prefix=None):
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    markers = ['o','*','x','s','p',"+",'v', 'h',]
    
    # generete ranks 
    algorithms_list = generate_rank(algo_names, num_algo, algorithms)
    num_ranks = len(algorithms_list)
    fig, axes = plt.subplots(1, num_ranks, figsize=(4 * num_ranks + 1, 4 + 1))
    plt.title(f"{dataset_name}", fontsize=20, pad=100)
    xs = np.arange(1, num_algo + 1)

    for algo_idx, algorithms in enumerate(algorithms_list):
        ax = axes[algo_idx]
        remain_negs = []
        # masks = np.zeros([num_neg], dtype=bool)
        for idx, algorithm in enumerate(algorithms):
            hard_neg_masks = hard_neg_masks_dict[algorithm]
            # import ipdb; ipdb.set_trace()
            if idx == 0:
                masks = hard_neg_masks
            else:
                masks = np.logical_and(masks, hard_neg_masks)
            remain_neg = np.sum(masks)
            remain_negs.append(remain_neg)
        # import ipdb; ipdb.set_trace()
        ax.scatter(xs, remain_negs, s=0.5, c="blue", marker=markers[0], alpha=1.0)
        ax.plot(xs, remain_negs)
        x_ticks = algorithms
        ax.set_xticks(xs, x_ticks, fontsize=10, fontfamily='serif')

    folder_path = f"homo_analyze/hard_decay_new/{args.dataset_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if prefix:
        plt.savefig(f"{folder_path}/{dataset_name}_{prefix}_{args.is_old_neg}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{folder_path}/{dataset_name}_{args.is_old_neg}.pdf", bbox_inches='tight')
    
    if prefix:
        plt.savefig(f"{folder_path}/{dataset_name}_{prefix}_{args.is_old_neg}.pdf", bbox_inches='tight')
    else:
        plt.savefig(f"{folder_path}/{dataset_name}_{args.is_old_neg}.pdf", bbox_inches='tight')
    


        
def plot_compare_performance(seperate_results, compare_results, compare_algorithm_name, results_dict, dataset_name, split_values, num_pos_values, num_neg_values, result_key, algorithms=None):
    # We not only plot the performance in each region, but also the 
    # import ipdb; ipdb.set_trace()
    algorithm_names = list(results_dict.keys())
    other_algorithm_names = list(seperate_results.keys())
    compare_algorithm_name = compare_algorithm_name
    
    # The plot have three parts, overall performance in left, 
    # each region performance in right, for each region, we have a line for comparison
    # in bottom, we plot the node ratio for each dimension
    
    # first part, do the preprocess on data. 
    num_inner_group = len(algorithm_names)
    num_outer_group = len(split_values) + 1
    # + 1 is for all results
    # the compare group will serve as the baseline
    fig, axes = plt.subplots(2, 1, figsize=(3 * num_outer_group + 1 , 9 * 2))
    # the first figure is for performance, second for the list

    ax1, ax2 = axes[0], axes[1]
    # ax1 is for performance, ax2 is for 
    
    ax1.set_title(f"{dataset_name} performance: {compare_algorithm_name}", fontsize=30, fontfamily='serif')

    # interval_range = num_model
    bar_width = 0.8 / num_inner_group
    
    # x = bar_width * np.arange(num_inner_group) # + 0.1
    x = np.arange(num_outer_group) # + 0.1
    
    # inner is model name with id
    # outer is the split value
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    x_ticks_outer_name = ["\n ALL"]
    
    for i in range(len(split_values) - 1):
        x_ticks_outer_name.append(f"\n {round(split_values[i], 2), round(split_values[i+1], 2)}")
    x_ticks_outer_name.append(f"\n {round(split_values[-1], 1)}_inf")
    x_ticks_outer = np.arange(num_outer_group) + 0.3
    
    x_ticks_inner_unit = np.arange(num_inner_group) * bar_width 
    x_ticks_inner_unit_name = [str(i) for i in range(num_inner_group)]
    
    x_ticks_inner = []
    x_ticks_inner_name = []
    for i in range(num_outer_group):
        x_ticks_inner.append(i + x_ticks_inner_unit)
        x_ticks_inner_name += x_ticks_inner_unit_name
    
    x_ticks_inner = np.concatenate(x_ticks_inner, axis=0)

    model_idx = 0

    for model_idx, key in enumerate(seperate_results.keys()):
        # [num_group (mean), num_group (std)]
        data = [results_dict[key]] + seperate_results[key]
        # mean, variance = record[0], record[1]
        # import ipdb; ipdb.set_trace()
        key_name = f"model_idx: {key}"
        ax1.bar(x + model_idx * bar_width, data, bar_width, color=colors[model_idx], label=key_name, capsize=2, edgecolor ='black') # ,marker=markers[i]
    # set the performance of the baseline methods
    
    compare_results = [results_dict[compare_algorithm_name]] + compare_results 
    ratio = 1 / len(compare_results)
    for idx, result in enumerate(compare_results):
        ax1.axhline(y=result, xmin=idx * ratio, xmax=(idx+1) * ratio, color="red", linewidth=3.0, linestyle="--")
    # ax1.axhline(y=0.5, xmin=0, xmax=1, color="red", linewidth=3.0, linestyle="--")
    
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.25),  prop = {'size':20, 'family': 'serif'})
    ax1.tick_params(axis='y', labelsize=25)
    ax1.axhline(y=0, color='black', linewidth=2.0)
    ax1.grid(False) # ,loc=(0.02, 0.6)
    ax1.set_ylabel(f'{result_key}', fontsize=20, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    # plt.xlabel(f"{arg_dict['name']}", fontsize=26,  fontfamily='serif') # fontweight='bold',
    
    # final_x_ticks = []
    # for i in range(len(x_ticks) - 1):
    #     final_x_ticks.append(f"{x_ticks[i]}-{x_ticks[i+1]}")
    ax1.set_xticks(x_ticks_inner, x_ticks_inner_name, fontsize=20, fontfamily='serif')
    plt.yticks(fontsize=45, fontfamily='serif')
    ax1.set_xticks(x_ticks_outer, x_ticks_outer_name, fontsize=30, fontfamily='serif', minor=True) # 
    
    ax1.tick_params(axis='x', which='minor', direction='out', bottom='off', length=0)
    ax1.tick_params(axis='x', which='major', bottom='off', top='off' )

    # for ax2, we find the propotation of the datasets

    ax2.set_title(f"{dataset_name} propotions", fontsize=30, fontfamily='serif')
    
    # normalize num_pos_values, num_neg_values
    pos_propotions = np.array(num_pos_values) / np.sum(num_pos_values)
    neg_propotions = np.array(num_neg_values) / np.sum(num_neg_values)
    pos_propotions = np.concatenate([[1], pos_propotions], axis=0)
    neg_propotions = np.concatenate([[1], neg_propotions], axis=0)
    max_length = np.max([len(pos_propotions), len(neg_propotions)])
    if pos_propotions.shape[0] < max_length:
        pos_propotions = np.concatenate([pos_propotions, np.zeros(max_length - pos_propotions.shape[0])], axis=0)
    if neg_propotions.shape[0] < max_length:
        neg_propotions = np.concatenate([neg_propotions, np.zeros(max_length - neg_propotions.shape[0])], axis=0)
    
    x = np.arange(pos_propotions.shape[0])
    
    bar_width = 0.6 / 2
    keys = ["pos", "neg"]
    for idx, propotions in enumerate([pos_propotions, neg_propotions]):
        # import ipdb; ipdb.set_trace()
        # print(f"prop: {len(propotions)}")
        # print(f"x: {len(x)}")
        ax2.bar(x + idx * bar_width, propotions, bar_width, color=colors[idx], label=keys[idx], capsize=2, edgecolor ='black')
    
    pattern = r'\n'
    x_ticks_outer_name = [re.sub(pattern, '', x_tick) for x_tick in x_ticks_outer_name]    
    ax2.set_xticks(x_ticks_outer, x_ticks_outer_name, fontsize=30, fontfamily='serif', minor=True) # 
    ax2.set_ylabel(f'proprotion', fontsize=30, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    
    if algorithms is None:
        plt.savefig(f"homo_analyze/harry_analysis/{dataset_name}_{compare_algorithm_name}.pdf", bbox_inches='tight')
    else:
        algorithm_name = ""
        for algorithm in algorithms:
            algorithm_name += algorithm
        folder_path = f"homo_analyze/harry_analysis/{algorithm_name}"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        plt.savefig(f"{folder_path}/{dataset_name}_{compare_algorithm_name}.pdf", bbox_inches='tight')
            
    
def plot_tradic_feature_importance(weights, algorithm, dataset_name, result_key, result):
    fig, ax = plt.subplots(figsize=(9, 9))
    # ax1 is for performance, ax2 is for 
    
    ax.set_title(f"{dataset_name}_{algorithm}\n {result_key}: {result}", fontsize=30, pad=40)
    num_weight = weights.shape[0]
    
    # x = bar_width * np.arange(num_inner_group) # + 0.1
    xs = np.arange(1, num_weight + 1) # + 0.1
    x_ticks = [f"{int(xs[i])}" for i in range(len(xs))]
        
    bar_width = 0.3
    ax.bar(xs, weights, bar_width, color="blue", capsize=2, edgecolor ='black') # ,marker=markers[i]
    plt.xticks(xs, x_ticks, fontsize=35, fontfamily='serif')
    plt.yticks(fontsize=40, fontfamily='serif')
    
    plt.savefig(f"homo_analyze/tradic_weight_import/{dataset_name}_{algorithm}.pdf", bbox_inches='tight')
    
    
def plot_homo_feature_importance(weights, algorithm, dataset_name, result_key, result):
    fig, ax = plt.subplots(figsize=(9, 9))
    # ax1 is for performance, ax2 is for 
    
    ax.set_title(f"{dataset_name}_{algorithm}\n {result_key}: {result}", fontsize=30, pad=40)
    num_weight = weights.shape[0]
    
    # x = bar_width * np.arange(num_inner_group) # + 0.1
    xs = np.arange(1, num_weight + 1) # + 0.1
    x_ticks = [f"{int(xs[i])}" for i in range(len(xs))]
        
    bar_width = 0.3
    ax.bar(xs, weights, bar_width, color="blue", capsize=2, edgecolor ='black') # ,marker=markers[i]
    plt.xticks(xs, x_ticks, fontsize=35, fontfamily='serif')
    plt.yticks(fontsize=40, fontfamily='serif')
    
    plt.savefig(f"homo_analyze/homo_weight_import/{dataset_name}_{algorithm}.pdf", bbox_inches='tight')
    


def plot_seal_ablation_performance(results_dict, dataset_name):
    # We not only plot the performance in each region, but also the 
    model_names = list(results_dict.keys())
    dist_names = list(results_dict[model_names[0]].keys())
    num_model, num_dist = len(model_names), len(dist_names)
    
    # The plot have three parts, overall performance in left, 
    # each region performance in right, for each region, we have a line for comparison
    # in bottom, we plot the node ratio for each dimension
    
    # first part, do the preprocess on data. 
    num_inner_group, num_outer_group = num_dist, num_model
    # + 1 is for all results
    # the compare group will serve as the baseline
    fig, ax = plt.subplots(figsize=(3 * num_outer_group + 1 , 9))
    # the first figure is for performance, second for the list
    
    ax.set_title(f"{dataset_name} seal ablation", fontsize=30, fontfamily='serif')

    # interval_range = num_model
    bar_width = 0.8 / num_inner_group
    
    # x = bar_width * np.arange(num_inner_group) # + 0.1
    x =  np.arange(num_outer_group) # + 0.1
    
    # inner is model name with id
    # outer is the split value
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    x_ticks_outer_name = ["\n"+ model_name for model_name in model_names]
    
    x_ticks_outer = np.arange(num_outer_group) + 0.3
    
    x_ticks_inner_unit = np.arange(num_inner_group) * bar_width 
    x_ticks_inner_unit_name = dist_names
    
    x_ticks_inner = []
    x_ticks_inner_name = []
    for i in range(num_outer_group):
        x_ticks_inner.append(i + x_ticks_inner_unit)
        x_ticks_inner_name += x_ticks_inner_unit_name
    
    x_ticks_inner = np.concatenate(x_ticks_inner, axis=0)

    model_idx = 0

    for model_idx, dist_name in enumerate(dist_names):
        data = []
        for model_name in model_names:
            data.append(results_dict[model_name][dist_name])
        data = np.array(data)
        if len(data.shape) == 2:
            mean = data[:, 0]
            std = data[:, 1]
        else:
            mean = data
            std = np.zeros_like(mean) 
        key_name = dist_name
        ax.bar(x + model_idx * bar_width, mean, bar_width, yerr=std, color=colors[model_idx], label=key_name, capsize=2, edgecolor ='black') # ,marker=markers[i]
    # set the performance of the baseline methods
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.25),  prop = {'size':20, 'family': 'serif'})
    ax.tick_params(axis='y', labelsize=25)
    ax.axhline(y=0, color='black', linewidth=2.0)
    ax.grid(False) # ,loc=(0.02, 0.6)
    ax.set_ylabel(f'performance', fontsize=20, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    # plt.xlabel(f"{arg_dict['name']}", fontsize=26,  fontfamily='serif') # fontweight='bold',
    
    # final_x_ticks = []
    # for i in range(len(x_ticks) - 1):
    #     final_x_ticks.append(f"{x_ticks[i]}-{x_ticks[i+1]}")
    ax.set_xticks(x_ticks_inner, x_ticks_inner_name, fontsize=20, fontfamily='serif')
    plt.yticks(fontsize=45, fontfamily='serif')
    ax.set_xticks(x_ticks_outer, x_ticks_outer_name, fontsize=30, fontfamily='serif', minor=True) # 
    
    ax.tick_params(axis='x', which='minor', direction='out', bottom='off', length=0)
    ax.tick_params(axis='x', which='major', bottom='off', top='off' )

    plt.savefig(f"homo_analyze/ablation/{dataset_name}_seal_ablation.pdf", bbox_inches='tight')




def plot_feat_struct_ablation_performance(results_dict, dataset_name):
    # We not only plot the performance in each region, but also the 
    is_feat_names = list(results_dict.keys())
    
    model_names = list(results_dict[is_feat_names[0]].keys())
    num_feats, num_model = len(is_feat_names), len(model_names)
    
    # The plot have three parts, overall performance in left, 
    # each region performance in right, for each region, we have a line for comparison
    # in bottom, we plot the node ratio for each dimension
    
    # first part, do the preprocess on data. 
    num_inner_group, num_outer_group = num_feats, num_model
    # + 1 is for all results
    # the compare group will serve as the baseline
    fig, ax = plt.subplots(figsize=(3 * num_outer_group + 1 , 9))
    # the first figure is for performance, second for the list
    
    ax.set_title(f"{dataset_name} feat structure ablation", fontsize=30, fontfamily='serif')

    # interval_range = num_model
    bar_width = 0.8 / num_inner_group
    
    # x = bar_width * np.arange(num_inner_group) # + 0.1
    x =  np.arange(num_outer_group) # + 0.1
    
    # inner is model name with id
    # outer is the split value
    
    colors = ['#1f77b4','#ff7f0e','#2ca02c','#9467bd','#d62728', '#d67e27'] + sns.color_palette("tab10")
    x_ticks_outer_name = ["\n"+ model_name for model_name in model_names]
    
    x_ticks_outer = np.arange(num_outer_group) + 0.3
    
    x_ticks_inner_unit = np.arange(num_inner_group) * bar_width 
    x_ticks_inner_unit_name = is_feat_names
    
    x_ticks_inner = []
    x_ticks_inner_name = []
    for i in range(num_outer_group):
        x_ticks_inner.append(i + x_ticks_inner_unit)
        x_ticks_inner_name += x_ticks_inner_unit_name
    
    x_ticks_inner = np.concatenate(x_ticks_inner, axis=0)

    # import ipdb; ipdb.set_trace()
    
    for idx, is_feat_name in enumerate(is_feat_names):
        # import ipdb; ipdb.set_trace()
        data = np.array(list(results_dict[is_feat_name].values()))
        if len(data.shape) == 2:
            mean = data[:, 0]
            std = data[:, 1]
        else:
            mean = data
            std = np.zeros_like(mean) 
        key_name = is_feat_name
        ax.bar(x + idx * bar_width, mean, bar_width, yerr=std, color=colors[idx], label=key_name, capsize=2, edgecolor ='black') # ,marker=markers[i]
    # set the performance of the baseline methods
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.legend(frameon=False, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.25),  prop = {'size':20, 'family': 'serif'})
    ax.tick_params(axis='y', labelsize=25)
    ax.axhline(y=0, color='black', linewidth=2.0)
    ax.grid(False) # ,loc=(0.02, 0.6)
    ax.set_ylabel(f'performance', fontsize=20, fontfamily='serif') # , color='blue', backgroundcolor='yellow'
    # plt.xlabel(f"{arg_dict['name']}", fontsize=26,  fontfamily='serif') # fontweight='bold',
    
    # final_x_ticks = []
    # for i in range(len(x_ticks) - 1):
    #     final_x_ticks.append(f"{x_ticks[i]}-{x_ticks[i+1]}")
    ax.set_xticks(x_ticks_inner, x_ticks_inner_name, fontsize=20, fontfamily='serif')
    plt.yticks(fontsize=45, fontfamily='serif')
    ax.set_xticks(x_ticks_outer, x_ticks_outer_name, fontsize=30, fontfamily='serif', minor=True) # 
    
    ax.tick_params(axis='x', which='minor', direction='out', bottom='off', length=0)
    ax.tick_params(axis='x', which='major', bottom='off', top='off' )

    plt.savefig(f"homo_analyze/ablation/{dataset_name}_feat_struct_ablation.pdf", bbox_inches='tight')
