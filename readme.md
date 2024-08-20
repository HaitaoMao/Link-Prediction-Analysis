# Revisiting Link Predition: A Data Perspective

Link prediction, a fundamental task on graphs, has proven indispensable in various applications, e.g., friend recommendation, protein analysis, and drug interaction prediction. However, since datasets span a multitude of domains, they could have distinct underlying mechanisms of link formation. Evidence in existing literature underscores the absence of a universally best algorithm suitable for all datasets. In this paper, we endeavor to explore principles of link prediction across diverse datasets from a data-centric perspective. We recognize three fundamental factors critical to link prediction: local structural proximity, global structural proximity, and feature proximity. We then unearth relationships among those factors where (i) global structural proximity only shows effectiveness when local structural proximity is deficient. (ii) The incompatibility can be found between feature and structural proximity. Such incompatibility leads to GNNs for Link Prediction (GNN4LP) consistently underperforming on edges where the feature proximity factor dominates. Inspired by these new insights from a data perspective, we offer practical instruction for GNN4LP model design and guidelines for selecting appropriate benchmark datasets for more comprehensive evaluations.

## Introduction

This is a implementation of the code used for "Revisiting Link Prediction: A Data Perspective" https://openreview.net/pdf?id=m1oqEOAozQU which was accepted in ICLR 2024.

## Dataset and Preprocessing

Create a root level folder

```
./dataset
```

Datasets will automatically be downloaded to this folder provided you are connected to the internet.

## Running experiments

### Requirements

Dependencies (with python >= 3.9):
Main dependencies are

pytorch==1.13

torch_geometric==2.2.0

torch-scatter==2.1.1+pt113cpu

torch-sparse==0.6.17+pt113cpu

torch-spline-conv==1.2.2+pt113cpu


Example commands to install the dependencies in a new conda environment (tested on a Linux machine without GPU).

```
conda create --name ss python=3.9
conda activate ss
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install torch_geometric
pip install fast-pagerank wandb datasketch ogb
```


For GPU installation (assuming CUDA 11.8): 

```
conda create --name ss python=3.9
conda activate ss
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pytorch-sparse -c pyg
conda install pyg -c pyg
```

### Analsysis experiment 

To run experiments

```
cd data_analysis
conda activate ss
python runners/run_analysis.py --mode 1
```

### Baselined Experiments

To run experiments, the first choice is our benchmark repo at [here](https://github.com/Juanhui28/HeaRT). All the baseline results from this paper are from that repo.

You can also run some baselines in this repo.

```
conda activate ss
python runners/run.py --dataset_name Cora --model ELPH
python runners/run.py --dataset_name Cora --model BUDDY
python runners/run.py --dataset_name Citeseer --model ELPH
python runners/run.py --dataset_name Citeseer --model BUDDY
python runners/run.py --dataset_name Pubmed --max_hash_hops 3 --feature_dropout 0.2 --model ELPH
python runners/run.py --dataset_name Pubmed --max_hash_hops 3 --feature_dropout 0.2 --model BUDDY
python runners/run.py --dataset_name ogbl-collab --K 50 --lr 0.01 --feature_dropout 0.05 --add_normed_features 1 --label_dropout 0.1 --batch_size 2048 --year 2007 --model ELPH
python runners/run.py --dataset_name ogbl-collab --K 50 --lr 0.02 --feature_dropout 0.05 --add_normed_features 1 --cache_subgraph_features --label_dropout 0.1 --year 2007 --model BUDDY
python runners/run.py --dataset_name ogbl-ppa --label_dropout 0.1 --use_feature 0 --use_RA 1 --lr 0.03 --epochs 100 --hidden_channels 256 --cache_subgraph_features --add_normed_features 1 ----use_zero_one 1 model BUDDY
python runners/run.py --dataset ogbl-ddi --K 20 --train_node_embedding --propagate_embeddings --label_dropout 0.25 --epochs 150 --hidden_channels 256 --lr 0.0015 --num_negs 6 --use_feature 0 --sign_k 2 --batch_size 131072 --model ELPH
python runners/run.py --dataset ogbl-ddi --K 20 --train_node_embedding --propagate_embeddings --label_dropout 0.25 --epochs 150 --hidden_channels 256 --lr 0.0015 --num_negs 6 --use_feature 0 --sign_k 2 --cache_subgraph_features --batch_size 131072 --model BUDDY
python runners/run.py --dataset ogbl-citation2 --hidden_channels 128 --num_negs 5 --lr 0.0005 --sign_dropout 0.2 --feature_dropout 0.7 --label_dropout 0.8 --sign_k 3 --batch_size 261424 --eval_batch_size 522848 --cache_subgraph_features --model BUDDY
```

You may need to adjust 

```
--batch_size 
--num_workers
```

and 

```
--eval_batch_size
```

based on available (GPU) memory and CPU cores.

Most of the runtime of BUDDY is building hashes and subgraph features. If you intend to run BUDDY more than once, then set the flag

```
--cache_subgraph_features
```

to store subgraph features on disk and read them if previously cached.


## Cite us

If you found this work useful, please cite our paper

```
@inproceedings{mao2024revisiting,
  title={Revisiting Link Prediction: a data perspective},
  author={Mao, Haitao and Li, Juanhui and Shomer, Harry and Li, Bingheng and Fan, Wenqi and Ma, Yao and Zhao, Tong and Shah, Neil and Tang, Jiliang},
  booktitle={The Twelfth International Conference on Learning Representations}
  year={2024}
}
```