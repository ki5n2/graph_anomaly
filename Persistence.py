#%%
import os
import re
import gc
import sys
import json
import math
import time
import wandb
import torch
import random
import argparse
import numpy as np
import gudhi as gd
import seaborn as sns
import os.path as osp
import torch.nn as nn
import numpy.typing as npt
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch_geometric.utils as utils

from torch.nn import init
from typing import List, Tuple, Dict, Any
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, OneCycleLR
from torch_geometric.utils import to_networkx, get_laplacian, to_dense_adj, to_dense_batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from scipy.stats import linregress
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, silhouette_score, silhouette_samples, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, LeaveOneOut
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph, KernelDensity
from sklearn.manifold import SpectralEmbedding
from sklearn.preprocessing import StandardScaler

from functools import partial
from scipy.linalg import eigh
from multiprocessing import Pool

from modules.loss import loss_cal
from modules.utils import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original, split_batch_graphs, process_batch_graphs, scott_rule_bandwidth, loocv_bandwidth_selection

import networkx as nx


#%%
BATCH_SIZE = 300
TEST_BATCH_SIZE = 128
NUM_EPOCHS = 1000
LEARNING_RATE = 0.001
RANDOM_SEED = 0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


#%%
def create_balanced_split(dataset_name):
    path = 'dataset'
    dataset = TUDataset(path, name=dataset_name)
    
    # 전체 데이터의 레이블 분포 확인
    normal_indices = []
    anomaly_indices = []
    for idx in range(len(dataset)):
        if dataset[idx].y.item() == 0:
            normal_indices.append(idx)
        else:
            anomaly_indices.append(idx)
    
    print(f"Original distribution:")
    print(f"Normal samples: {len(normal_indices)}")
    print(f"Anomaly samples: {len(anomaly_indices)}")
    
    # 더 적은 클래스의 샘플 수에 맞춰서 다운샘플링
    min_samples = min(len(normal_indices), len(anomaly_indices))
    
    # 각 클래스에서 동일한 수의 샘플을 무작위로 선택
    np.random.shuffle(normal_indices)
    np.random.shuffle(anomaly_indices)
    
    selected_normal = normal_indices[:min_samples]
    selected_anomaly = anomaly_indices[:min_samples]
    
    # 훈련/테스트 세트로 50:50 분할
    n_train = min_samples // 2
    
    train_indices = np.concatenate([
        selected_normal[:n_train],
        selected_anomaly[:n_train]
    ])
    test_indices = np.concatenate([
        selected_normal[n_train:2*n_train],
        selected_anomaly[n_train:2*n_train]
    ])
    
    # 인덱스 섞기
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    print(f"\nBalanced split:")
    print(f"Train set size: {len(train_indices)} (Normal: {n_train}, Anomaly: {n_train})")
    print(f"Test set size: {len(test_indices)} (Normal: {n_train}, Anomaly: {n_train})")
    
    return (train_indices, test_indices)


def get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset_ = TUDataset(path, name=dataset_name)
        
    prefix = os.path.join(path, dataset_name, 'raw', dataset_name)

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')
        
    node_attrs = np.array(node_attrs)

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')
    
    dataset = []
    node_idx = 0
    for i in range(len(dataset_)):
        old_data = dataset_[i]
        num_nodes = old_data.num_nodes
        new_x = torch.tensor(node_attrs[node_idx:node_idx+num_nodes], dtype=torch.float)
        node_label_graph = torch.tensor(node_labels[node_idx:node_idx+num_nodes], dtype=torch.float)   
                
        if dataset_name != 'NCI1':
            new_data = Data(x=new_x, edge_index=old_data.edge_index, y=old_data.y, node_label = node_label_graph)
        else:
            new_data = Data(x=old_data.x, edge_index=old_data.edge_index, y=old_data.y, node_label = node_label_graph)
        
        dataset.append(new_data)
        node_idx += num_nodes

    dataset_num_features = dataset[0].x.shape[1]
    
    data_list = []
    label_list = []

    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())

    (train_index, test_index) = split
    data_train = [data_list[i] for i in train_index]
    data_test = [data_list[i] for i in test_index]

    max_nodes = max([dataset[i].num_nodes for i in range(len(dataset))])
    dataloader = DataLoader(data_train, batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'num_test':len(data_test), 'num_edge_feat':0, 'max_nodes':max_nodes, 'max_node_label':num_unique_node_labels}
    loader_dict = {'train': dataloader, 'test': dataloader_test}
    
    return loader_dict, meta


#%%
def train_and_evaluate():
    set_seed(RANDOM_SEED)
    
    # 데이터셋 로드
    dataset_name = 'COX2'
    split = create_balanced_split(dataset_name)
    if dataset_name == 'AIDS' or dataset_name == 'NCI1' or dataset_name == 'DHFR':
        dataset_AN = True
    else:
        dataset_AN = False
        
    loaders, meta = get_data_loaders_TU(dataset_name, batch_size=BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, split=split, dataset_AN=dataset_AN)
    
    train_loader = loaders['train']
    test_loader = loaders['test']
    
    # 모델 초기화
    model = PersistenceClassifier().to(DEVICE)
    criterion = nn.BCELoss()  # 일반 BCELoss 사용
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels_list = []
        
        for data in train_loader:
            data = process_batch_graphs(data)
            features = data.true_stats.to(DEVICE)
            labels = data.y.float().to(DEVICE)
            
            # 특징값 확인 (첫 배치, 첫 에폭에만)
            if epoch == 0 and len(train_preds) == 0:
                print("Feature shape:", features.shape)
                print("Feature sample:", features[0])
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())
        
        # Training metrics
        train_preds = np.array(train_preds)
        train_labels_list = np.array(train_labels_list)
        train_auroc = roc_auc_score(train_labels_list, train_preds)
        
        # Evaluation
        model.eval()
        test_preds = []
        test_labels = []
        test_loss = 0
        
        with torch.no_grad():
            for data in test_loader:
                data = process_batch_graphs(data)
                features = data.true_stats.to(DEVICE)
                labels = data.y.float().to(DEVICE)
                
                outputs = model(features)
                loss = criterion(outputs.squeeze(), labels)
                test_loss += loss.item()
                
                test_preds.extend(outputs.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        test_preds = np.array(test_preds)
        test_labels = np.array(test_labels)
        
        auroc = roc_auc_score(test_labels, test_preds)
        auprc = average_precision_score(test_labels, test_preds)

        if (epoch + 1) % 10 == 0:
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
            print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train AUROC: {train_auroc:.4f}")
            print(f"Test Loss: {test_loss/len(test_loader):.4f}")
            print(f"Test AUROC: {auroc:.4f}")
            print(f"Test AUPRC: {auprc:.4f}")
            print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 예측값 분포 확인
            print("\nPrediction distribution:")
            print("Train predictions:", np.percentile(train_preds, [0, 25, 50, 75, 100]))
            print("Test predictions:", np.percentile(test_preds, [0, 25, 50, 75, 100]))


#%%
class PersistenceClassifier(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(input_dim),  # 특징 정규화 추가
            nn.Linear(input_dim, 64),   # 더 넓은 레이어
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.classifier(x)


#%%
if __name__ == "__main__":
    train_and_evaluate()
    