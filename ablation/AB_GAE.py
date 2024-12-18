#%%
'''IMPORTS'''
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
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, silhouette_score, silhouette_samples
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

# 메모리 설정
torch.cuda.empty_cache()
gc.collect()


#%%
def train(model, train_loader, recon_optimizer, epoch, dataset_name, device, cluster_centers=None):
    model.train()
    total_loss = 0
    num_sample = 0
    reconstruction_errors = []
    
    for data in train_loader:
        data = process_batch_graphs(data)
        data = data.to(device)
        recon_optimizer.zero_grad()
        x, edge_index, batch, num_graphs, true_stats = data.x, data.edge_index, data.batch, data.num_graphs, data.true_stats
        
        # Forward pass
        graph_embeddings, x_recon, stats_pred, adj_recon_list = model(x, edge_index, batch, num_graphs)
        
        if epoch % 5 == 0:
            graph_embeddings_np = graph_embeddings.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_cluster, random_state=random_seed)
            kmeans.fit(graph_embeddings_np)
            cluster_centers = kmeans.cluster_centers_
            
        loss = 0
        node_loss = 0
        edge_loss = 0
        start_node = 0
        
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            # 노드 특징 재구성 손실
            if dataset_name == 'AIDS':
                node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2
            else:
                node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
            node_loss += node_loss_
            
            # 엣지 재구성 손실
            graph_edges = edge_index[:, (edge_index[0] >= start_node) & (edge_index[0] < end_node)]
            graph_edges = graph_edges - start_node
            true_adj = torch.zeros((model.edge_recon.max_nodes, model.edge_recon.max_nodes), device=device)
            true_adj[graph_edges[0], graph_edges[1]] = 1
            true_adj = true_adj + true_adj.t()
            true_adj = (true_adj > 0).float()
            
            node_mask = torch.zeros_like(adj_recon_list[i], dtype=torch.bool)
            node_mask[:num_nodes, :num_nodes] = True
            edge_loss_ = torch.sum((adj_recon_list[i][node_mask] - true_adj[node_mask]) ** 2) / node_mask.sum()
            edge_loss += edge_loss_
            
            if epoch % 5 == 0:
                node_loss_scaled = node_loss_.item() * alpha
                graph_vec = graph_embeddings[i].detach().cpu().numpy()
                distances = cdist([graph_vec], cluster_centers, metric='euclidean')
                min_distance = distances.min().item() * gamma
                
                edge_loss__ = edge_loss_.item() * beta    
                
                reconstruction_errors.append({
                    'reconstruction': node_loss_scaled + edge_loss__,
                    'clustering': min_distance,
                    'type': 'train_normal'
                })
            
            start_node = end_node
            
        stats_loss = persistence_stats_loss(stats_pred, true_stats)
        
        # 손실 가중치 적용
        edge_loss = beta * edge_loss
        stats_loss = alpha * stats_loss

        loss = node_loss + edge_loss + stats_loss
        
        num_sample += num_graphs
        loss.backward()
        recon_optimizer.step()
        total_loss += loss.item()
        
    if epoch % 5 == 0:
        # 시각화
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        recon_errors = [point['reconstruction'] for point in reconstruction_errors]
        cluster_errors = [point['clustering'] for point in reconstruction_errors]
        
        # Normal scale plot
        ax1.scatter(recon_errors, cluster_errors, c='blue', alpha=0.6)
        ax1.set_xlabel('Reconstruction Error')
        ax1.set_ylabel('Clustering Distance')
        ax1.set_title(f'Training Error Distribution (Epoch {epoch})')
        ax1.grid(True)

        # Log scale plot
        ax2.scatter(recon_errors, cluster_errors, c='blue', alpha=0.6)
        ax2.set_xlabel('Reconstruction Error')
        ax2.set_ylabel('Clustering Distance')
        ax2.set_title(f'Training Error Distribution - Log Scale (Epoch {epoch})')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True)

        plt.tight_layout()
        save_path = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/plot/{dataset_name}_time_{current_time}/train_error_distribution_epoch_{epoch}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
    return total_loss / len(train_loader), num_sample, cluster_centers, reconstruction_errors

def evaluate_model(model, test_loader, cluster_centers, reconstruction_errors, epoch, dataset_name, device):
    model.eval()
    total_loss_ = 0
    total_loss_anomaly_ = 0
    all_labels = []
    all_scores = []
    reconstruction_errors_test = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
            
            # Forward pass
            graph_embeddings, x_recon, stats_pred, adj_recon_list = model(x, edge_index, batch, num_graphs)
            graph_embeddings_np = graph_embeddings.detach().cpu().numpy()
            
            recon_errors = []
            start_idx = 0
            for i in range(num_graphs):
                num_nodes = (batch == i).sum().item()
                end_idx = start_idx + num_nodes
                
                # Reconstruction error 계산
                if dataset_name == 'AIDS':
                    node_loss = torch.norm(x[start_idx:end_idx] - x_recon[start_idx:end_idx], p='fro')**2
                else:
                    node_loss = torch.norm(x[start_idx:end_idx] - x_recon[start_idx:end_idx], p='fro')**2 / num_nodes
                node_loss = node_loss.item() * alpha
                
                # 엣지 재구성 오류
                graph_edges = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)]
                graph_edges = graph_edges - start_idx
                true_adj = torch.zeros((model.edge_recon.max_nodes, model.edge_recon.max_nodes), device=device)
                true_adj[graph_edges[0], graph_edges[1]] = 1
                true_adj = true_adj + true_adj.t()
                true_adj = (true_adj > 0).float()
                
                node_mask = torch.zeros_like(adj_recon_list[i], dtype=torch.bool)
                node_mask[:num_nodes, :num_nodes] = True
                edge_loss = torch.sum((adj_recon_list[i][node_mask] - true_adj[node_mask]) ** 2) / node_mask.sum()
                edge_loss = edge_loss.item() * beta
                
                graph_vec = graph_embeddings_np[i].reshape(1, -1)
                distances = cdist(graph_vec, cluster_centers, metric='euclidean')
                min_distance = distances.min().item() * gamma
                
                recon_error = node_loss + min_distance + edge_loss
                
                reconstruction_errors_test.append({
                    'reconstruction': node_loss + edge_loss,
                    'clustering': min_distance,
                    'type': 'test_normal' if data.y[i].item() == 0 else 'test_anomaly'
                })
                
                recon_errors.append(recon_error)
                
                if data.y[i].item() == 0:
                    total_loss_ += recon_error
                else:
                    total_loss_anomaly_ += recon_error
                    
                start_idx = end_idx
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())
    
    # 밀도 기반 스코어링
    train_data = np.array([[error['reconstruction'], error['clustering']] 
                          for error in reconstruction_errors if error['type'] == 'train_normal'])
    test_normal = np.array([[error['reconstruction'], error['clustering']] 
                           for error in reconstruction_errors_test if error['type'] == 'test_normal'])
    test_anomaly = np.array([[error['reconstruction'], error['clustering']] 
                            for error in reconstruction_errors_test if error['type'] == 'test_anomaly'])
    
    bandwidth, _ = loocv_bandwidth_selection(train_data)
    density_scorer = DensityBasedScoring(bandwidth=bandwidth)
    density_scorer.fit(train_data)
    
    normal_scores = density_scorer.score_samples(test_normal)
    anomaly_scores = density_scorer.score_samples(test_anomaly)
    
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.array([0] * len(normal_scores) + [1] * len(anomaly_scores))
    
    # 성능 평가
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    auprc = auc(recall, precision)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred_labels = (all_scores > optimal_threshold).astype(int)
    
    precision = precision_score(all_labels, pred_labels)
    recall = recall_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)
    
    # 밀도 등고선 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    contour = density_scorer.plot_density_contours(train_data, test_normal, test_anomaly, ax1)
    fig.colorbar(contour, ax=ax1)
    
    ax2.hist(normal_scores, bins=50, alpha=0.5, density=True, label='Normal', color='green')
    ax2.hist(anomaly_scores, bins=50, alpha=0.5, density=True, label='Anomaly', color='red')
    
    plt.tight_layout()
    save_path = f'/home1/rldnjs16/graph_anomaly_detection/density_analysis/{dataset_name}_time_{current_time}/epoch_{epoch}_fold_{trial}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    visualization_data = {
        'normal': [{'reconstruction': error['reconstruction'], 
                   'clustering': error['clustering']}
                  for error in reconstruction_errors_test if error['type'] == 'test_normal'],
        'anomaly': [{'reconstruction': error['reconstruction'], 
                    'clustering': error['clustering']}
                   for error in reconstruction_errors_test if error['type'] == 'test_anomaly']
    }
    
    total_loss_mean = total_loss_ / sum(all_labels == 0)
    total_loss_anomaly_mean = total_loss_anomaly_ / sum(all_labels == 1)
    
    return auroc, auprc, precision, recall, f1, total_loss_mean, total_loss_anomaly_mean, all_scores, all_labels, reconstruction_errors_test, visualization_data


#%%
class DensityBasedScoring:
    def __init__(self, bandwidth=0.5):
        self.kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        self.scaler = StandardScaler()
        
    def fit(self, X):
        """
        정상 데이터의 2D 특징(재구성 오차, 클러스터링 거리)에 대해 KDE를 학습
        
        Args:
            X: shape (n_samples, 2) 형태의 array. 
               각 행은 [reconstruction_error, clustering_distance]
        """
        # 특징 정규화
        X_scaled = self.scaler.fit_transform(X)
        # KDE 학습
        self.kde.fit(X_scaled)
        
    def score_samples(self, X):
        """
        샘플들의 밀도 기반 이상 스코어 계산
        """
        X_scaled = self.scaler.transform(X)
        
        # log density 계산
        log_density = self.kde.score_samples(X_scaled)
        
        # -inf 값을 처리
        log_density = np.nan_to_num(log_density, neginf=-10000)
        
        # 이상 스코어 계산 및 클리핑
        anomaly_scores = -log_density  # 더 낮은 밀도 = 더 높은 이상 스코어
        anomaly_scores = np.clip(anomaly_scores, 0, 10000)  # 매우 큰 값 제한
        
        return anomaly_scores
    
    def plot_density_contours(self, X_train, X_test_normal, X_test_anomaly, ax, num_points=100):
        """
        밀도 등고선과 데이터 포인트를 시각화
        Args:
            X_train: 학습 데이터 (정상)
            X_test_normal: 테스트 데이터 (정상)
            X_test_anomaly: 테스트 데이터 (이상)
            ax: matplotlib axis
            num_points: 그리드 포인트 수
        """
        # 그리드 생성
        x_min = min(X_train[:, 0].min(), X_test_normal[:, 0].min(), X_test_anomaly[:, 0].min()) - 0.1
        x_max = max(X_train[:, 0].max(), X_test_normal[:, 0].max(), X_test_anomaly[:, 0].max()) + 0.1
        y_min = min(X_train[:, 1].min(), X_test_normal[:, 1].min(), X_test_anomaly[:, 1].min()) - 0.1
        y_max = max(X_train[:, 1].max(), X_test_normal[:, 1].max(), X_test_anomaly[:, 1].max()) + 0.1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, num_points),
                            np.linspace(y_min, y_max, num_points))
        
        # 그리드 포인트에서 밀도 계산
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.score_samples(grid_points)
        Z = Z.reshape(xx.shape)
        
        # 등고선 플롯
        levels = np.linspace(Z.min(), Z.max(), 20)
        contour = ax.contourf(xx, yy, Z, levels=levels, cmap='RdYlBu_r', alpha=0.7)
        
        # 데이터 포인트 플롯
        scatter_params = {
            'alpha': 0.7,
            'edgecolor': 'white',
            's': 80,
            'linewidth': 1.5
        }
        
        ax.scatter(X_train[:, 0], X_train[:, 1],
                c='dodgerblue', label='Train (Normal)',
                marker='o', **scatter_params)
        ax.scatter(X_test_normal[:, 0], X_test_normal[:, 1],
                c='mediumseagreen', label='Test (Normal)',
                marker='o', **scatter_params)
        ax.scatter(X_test_anomaly[:, 0], X_test_anomaly[:, 1],
                c='crimson', label='Test (Anomaly)',
                marker='o', **scatter_params)
        
        # 축 레이블과 그리드 설정
        ax.set_xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
        ax.set_ylabel('Clustering Distance', fontsize=12, fontweight='bold')
        
        # 그리드 스타일 개선
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)  # 그리드를 데이터 포인트 아래로
        
        # 범례 스타일 개선
        ax.legend(fontsize=10, frameon=True, facecolor='white', 
                edgecolor='gray', loc='upper right',
                bbox_to_anchor=(1.0, 1.0))
        
        # 축 스타일 개선
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)
        
        return contour


def persistence_stats_loss(pred_stats, true_stats):
    continuous_loss = F.mse_loss(pred_stats[:, :5], true_stats[:, :5])
    
    return continuous_loss


def plot_error_distribution(train_errors, test_errors, epoch, trial, dataset_name, current_time):
    # 데이터 분리
    train_normal_recon = [e['reconstruction'] for e in train_errors if e['type'] == 'train_normal']
    train_normal_cluster = [e['clustering'] for e in train_errors if e['type'] == 'train_normal']
    
    test_normal_recon = [e['reconstruction'] for e in test_errors if e['type'] == 'test_normal']
    test_normal_cluster = [e['clustering'] for e in test_errors if e['type'] == 'test_normal']
    
    test_anomaly_recon = [e['reconstruction'] for e in test_errors if e['type'] == 'test_anomaly']
    test_anomaly_cluster = [e['clustering'] for e in test_errors if e['type'] == 'test_anomaly']

    # 그래프 생성
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 일반 스케일 플롯
    ax1.scatter(train_normal_recon, train_normal_cluster, c='blue', label='Train (Normal)', alpha=0.6)
    ax1.scatter(test_normal_recon, test_normal_cluster, c='green', label='Test (Normal)', alpha=0.6)
    ax1.scatter(test_anomaly_recon, test_anomaly_cluster, c='red', label='Test (Anomaly)', alpha=0.6)
    
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Clustering Distance')
    ax1.set_title(f'Error Distribution (Epoch {epoch})')
    ax1.legend()
    ax1.grid(True)

    # 로그 스케일 플롯
    ax2.scatter(train_normal_recon, train_normal_cluster, c='blue', label='Train (Normal)', alpha=0.6)
    ax2.scatter(test_normal_recon, test_normal_cluster, c='green', label='Test (Normal)', alpha=0.6)
    ax2.scatter(test_anomaly_recon, test_anomaly_cluster, c='red', label='Test (Anomaly)', alpha=0.6)
    
    ax2.set_xlabel('Reconstruction Error')
    ax2.set_ylabel('Clustering Distance')
    ax2.set_title(f'Error Distribution - Log Scale (Epoch {epoch})')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/AB_GAE/plot/{dataset_name}_time_{current_time}/combined_error_distribution_epoch_{epoch}_fold_{trial}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

    # JSON으로 데이터 저장
    error_data = {
        'train_normal': [{'reconstruction': r, 'clustering': c} 
                        for r, c in zip(train_normal_recon, train_normal_cluster)],
        'test_normal': [{'reconstruction': r, 'clustering': c} 
                       for r, c in zip(test_normal_recon, test_normal_cluster)],
        'test_anomaly': [{'reconstruction': r, 'clustering': c} 
                        for r, c in zip(test_anomaly_recon, test_anomaly_cluster)]
    }
    
    json_path = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/AB_GAE/json/{dataset_name}_time_{current_time}/combined_error_distribution_epoch_{epoch}_fold_{trial}.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(error_data, f)


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='COX2')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--n-cluster", type=int, default=1)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--random-seed", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=300)
parser.add_argument("--log-interval", type=int, default=5)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256])

parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--dropout-rate", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.0001)
parser.add_argument("--learning-rate", type=float, default=0.0001)

parser.add_argument("--alpha", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=1.0)
parser.add_argument("--gamma", type=float, default=1.0)
parser.add_argument("--gamma-cluster", type=float, default=0.5)
parser.add_argument("--node-theta", type=float, default=0.03)
parser.add_argument("--adj-theta", type=float, default=0.01)

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


#%%
'''OPTIONS'''
data_root: str = args.data_root
assets_root: str = args.assets_root
dataset_name: str = args.dataset_name

epochs: int = args.epochs
patience: int = args.patience
n_cluster: int = args.n_cluster
step_size: int = args.step_size
batch_size: int = args.batch_size
n_cross_val: int = args.n_cross_val
random_seed: int = args.random_seed
hidden_dims: list = args.hidden_dims
log_interval: int = args.log_interval
n_test_anomaly: int = args.n_test_anomaly
test_batch_size: int = args.test_batch_size

factor: float = args.factor
test_size: float = args.test_size
weight_decay: float = args.weight_decay
dropout_rate: float = args.dropout_rate
learning_rate: float = args.learning_rate

alpha: float = args.alpha
beta: float = args.beta
gamma: float = args.gamma
gamma_cluster: float = args.gamma_cluster
node_theta: float = args.node_theta
adj_theta: float = args.adj_theta

set_seed(random_seed)

device = set_device()
# device = torch.device("cpu")
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

wandb.init(project="graph anomaly detection", entity="ki5n2")

wandb.config.update(args)

wandb.config = {
  "random_seed": random_seed,
  "learning_rate": learning_rate,
  "epochs": epochs
}


#%%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, negative_slope=0.1):
        super(ResidualBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, improved=True, add_self_loops=True, normalize=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', self.negative_slope)
        nn.init.xavier_uniform_(self.conv.lin.weight, gain=gain)
        nn.init.zeros_(self.conv.bias)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.xavier_uniform_(self.shortcut.weight, gain=1.0)
            nn.init.zeros_(self.shortcut.bias)

    def forward(self, x, edge_index):
        residual = self.shortcut(x)
        
        # 정규화 트릭 적용
        edge_index, _ = utils.add_self_loops(edge_index, num_nodes=x.size(0))
        deg = utils.degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        
        x = self.conv(x, edge_index, norm)
        x = self.activation(self.bn(x))
        x = self.dropout(x)
        
        return self.activation(x + residual)

class Encoder(nn.Module):
    def __init__(self, num_features, hidden_dims, dropout_rate=0.1):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList()
        dims = [num_features] + hidden_dims
        for i in range(len(dims) - 1):
            self.blocks.append(ResidualBlock(dims[i], dims[i+1], dropout_rate))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        for block in self.blocks:
            x = block(x, edge_index)
            x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)

class FeatureDecoder(nn.Module):
    def __init__(self, embed_dim, num_features, dropout_rate=0.1):
        super(FeatureDecoder, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim//2)
        self.fc2 = nn.Linear(embed_dim//2, num_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        z = self.leaky_relu(self.fc1(z))
        z = self.dropout(z)
        z = self.fc2(z)
        return z

class BilinearEdgeDecoder(nn.Module):
    def __init__(self, max_nodes):
        super(BilinearEdgeDecoder, self).__init__()
        self.max_nodes = max_nodes
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        actual_nodes = z.size(0)
        
        z_norm = F.normalize(z, p=2, dim=1)
        cos_sim = torch.mm(z_norm, z_norm.t())
        adj = self.sigmoid(cos_sim)
        adj = adj * (1 - torch.eye(actual_nodes, device=z.device))
        
        padded_adj = torch.zeros(self.max_nodes, self.max_nodes, device=z.device)
        padded_adj[:actual_nodes, :actual_nodes] = adj
        
        return padded_adj

class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_features, hidden_dims, dropout_rate)
        self.feature_decoder = FeatureDecoder(hidden_dims[-1], num_features)
        self.edge_recon = BilinearEdgeDecoder(max_nodes)
        self.stats_predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], 5)
        )

    def forward(self, x, edge_index, batch, num_graphs):
        # Encode
        node_embeddings = self.encoder(x, edge_index)
        
        # Pool for graph-level embeddings
        graph_embeddings = []
        start_idx = 0
        for i in range(num_graphs):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            end_idx = start_idx + num_nodes
            
            # Mean pooling for graph representation
            graph_embed = torch.mean(node_embeddings[start_idx:end_idx], dim=0)
            graph_embeddings.append(graph_embed)
            start_idx = end_idx
            
        graph_embeddings = torch.stack(graph_embeddings)
        
        # Decode features
        x_recon = self.feature_decoder(node_embeddings)
        
        # Predict stats
        stats_pred = self.stats_predictor(graph_embeddings)
        
        # Edge reconstruction
        adj_recon_list = []
        start_idx = 0
        for i in range(num_graphs):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            end_idx = start_idx + num_nodes
            
            current_embeddings = node_embeddings[start_idx:end_idx]
            adj_recon = self.edge_recon(current_embeddings)
            adj_recon_list.append(adj_recon)
            
            start_idx = end_idx
            
        return graph_embeddings, x_recon, stats_pred, adj_recon_list
        

#%%
'''DATASETS'''
if dataset_name == 'AIDS' or dataset_name == 'NCI1' or dataset_name == 'DHFR':
    dataset_AN = True
else:
    dataset_AN = False

splits = get_ad_split_TU(dataset_name, n_cross_val)
loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, splits[0], dataset_AN)
num_train = meta['num_train']
num_features = meta['num_feat']
num_edge_features = meta['num_edge_feat']
max_nodes = meta['max_nodes']

print(f'Number of graphs: {num_train}')
print(f'Number of features: {num_features}')
print(f'Number of edge features: {num_edge_features}')
print(f'Max nodes: {max_nodes}')

current_time_ = time.localtime()
current_time = time.strftime("%Y_%m_%d_%H_%M", current_time_)
print(f'random number saving: {current_time}')


#%%
def run(dataset_name, random_seed, dataset_AN, trial, device=device, epoch_results=None):
    if epoch_results is None:
        epoch_results = {}
    epoch_interval = 10
    
    set_seed(random_seed)    
    all_results = []
    split = splits[trial]
    
    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN)
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']
    
    model = GRAPH_AUTOENCODER(
        num_features=num_features, 
        hidden_dims=hidden_dims, 
        max_nodes=max_nodes,
        dropout_rate=dropout_rate
    ).to(device)
    
    train_loader = loaders['train']
    test_loader = loaders['test']
        
    # 학습
    print("Training autoencoder...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for epoch in range(1, epochs+1):
        train_loss, num_sample, cluster_centers, train_errors = train(
            model, train_loader, optimizer, epoch, dataset_name, device
        )
        
        info_train = 'Epoch {:3d}, Loss {:.4f}'.format(epoch, train_loss)

        if epoch % log_interval == 0:
            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly, all_scores, all_labels, test_errors, visualization_data = evaluate_model(
                model, test_loader, cluster_centers, train_errors, epoch, dataset_name, device
            )
            
            all_results.append((auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly))
            print(f'Epoch {epoch}: Training Loss = {train_loss:.4f}, Test loss = {test_loss:.4f}, '
                  f'Test loss anomaly = {test_loss_anomaly:.4f}, AUC = {auroc:.4f}, '
                  f'AUPRC = {auprc:.4f}, Precision = {precision:.4f}, '
                  f'Recall = {recall:.4f}, F1 = {f1:.4f}')
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_loss_anomaly": test_loss_anomaly,
                "auroc": auroc,
                "auprc": auprc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            info_test = 'AD_AUC:{:.4f}, AD_AUPRC:{:.4f}, Test_Loss:{:.4f}, Test_Loss_Anomaly:{:.4f}'.format(
                auroc, auprc, test_loss, test_loss_anomaly)
            print(info_train + '   ' + info_test)
            
            # Plot error distribution
            plot_error_distribution(train_errors, test_errors, epoch, trial, dataset_name, current_time)

            # Save visualization data
            for path_type in ['json', 'plot']:
                save_path = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/{path_type}/{dataset_name}_time_{current_time}/'
                os.makedirs(save_path, exist_ok=True)
            
            json_path = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}_time_{current_time}/error_distribution_epoch_{epoch}_fold_{trial}.json'
            with open(json_path, 'w') as f:
                json.dump(visualization_data, f)

            if epoch % epoch_interval == 0:
                if epoch not in epoch_results:
                    epoch_results[epoch] = {'aurocs': [], 'auprcs': [], 'precisions': [], 'recalls': [], 'f1s': []}
                
                epoch_results[epoch]['aurocs'].append(auroc)
                epoch_results[epoch]['auprcs'].append(auprc)
                epoch_results[epoch]['precisions'].append(precision)
                epoch_results[epoch]['recalls'].append(recall)
                epoch_results[epoch]['f1s'].append(f1)

    return auroc, epoch_results


#%%
'''MAIN'''
if __name__ == '__main__':
    ad_aucs = []
    fold_times = []
    epoch_results = {}
    splits = get_ad_split_TU(dataset_name, n_cross_val)    
    start_time = time.time()

    for trial in range(n_cross_val):
        fold_start = time.time()
        print(f"Starting fold {trial + 1}/{n_cross_val}")
        ad_auc, epoch_results = run(dataset_name, random_seed, dataset_AN, trial, device=device, epoch_results=epoch_results)
        ad_aucs.append(ad_auc)
        
        fold_end = time.time()
        fold_duration = fold_end - fold_start
        fold_times.append(fold_duration)
        
        print(f"Fold {trial + 1} finished in {fold_duration:.2f} seconds.")
    
    epoch_means = {}
    epoch_stds = {}
    for epoch in epoch_results.keys():
        epoch_means[epoch] = {
            'auroc': np.mean(epoch_results[epoch]['aurocs']),
            'auprc': np.mean(epoch_results[epoch]['auprcs']),
            'precision': np.mean(epoch_results[epoch]['precisions']),
            'recall': np.mean(epoch_results[epoch]['recalls']),
            'f1': np.mean(epoch_results[epoch]['f1s'])
        }
        epoch_stds[epoch] = {
            'auroc': np.std(epoch_results[epoch]['aurocs']),
            'auprc': np.std(epoch_results[epoch]['auprcs']),
            'precision': np.std(epoch_results[epoch]['precisions']),
            'recall': np.std(epoch_results[epoch]['recalls']),
            'f1': np.std(epoch_results[epoch]['f1s'])
        }
        
    best_epoch = max(epoch_means.keys(), key=lambda x: epoch_means[x]['auroc'])
    
    print("\n=== Performance at every 10 epochs (averaged over all folds) ===")
    for epoch in sorted(epoch_means.keys()):
        print(f"Epoch {epoch}: AUROC = {epoch_means[epoch]['auroc']:.4f} ± {epoch_stds[epoch]['auroc']:.4f}")
    
    print(f"\nBest average performance achieved at epoch {best_epoch}:")
    print(f"AUROC = {epoch_means[best_epoch]['auroc']:.4f} ± {epoch_stds[best_epoch]['auroc']:.4f}")
    print(f"AUPRC = {epoch_means[best_epoch]['auprc']:.4f} ± {epoch_stds[best_epoch]['auprc']:.4f}")
    print(f"F1 = {epoch_means[best_epoch]['f1']:.4f} ± {epoch_stds[best_epoch]['f1']:.4f}")
    
    total_time = time.time() - start_time
    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print('[FINAL RESULTS] ' + results)
    print(f"Total execution time: {total_time:.2f} seconds")
    
    results_path = f'/home1/rldnjs16/graph_anomaly_detection/cross_val_results/autoencoder_{dataset_name}_time_{current_time}_seed{random_seed}.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'epoch_means': epoch_means,
            'epoch_stds': epoch_stds,
            'best_epoch': int(best_epoch),
            'final_auroc_mean': float(np.mean(ad_aucs)),
            'final_auroc_std': float(np.std(ad_aucs))
        }, f, indent=4)

wandb.finish()
