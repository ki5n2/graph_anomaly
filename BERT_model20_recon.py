#%%
'''PRESENT'''
print('이번 BERT 모델 19은 AIDS, BZR, COX2, DHFR에 대한 실험 파일입니다. 마스크 토큰 재구성을 통한 프리-트레인이 이루어집니다. 이후 기존과 같이 노드 특성을 재구성하는 모델을 통해 이상을 탐지합니다. 기존 파일과 다른 점은 성능 평가 결과 비교를 코드 내에서 수행하고자 하였으며, 해당 파일만 실행하면 결과까지 한 번에 볼 수 있도록 하였습니다. 또한, 재구성 오류에 대한 정규화가 이루어져 있습니다. 추가로 훈련 데이터에 대한 산점도와 로그 스케일이 적용되어 있습니다. 그리고 2D density estimation이 적용되어 있습니다. 그리고 TopER 과정이 반영되어 있습니다. 밀도 기반 이상 스코어. 프리트레인 과정이 변경되었습니다.')

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
from modules.utils import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original, split_batch_graphs, compute_persistence, process_batch_graphs, scott_rule_bandwidth, loocv_bandwidth_selection

import networkx as nx

# 메모리 설정
torch.cuda.empty_cache()
gc.collect()


#%%
'''TRAIN BERT'''
def pretrain_graph_bert(model, train_loader, optimizer, device):
    """Graph-BERT pre-training with node reconstruction and structure recovery"""
    model.train()
    total_recon_loss = 0
    total_struct_loss = 0
    num_samples = 0
    
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # 15% 노드를 마스킹
        mask_indices = torch.rand(data.x.size(0), device=device) < 0.15
        
        # Forward pass - pre-training mode
        transformed, masked_outputs, structure_outputs = model(
            data.x, data.edge_index, data.batch, data.num_graphs,
            mask_indices=mask_indices, is_pretrain=True
        )
        
        # Node reconstruction loss
        if masked_outputs is not None and mask_indices.sum() > 0:
            masked_features = data.x[mask_indices]
            recon_loss = F.mse_loss(
                masked_outputs,
                masked_features,
                reduction='mean'
            )
        else:
            recon_loss = torch.tensor(0.0, device=device)
            
        # Structure recovery loss
        intimacy_scores = []
        start_idx = 0
        for i in range(data.num_graphs):
            mask = (data.batch == i)
            num_nodes = mask.sum().item()
            graph_edge_index = data.edge_index[:, (data.edge_index[0] >= start_idx) & 
                                               (data.edge_index[0] < start_idx + num_nodes)]
            graph_edge_index = graph_edge_index - start_idx
            
            intimacy_matrix = model.encoder.compute_intimacy_matrix(
                graph_edge_index, num_nodes
            )
            intimacy_scores.append(intimacy_matrix)
            start_idx += num_nodes
            
        intimacy_target = torch.block_diag(*intimacy_scores)
        struct_loss = F.mse_loss(
            structure_outputs @ structure_outputs.t(),
            intimacy_target,
            reduction='mean'
        )
        
        print(f'recon_loss: {recon_loss}')
        print(f'struct_loss: {struct_loss}')
        
        loss = recon_loss + struct_loss
        loss.backward()
        optimizer.step()
        
        total_recon_loss += recon_loss.item() * data.num_graphs
        total_struct_loss += struct_loss.item() * data.num_graphs
        num_samples += data.num_graphs
        
    return (total_recon_loss + total_struct_loss) / num_samples, num_samples


def train(model, train_loader, recon_optimizer, device, epoch, cluster_centers=None):
    model.train()
    total_loss = 0
    num_sample = 0
    reconstruction_errors = []
    
    for data in train_loader:
        # data = TopER_Embedding(data)
        data = data.to(device)
        recon_optimizer.zero_grad()
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        # toper_target = data.toper_embeddings
        
        # Forward pass - fine-tuning mode
        train_cls_outputs, x_recon = model(
            x, edge_index, batch, num_graphs,
            is_pretrain=False
        )
        
        if epoch % 5 == 0:
            cls_outputs_np = train_cls_outputs.detach().cpu().numpy()
            kmeans = KMeans(n_clusters=n_cluster, random_state=random_seed)
            kmeans.fit(cls_outputs_np)
            cluster_centers = kmeans.cluster_centers_
            
        loss = 0
        node_loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
            node_loss += node_loss_
            
            if epoch % 5 == 0:
                node_loss_scaled = node_loss_.item() * alpha
                cls_vec = train_cls_outputs[i].detach().cpu().numpy()
                distances = cdist([cls_vec], cluster_centers, metric='euclidean')
                min_distance = distances.min().item() * gamma
                
                reconstruction_errors.append({
                    'reconstruction': node_loss_scaled,
                    'clustering': min_distance,
                    'type': 'train_normal'  # 훈련 데이터는 모두 정상
                })
            
            start_node = end_node
            
        # stats_loss = persistence_stats_loss(stats_pred, toper_target)
        
        # alpha_ = 10
        # stats_loss = alpha_ * stats_loss

        loss = node_loss
        
        print(f'node_loss: {node_loss}')
        # print(f'stats_loss: {stats_loss}')
        
        num_sample += num_graphs
        loss.backward()
        recon_optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader), num_sample, cluster_centers, reconstruction_errors


#%%
def evaluate_model(model, test_loader, cluster_centers, dataset_name, device):
    model.eval()
    total_loss_ = 0
    total_loss_anomaly_ = 0
    all_labels = []
    all_scores = []
    reconstruction_errors_test = []  # 새로 추가
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs 
            # Forward pass - evaluation mode
            e_cls_output, x_recon = model(
                x, edge_index, batch, num_graphs,
                is_pretrain=False
            )
            
            e_cls_outputs_np = e_cls_output.detach().cpu().numpy()  # [num_graphs, hidden_dim]
            
            recon_errors = []
            start_node = 0
            for i in range(num_graphs):
                num_nodes = (batch == i).sum().item()
                end_node = start_node + num_nodes
                
                # Reconstruction error 계산
                if dataset_name == 'AIDS':
                    node_loss = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2
                else:
                    node_loss = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
                
                node_loss = node_loss.item() * alpha
                
                cls_vec = e_cls_outputs_np[i].reshape(1, -1)
                distances = cdist(cls_vec, cluster_centers, metric='euclidean')
                min_distance = distances.min().item() * gamma
                
                # 변환된 값들 저장
                reconstruction_errors_test.append({
                    'reconstruction': node_loss,
                    'clustering': min_distance,
                    'type': 'test_normal' if data.y[i].item() == 0 else 'test_anomaly'
                })

                # 전체 에러는 변환된 값들의 평균으로 계산
                recon_error = node_loss + min_distance              
                recon_errors.append(recon_error)
                
                print(f'test_node_loss: {node_loss}')
                print(f'test_min_distance: {min_distance}')
                
                if data.y[i].item() == 0:
                    total_loss_ += recon_error
                else:
                    total_loss_anomaly_ += recon_error
                    
                start_node = end_node
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())
    
    # 시각화를 위한 데이터 변환
    visualization_data = {
        'normal': [
            {'reconstruction': error['reconstruction'], 
             'clustering': error['clustering']}
            for error in reconstruction_errors_test if error['type'] == 'test_normal'
        ],
        'anomaly': [
            {'reconstruction': error['reconstruction'], 
             'clustering': error['clustering']}
            for error in reconstruction_errors_test if error['type'] == 'test_anomaly'
        ]
    }
    
    # 메트릭 계산
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # 성능 메트릭 계산
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auroc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    auprc = auc(recall, precision)
    
    # 최적 임계값 찾기
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    pred_labels = (all_scores > optimal_threshold).astype(int)
    
    precision = precision_score(all_labels, pred_labels)
    recall = recall_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)
    
    total_loss_mean = total_loss_ / sum(all_labels == 0)
    total_loss_anomaly_mean = total_loss_anomaly_ / sum(all_labels == 1)
    
    return auroc, auprc, precision, recall, f1, total_loss_mean, total_loss_anomaly_mean, all_scores, all_labels, reconstruction_errors_test, visualization_data


#%%
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
    
    ax1.set_xlabel('Reconstruction Error (node_loss * alpha)')
    ax1.set_ylabel('Clustering Distance (min_distance * gamma)')
    ax1.set_title(f'Error Distribution (Epoch {epoch})')
    ax1.legend()
    ax1.grid(True)

    # 로그 스케일 플롯
    ax2.scatter(train_normal_recon, train_normal_cluster, c='blue', label='Train (Normal)', alpha=0.6)
    ax2.scatter(test_normal_recon, test_normal_cluster, c='green', label='Test (Normal)', alpha=0.6)
    ax2.scatter(test_anomaly_recon, test_anomaly_cluster, c='red', label='Test (Anomaly)', alpha=0.6)
    
    ax2.set_xlabel('Reconstruction Error (node_loss * alpha)')
    ax2.set_ylabel('Clustering Distance (min_distance * gamma)')
    ax2.set_title(f'Error Distribution - Log Scale (Epoch {epoch})')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    save_path = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/plot/{dataset_name}_time_{current_time}/combined_error_distribution_epoch_{epoch}_fold_{trial}.png'
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
    
    json_path = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}_time_{current_time}/combined_error_distribution_epoch_{epoch}_fold_{trial}.json'
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(error_data, f)


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


# Loss 함수 정의
def persistence_stats_loss(pred_stats, true_stats):
    continuous_loss = F.mse_loss(pred_stats[:, :2], true_stats[:, :2])
    
    return continuous_loss


#%%
class PyGTopER:
    def __init__(self, thresholds: List[float]):
        # thresholds를 float32로 변환
        self.thresholds = torch.tensor(sorted(thresholds), dtype=torch.float32)

    
    def _get_graph_structure(self, x: torch.Tensor, edge_index: torch.Tensor, 
                           batch: torch.Tensor, graph_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # 모든 텐서를 float32로 변환
        x = x.float()  # float32로 명시적 변환
        
        # CPU로 이동 및 처리
        x = x.cpu()
        edge_index = edge_index.cpu()
        batch = batch.cpu()
        
        mask = batch == graph_idx
        nodes = x[mask]
        
        node_idx = torch.arange(len(batch), dtype=torch.long, device='cpu')[mask]
        idx_map = {int(old): new for new, old in enumerate(node_idx)}
        
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
        graph_edges = edge_index[:, edge_mask]
        
        graph_edges = torch.tensor([[idx_map[int(i)] for i in graph_edges[0]],
                                  [idx_map[int(i)] for i in graph_edges[1]]], 
                                 dtype=torch.long,  # long 타입 명시
                                 device='cpu')
        
        return nodes, graph_edges
    
    def _get_node_filtration(self, nodes: torch.Tensor, edges: torch.Tensor, 
                            node_values: torch.Tensor) -> List[Tuple[int, int]]:
        """Compute filtration sequence for a single graph"""
        sequences = []
        for threshold in self.thresholds:
            # Get nodes below threshold
            mask = node_values <= threshold
            if not torch.any(mask):
                sequences.append((0, 0))
                continue
                
            # Get induced edges
            edge_mask = mask[edges[0]] & mask[edges[1]]
            filtered_edges = edges[:, edge_mask]
            
            sequences.append((torch.sum(mask).item(), 
                            filtered_edges.shape[1] // 2))
            
        return sequences

    def _compute_degree_values(self, num_nodes: int, edges: torch.Tensor) -> torch.Tensor:
        degrees = torch.zeros(num_nodes, dtype=torch.float32, device='cpu')  # float32 명시
        unique, counts = torch.unique(edges[0], return_counts=True)
        degrees[unique] += counts.float()  # counts를 float32로 변환
        return degrees
    
    def _compute_popularity_values(self, num_nodes: int, edges: torch.Tensor) -> torch.Tensor:
        """Compute popularity values as defined in the paper"""
        degrees = self._compute_degree_values(num_nodes, edges)
        
        popularity = torch.zeros(num_nodes, device='cpu')
        for i in range(num_nodes):
            neighbors = edges[1][edges[0] == i]
            if len(neighbors) > 0:
                neighbor_degrees = degrees[neighbors]
                popularity[i] = degrees[i] + neighbor_degrees.mean()
            else:
                popularity[i] = degrees[i]
                
        return popularity
    
    def _refine_sequences(self, x_vals: np.ndarray, y_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Refine sequences according to the paper's methodology"""
        refined_x = []
        refined_y = []
        i = 0
        while i < len(y_vals):
            # Find consecutive points with same y value
            j = i + 1
            while j < len(y_vals) and y_vals[j] == y_vals[i]:
                j += 1
            
            # Calculate mean of x values
            x_mean = x_vals[i:j].mean()
            refined_x.append(x_mean)
            refined_y.append(y_vals[i])
            
            i = j
            
        return np.array(refined_x), np.array(refined_y)
    
    def _fit_line(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Fit line to refined sequences with float32 precision"""
        if len(x) < 2:
            return 0.0, 0.0
            
        slope, intercept, _, _, _ = linregress(x.astype(np.float32), y.astype(np.float32))
        return float(intercept), float(slope)  # float32 반환

    def compute_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor, 
                          batch: torch.Tensor, filtration: str = 'degree') -> torch.Tensor:
        device = x.device
        num_graphs = batch.max().item() + 1
        embeddings = []
        
        for graph_idx in range(num_graphs):
            nodes, edges = self._get_graph_structure(x, edge_index, batch, graph_idx)
            
            if filtration == 'degree':
                values = self._compute_degree_values(len(nodes), edges)
            elif filtration == 'popularity':
                values = self._compute_popularity_values(len(nodes), edges)
            else:
                raise ValueError(f"Unknown filtration type: {filtration}")
            
            sequences = self._get_node_filtration(nodes, edges, values)
            x_vals, y_vals = zip(*sequences)
            
            # numpy 배열을 float32로 변환
            x_refined, y_refined = self._refine_sequences(
                np.array(x_vals, dtype=np.float32), 
                np.array(y_vals, dtype=np.float32)
            )
            
            pivot, growth = self._fit_line(x_refined, y_refined)
            embeddings.append([pivot, growth])
            
        # 최종 결과를 float32 텐서로 변환
        return torch.tensor(embeddings, dtype=torch.float32, device=device)

    
def TopER_Embedding(data):
    # TopER 임베딩 계산
    thresholds = np.linspace(0, 5, 20)  # 논문에서 사용한 값 범위로 조정 필요
    toper = PyGTopER(thresholds)
    toper_embeddings = toper.compute_embeddings(data.x, data.edge_index, data.batch)
    
    # 데이터에 TopER 임베딩 추가
    data.toper_embeddings = toper_embeddings
    return data


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='COX2')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--n-head-BERT", type=int, default=2)
parser.add_argument("--n-layer-BERT", type=int, default=2)
parser.add_argument("--n-head", type=int, default=2)
parser.add_argument("--n-layer", type=int, default=2)
parser.add_argument("--BERT-epochs", type=int, default=30)
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
parser.add_argument("--beta", type=float, default=0.001)
parser.add_argument("--gamma", type=float, default=0.1)
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

BERT_epochs: int = args.BERT_epochs
epochs: int = args.epochs
n_head_BERT: int = args.n_head_BERT
n_layer_BERT: int = args.n_layer_BERT
n_head: int = args.n_head
n_layer: int = args.n_layer
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


# %%
'''MODEL CONSTRUCTION'''
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
        
        # 정규화된 인접 행렬을 사용하여 합성곱 적용
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
        
        z_norm = F.normalize(z, p=2, dim=1) # 각 노드 벡터를 정규화
        cos_sim = torch.mm(z_norm, z_norm.t()) # 코사인 유사도 계산 (내적으로 계산됨)
        adj = self.sigmoid(cos_sim)
        adj = adj * (1 - torch.eye(actual_nodes, device=z.device))
        
        # max_nodes 크기로 패딩
        padded_adj = torch.zeros(self.max_nodes, self.max_nodes, device=z.device)
        padded_adj[:actual_nodes, :actual_nodes] = adj
        
        return padded_adj

    
#%%
class GraphBertEncoder(nn.Module):
    def __init__(self, num_features, hidden_dims, d_model, nhead, num_layers, max_nodes, dropout_rate=0.1):
        super().__init__()
        # 노드 임베딩 컴포넌트
        self.raw_embeddings = nn.Linear(num_features, hidden_dims[-1])
        self.wl_embeddings = nn.Embedding(100, hidden_dims[-1])
        self.pos_embeddings = nn.Embedding(max_nodes, hidden_dims[-1])
        self.hop_embeddings = nn.Embedding(max_nodes, hidden_dims[-1])
        
        # 임베딩 정규화
        self.layer_norm = nn.LayerNorm(hidden_dims[-1])
        self.dropout = nn.Dropout(dropout_rate)
        
        # Transformer 인코더
        encoder_layers = []
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                'attention': nn.MultiheadAttention(
                    hidden_dims[-1], nhead, dropout=dropout_rate, batch_first=True
                ),
                'norm1': nn.LayerNorm(hidden_dims[-1]),
                'norm2': nn.LayerNorm(hidden_dims[-1]),
                'ffn': nn.Sequential(
                    nn.Linear(hidden_dims[-1], hidden_dims[-1] * 4),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dims[-1] * 4, hidden_dims[-1])
                )
            })
            encoder_layers.append(layer)
        self.encoder_layers = nn.ModuleList(encoder_layers)
        
        # 마스크 토큰과 Graph Residual을 위한 컴포넌트
        self.mask_token = nn.Parameter(torch.randn(1, hidden_dims[-1]))
        self.graph_residual = nn.Linear(num_features, hidden_dims[-1])
        
        # Pre-training heads
        self.node_predictor = nn.Linear(hidden_dims[-1], num_features)
        self.structure_predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        
        self.max_nodes = max_nodes
        self.hidden_dims = hidden_dims
        self.alpha = 0.15  # pagerank damping factor

    def compute_wl_roles(self, edge_index, num_nodes, num_iters=3):
        """Enhanced WL algorithm with better structural feature capture"""
        edge_index_np = edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(zip(edge_index_np[0], edge_index_np[1]))
        
        # 초기 레이블은 [degree, clustering coefficient, pagerank]
        initial_labels = {}
        pagerank = nx.pagerank(G, alpha=0.85)
        clustering = nx.clustering(G)
        
        for node in G.nodes():
            initial_labels[node] = (
                G.degree(node),
                clustering.get(node, 0),
                int(pagerank.get(node, 0) * 1000)
            )
        
        labels = initial_labels
        for _ in range(num_iters):
            new_labels = {}
            for node in G.nodes():
                # 이웃의 레이블 집계
                neighbor_labels = sorted([labels[n] for n in G.neighbors(node)])
                if not neighbor_labels:
                    neighbor_labels = [(0, 0, 0)]
                
                # 구조적 특징 업데이트
                deg_sum = sum(l[0] for l in neighbor_labels)
                clust_sum = sum(l[1] for l in neighbor_labels)
                pr_sum = sum(l[2] for l in neighbor_labels)
                
                new_label = (
                    labels[node][0] + deg_sum,
                    labels[node][1] + clust_sum / (len(neighbor_labels) + 1e-6),
                    labels[node][2] + pr_sum / (len(neighbor_labels) + 1e-6)
                )
                new_labels[node] = new_label
            
            labels = new_labels
        
        # 레이블을 100개의 버킷으로 해싱
        final_colors = []
        for node in range(num_nodes):
            label = labels[node]
            hash_val = hash(str(label)) % 100
            final_colors.append(hash_val)
        
        return torch.tensor(final_colors, device=edge_index.device)

    def compute_intimacy_matrix(self, edge_index, num_nodes):
        """Improved intimacy matrix computation using multiple metrics"""
        # 정규화된 인접 행렬 계산
        adj = torch.sparse_coo_tensor(
            edge_index,
            torch.ones(edge_index.size(1), device=edge_index.device),
            (num_nodes, num_nodes)
        ).to_dense()
        
        # PageRank 계산
        deg = adj.sum(dim=1)
        deg_inv = 1.0 / deg.clamp(min=1.)
        norm_adj = adj * deg_inv.unsqueeze(1)
        identity = torch.eye(num_nodes, device=edge_index.device)
        pagerank = self.alpha * torch.inverse(identity - (1 - self.alpha) * norm_adj)
        
        # 구조적 유사도 계산 (Katz 유사도)
        beta = 0.001  # 감쇠 계수
        katz = torch.inverse(identity - beta * norm_adj) - identity
        
        # 최종 intimacy score는 PageRank와 Katz의 조합
        intimacy = (pagerank + katz) / 2
        return F.normalize(intimacy, p=2, dim=1)

    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, training=False):
        device = x.device
        z_list = []
        subgraphs = []
        start_idx = 0
        
        # Graph residual 계산
        residual = self.graph_residual(x)  # [total_nodes, hidden_dim]
        
        for i in range(num_graphs):
            # 현재 그래프 추출
            mask = (batch == i)
            num_nodes = mask.sum().item()
            current_x = x[mask]
            
            # Linkless subgraph 생성을 위한 context 계산
            if training:
                # Pre-training 시에만 edge_index 사용
                current_edge_index = edge_index[:, (edge_index[0] >= start_idx) & 
                                               (edge_index[0] < start_idx + num_nodes)]
                current_edge_index = current_edge_index - start_idx
                
                intimacy_matrix = self.compute_intimacy_matrix(current_edge_index, num_nodes)
                wl_codes = self.compute_wl_roles(current_edge_index, num_nodes)
            else:
                # Fine-tuning과 추론 시에는 attention만 사용
                intimacy_matrix = torch.eye(num_nodes, device=device)
                wl_codes = torch.zeros(num_nodes, device=device)
            
            # 각 노드에 대한 context 선택
            for node_idx in range(num_nodes):
                # Context sampling
                context_scores = intimacy_matrix[node_idx]
                _, context_indices = torch.topk(context_scores, k=min(5, num_nodes-1))
                subgraph_nodes = torch.cat([torch.tensor([node_idx], device=device), context_indices])
                
                # Feature embeddings
                subgraph_x = current_x[subgraph_nodes]
                raw_embed = self.raw_embeddings(subgraph_x)
                
                # Positional embeddings
                pos_ids = torch.arange(len(subgraph_nodes), device=device)
                pos_embed = self.pos_embeddings(pos_ids)
                
                # WL role embeddings
                subgraph_wl = wl_codes[subgraph_nodes.long()]  # subgraph_nodes를 long 타입으로 변환
                wl_embed = self.wl_embeddings(subgraph_wl.long())  # wl_codes도 long 타입으로 변환
                
                # Distance embeddings (using attention scores as proxy)
                dist_scores = intimacy_matrix[node_idx, subgraph_nodes]
                hop_ids = (1 / dist_scores * 5).long().clamp(0, self.max_nodes-1)
                hop_embed = self.hop_embeddings(hop_ids)
                
                # Combine embeddings
                node_embeddings = raw_embed + pos_embed + wl_embed + hop_embed
                node_embeddings = self.layer_norm(node_embeddings)
                node_embeddings = self.dropout(node_embeddings)
                
                # Masking
                if training and mask_indices is not None and mask_indices[start_idx + node_idx]:
                    node_embeddings[0] = self.mask_token
                
                z_list.append(node_embeddings)
                subgraphs.append((node_idx, subgraph_nodes))
            
            start_idx += num_nodes
        
        # Transformer encoding with graph residual
        batch_size = len(z_list)
        max_size = max(z.size(0) for z in z_list)
        padded_z = torch.zeros(batch_size, max_size, self.hidden_dims[-1], device=device)
        attention_mask = torch.ones(batch_size, max_size, device=device).bool()

        # Residual mapping 생성
        residual_mapped = torch.zeros_like(padded_z)
        start_idx = 0
        for i, z in enumerate(z_list):
            size = z.size(0)
            padded_z[i, :size] = z
            attention_mask[i, :size] = False
            
            # 현재 subgraph의 residual 매핑
            mask = (batch == i)
            node_indices = torch.where(mask)[0]
            
            # 크기 확인 및 안전한 슬라이싱
            valid_size = min(size, len(node_indices))
            if valid_size > 0:
                residual_mapped[i, :valid_size] = residual[node_indices[:valid_size]]
            
            start_idx += size
        

        # Multi-layer transformer with residual
        hidden_states = padded_z
        for layer in self.encoder_layers:
            # Self-attention with residual
            attn_output, _ = layer['attention'](
                hidden_states, hidden_states, hidden_states,
                key_padding_mask=attention_mask
            )
            hidden_states = layer['norm1'](hidden_states + attn_output + residual_mapped)
            
            # Feed-forward with residual
            ff_output = layer['ffn'](hidden_states)
            hidden_states = layer['norm2'](hidden_states + ff_output + residual_mapped)
        
        
        if training and mask_indices is not None:
            # Node reconstruction
            masked_outputs = []
            for idx, mask in enumerate(mask_indices):
                if mask:
                    graph_idx = batch[idx]
                    node_pos = idx - (batch < graph_idx).sum()
                    masked_outputs.append(self.node_predictor(hidden_states[node_pos, 0]))
            
            masked_outputs = torch.stack(masked_outputs) if masked_outputs else None
            
            # Structure prediction with improved loss
            structure_outputs = self.structure_predictor(hidden_states[:, 0])
            
            return hidden_states, masked_outputs, structure_outputs
        
        return hidden_states[:, 0]
        

#%%
class BatchUtils:
    @staticmethod
    def process_batch(x, edge_index, batch, num_graphs=None):
        """배치 데이터 전처리를 위한 유틸리티 메서드"""
        batch_size = num_graphs if num_graphs is not None else batch.max().item() + 1
        
        # 그래프별 노드 특징과 엣지 인덱스 분리
        z_list = [x[batch == i] for i in range(batch_size)]
        edge_index_list = []
        start_idx = 0
        
        for i in range(batch_size):
            num_nodes = z_list[i].size(0)
            graph_edges = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < start_idx + num_nodes)]
            graph_edges = graph_edges - start_idx
            edge_index_list.append(graph_edges)
            start_idx += num_nodes
            
        max_nodes_in_batch = max(z_graph.size(0) for z_graph in z_list)
        
        return z_list, edge_index_list, max_nodes_in_batch

    @staticmethod
    def add_cls_token(z_list, cls_token, max_nodes_in_batch, device):
        """CLS 토큰 추가 및 패딩 처리"""
        z_with_cls_list = []
        mask_list = []
        
        for z_graph in z_list:
            num_nodes = z_graph.size(0)
            cls_token = cls_token.to(device)
            z_graph = z_graph.unsqueeze(1)
            
            # 패딩
            pad_size = max_nodes_in_batch - num_nodes
            z_graph_padded = F.pad(z_graph, (0, 0, 0, 0, 0, pad_size), 'constant', 0)
            
            # CLS 토큰 추가
            z_with_cls = torch.cat([cls_token, z_graph_padded.transpose(0, 1)], dim=1)
            z_with_cls_list.append(z_with_cls)
            
            # 마스크 생성
            graph_mask = torch.cat([torch.tensor([False]), torch.tensor([False]*num_nodes + [True]*pad_size)])
            mask_list.append(graph_mask)
            
        z_with_cls_batch = torch.cat(z_with_cls_list, dim=0)
        mask = torch.stack(mask_list).to(device)
        
        return z_with_cls_batch, mask


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_nodes, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first = True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask):
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output
    
            
#%%
# GRAPH_AUTOENCODER 클래스 수정
class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, nhead_BERT, nhead, num_layers_BERT, num_layers, dropout_rate=0.1):
        super().__init__()
        self.encoder = GraphBertEncoder(  ## 수정할 부분!!!
            num_features=num_features,
            hidden_dims=hidden_dims,
            d_model=hidden_dims[-1],
            nhead=nhead_BERT,
            num_layers=num_layers_BERT,
            max_nodes=max_nodes,
            dropout_rate=dropout_rate
        )
        self.transformer_d = TransformerEncoder(
            d_model=hidden_dims[-1],
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=hidden_dims[-1] * 4,
            max_nodes=max_nodes,
            dropout=dropout_rate
        )
        self.feature_decoder = FeatureDecoder(hidden_dims[-1], num_features)
        self.u_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dims[-1]))        
        # self.stats_predictor = nn.Sequential(
        #     nn.Linear(hidden_dims[-1], 2)
        # )
        
    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, is_pretrain=False):
        # Pre-training phase (마스크 토큰 재구성 + 구조 복원)
        if is_pretrain:
            transformed, masked_outputs, structure_outputs = self.encoder( # 수정할 부분!!
                x, edge_index, batch, num_graphs,
                mask_indices=mask_indices,
                training=True
            )
            return transformed, masked_outputs, structure_outputs
        
        # Fine-tuning phase (이상 탐지)
        else:
            # Regular forward pass
            transformed = self.encoder(
                x, edge_index, batch, num_graphs,
                training=False
            )
            
            # 배치 처리
            z_list, edge_index_list, max_nodes_in_batch = BatchUtils.process_batch(transformed, edge_index, batch, num_graphs)
            z_with_cls_batch, mask = BatchUtils.add_cls_token(
                z_list, self.cls_token, max_nodes_in_batch, x.device
            )
            
            # Transformer 처리
            encoded = self.transformer_d(z_with_cls_batch, mask)
            
            # 출력 처리
            cls_output = encoded[:, 0, :]
            node_outputs = [encoded[i, 1:z_list[i].size(0)+1, :] for i in range(num_graphs)]
            u = torch.cat(node_outputs, dim=0)
            
            # 통계량 예측
            # stats_pred = self.stats_predictor(cls_output)
            
            # 디코딩
            u_prime = self.u_mlp(u)
            x_recon = self.feature_decoder(u_prime)
            
            return cls_output, x_recon
    

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
'''RUN'''
def run(dataset_name, random_seed, dataset_AN, trial, device=device, epoch_results=None):
    if epoch_results is None:
        epoch_results = {}
    epoch_interval = 10  # 10 에폭 단위로 비교
    
    set_seed(random_seed)    
    all_results = []
    split=splits[trial]
    
    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN)
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']
    max_node_label = meta['max_node_label']
    
    # BERT 모델 저장 경로
    bert_save_path = f'/home1/rldnjs16/graph_anomaly_detection/BERT_model/Class/all_pretrained_bert_{dataset_name}_fold{trial}_nhead{n_head_BERT}_seed{random_seed}_BERT_epochs{BERT_epochs}_try20.pth'
    
    model = GRAPH_AUTOENCODER(
        num_features=num_features, 
        hidden_dims=hidden_dims, 
        max_nodes=max_nodes,
        nhead_BERT=n_head_BERT,
        nhead=n_head,
        num_layers_BERT=n_layer_BERT,
        num_layers=n_layer,
        dropout_rate=dropout_rate
    ).to(device)
    
    train_loader = loaders['train']
    test_loader = loaders['test']
    
    # 훈련 단계에서 cls_outputs 저장할 리스트 초기화
    global train_cls_outputs
    train_cls_outputs = []
    
    # 1단계: BERT 임베딩 학습
    if os.path.exists(bert_save_path):
        print("Loading pretrained BERT...")
        # BERT 인코더의 가중치만 로드
        model.encoder.load_state_dict(torch.load(bert_save_path, weights_only=True))
    else:
        print("Training BERT from scratch...")
        # 1단계: BERT 임베딩 학습
        print("Stage 1: Training BERT embeddings...")

        pretrain_params = list(model.encoder.parameters())
        bert_optimizer = torch.optim.Adam(pretrain_params, lr=learning_rate)
        
        for epoch in range(1, BERT_epochs+1):
            train_loss, num_sample = pretrain_graph_bert(
                model, train_loader, bert_optimizer, device
            )
            
            if epoch % log_interval == 0:
                print(f'BERT Training Epoch {epoch}: Loss = {train_loss:.4f}')
                
        # 학습된 BERT 저장
        print("Saving pretrained BERT...")
        torch.save(model.encoder.state_dict(), bert_save_path)
        
    # 2단계: 재구성 학습
    print("\nStage 2: Training reconstruction...")
    recon_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    for epoch in range(1, epochs+1):
        fold_start = time.time()  # 현재 폴드 시작 시간
        train_loss, num_sample, train_cluster_centers, train_errors = train(model, train_loader, recon_optimizer, device, epoch)
        
        info_train = 'Epoch {:3d}, Loss {:.4f}'.format(epoch, train_loss)

        if epoch % log_interval == 0:
            print(f"\nClustering Analysis Results (Epoch {epoch}):")
                                                                                                                                        
            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly, all_scores, all_labels, test_errors, visualization_data = evaluate_model(model, test_loader, train_cluster_centers, dataset_name, device)
                                                                                                                                                    
            plot_error_distribution(train_errors, test_errors, epoch, trial, dataset_name, current_time)

            save_path_ = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}_time_{current_time}/'
            os.makedirs(os.path.dirname(save_path_), exist_ok=True)
            save_path_ = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/plot/{dataset_name}_time_{current_time}/'
            os.makedirs(os.path.dirname(save_path_), exist_ok=True)
            
            save_path = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}_time_{current_time}/error_distribution_epoch_{epoch}_fold_{trial}.json'
            with open(save_path, 'w') as f:
                json.dump(visualization_data, f)
            
            all_results.append((auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly))
            print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation loss = {test_loss:.4f}, Validation loss anomaly = {test_loss_anomaly:.4f}, Validation AUC = {auroc:.4f}, Validation AUPRC = {auprc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}')
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": test_loss,
                "val_loss_anomaly": test_loss_anomaly,
                "auroc": auroc,
                "auprc": auprc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "learning_rate": recon_optimizer.param_groups[0]['lr']
            })
            
            info_test = 'AD_AUC:{:.4f}, AD_AUPRC:{:.4f}, Test_Loss:{:.4f}, Test_Loss_Anomaly:{:.4f}'.format(auroc, auprc, test_loss, test_loss_anomaly)
            print(info_train + '   ' + info_test)

            # 10 에폭 단위일 때 결과 저장
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
    epoch_results = {}  # 모든 폴드의 에폭별 결과를 저장
    splits = get_ad_split_TU(dataset_name, n_cross_val)    
    start_time = time.time()  # 전체 실행 시작 시간

    for trial in range(n_cross_val):
        fold_start = time.time()  # 현재 폴드 시작 시간
        print(f"Starting fold {trial + 1}/{n_cross_val}")
        ad_auc, epoch_results = run(dataset_name, random_seed, dataset_AN, trial, device=device, epoch_results=epoch_results)
        ad_aucs.append(ad_auc)
        
        fold_end = time.time()  # 현재 폴드 종료 시간   
        fold_duration = fold_end - fold_start  # 현재 폴드 실행 시간
        fold_times.append(fold_duration)
        
        print(f"Fold {trial + 1} finished in {fold_duration:.2f} seconds.")
    
    # 각 에폭별 평균 성능 계산
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
        
    # 최고 성능을 보인 에폭 찾기
    best_epoch = max(epoch_means.keys(), key=lambda x: epoch_means[x]['auroc'])
    
    # 결과 출력
    print("\n=== Performance at every 10 epochs (averaged over all folds) ===")
    for epoch in sorted(epoch_means.keys()):
        print(f"Epoch {epoch}: AUROC = {epoch_means[epoch]['auroc']:.4f} ± {epoch_stds[epoch]['auroc']:.4f}")
    
    print(f"\nBest average performance achieved at epoch {best_epoch}:")
    print(f"AUROC = {epoch_means[best_epoch]['auroc']:.4f} ± {epoch_stds[best_epoch]['auroc']:.4f}")
    print(f"AUPRC = {epoch_means[best_epoch]['auprc']:.4f} ± {epoch_stds[best_epoch]['auprc']:.4f}")
    print(f"F1 = {epoch_means[best_epoch]['f1']:.4f} ± {epoch_stds[best_epoch]['f1']:.4f}")
    
    # 최종 결과 저장
    total_time = time.time() - start_time
    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print('[FINAL RESULTS] ' + results)
    print(f"Total execution time: {total_time:.2f} seconds")
    
    # 모든 결과를 JSON으로 저장
    results_path = f'/home1/rldnjs16/graph_anomaly_detection/cross_val_results/all_pretrained_bert_{dataset_name}_time_{current_time}_nhead{n_head_BERT}_seed{random_seed}_BERT_epochs{BERT_epochs}_try20.json'
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
