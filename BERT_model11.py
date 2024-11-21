#%%
'''PRESENT'''
print('이번 BERT 모델 10은 AIDS, BZR, COX2, DHFR에 대한 실험 파일입니다. 마스크 토큰 재구성을 통한 프리-트레인이 이루어집니다. 이후 기존과 같이 노드 특성을 재구성하는 모델을 통해 이상을 탐지합니다. 기존 파일과 다른 점은 성능 평가 결과 비교를 코드 내에서 수행하고자 하였으며, 해당 파일만 실행하면 결과까지 한 번에 볼 수 있도록 하였습니다. 또한, 재구성 오류에 대한 정규화가 이루어져 있습니다. 훈련 데이터에 대한 산점도와 로그 스케일이 적용되어 있습니다. 또한, 클러스터링 품질 평가와 더불어 하이라키컬 클러스터링이 적용되어 있습니다.')

#%%
'''IMPORTS'''
import os
import re
import sys
import json
import math
import time
import wandb
import torch
import random
import argparse
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch_geometric.utils as utils

from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, OneCycleLR
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.utils import to_networkx, get_laplacian, to_dense_adj, to_dense_batch

from scipy.linalg import eigh
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster, cophenet

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score, silhouette_score, silhouette_samples

from functools import partial
from multiprocessing import Pool

from modules.loss import loss_cal
from modules.utils import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original

import networkx as nx


#%%
'''TRAIN BERT'''
def train_bert_embedding(model, train_loader, bert_optimizer, device):
    model.train()
    total_loss = 0
    num_sample = 0
    
    for data in train_loader:
        bert_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        
        # 마스크 생성
        mask_indices = torch.rand(x.size(0), device=device) < 0.15  # 15% 노드 마스킹
        
        # BERT 인코딩 및 마스크 토큰 예측만 수행
        _, _, masked_outputs = model(
            x, edge_index, batch, num_graphs, mask_indices, training=True, edge_training=False
        )
        
        mask_loss = torch.norm(masked_outputs - x[mask_indices], p='fro')**2 / mask_indices.sum()
        loss = mask_loss
        print(f'mask_node_feature:{mask_loss}')
        
        loss.backward()
        bert_optimizer.step()
        total_loss += loss.item()
        num_sample += num_graphs
    
    return total_loss / len(train_loader), num_sample


#%%
def train(model, train_loader, recon_optimizer, max_nodes, device, epoch):
    model.train()
    total_loss = 0
    num_sample = 0
    reconstruction_errors = []
    
    for data in train_loader:
        recon_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        
        train_cls_outputs, x_recon = model(x, edge_index, batch, num_graphs)
        
        if epoch % 5 == 0:
            cluster_centers = train_cls_outputs.mean(dim=0)
            cluster_centers = cluster_centers.detach().cpu().numpy()
            cluster_centers = cluster_centers.reshape(-1, hidden_dims[-1])
            
        loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            node_loss = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
            loss += node_loss
            
            if epoch % 5 == 0:
                node_loss_scaled = node_loss.item() * alpha
                cls_vec = train_cls_outputs[i].detach().cpu().numpy()
                distances = cdist([cls_vec], cluster_centers, metric='euclidean')
                min_distance = distances.min().item() * gamma
                
                reconstruction_errors.append({
                    'reconstruction': node_loss_scaled,
                    'clustering': min_distance
                })
            
            start_node = end_node
        
        num_sample += num_graphs
        loss.backward()
        recon_optimizer.step()
        total_loss += loss.item()

    if epoch % 5 == 0:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        recon_errors = [point['reconstruction'] for point in reconstruction_errors]
        cluster_errors = [point['clustering'] for point in reconstruction_errors]
        
        # Normal scale plot
        ax1.scatter(recon_errors, cluster_errors, c='blue', alpha=0.6)
        ax1.set_xlabel('Reconstruction Error (node_loss * alpha)')
        ax1.set_ylabel('Clustering Distance (min_distance * gamma)')
        ax1.set_title(f'Training Error Distribution (Epoch {epoch})')
        ax1.grid(True)

        # Log scale plot
        ax2.scatter(recon_errors, cluster_errors, c='blue', alpha=0.6)
        ax2.set_xlabel('Reconstruction Error (node_loss * alpha)')
        ax2.set_ylabel('Clustering Distance (min_distance * gamma)')
        ax2.set_title(f'Training Error Distribution - Log Scale (Epoch {epoch})')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True)

        plt.tight_layout()
        save_path = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/plot/{dataset_name}_time_{current_time}/train_error_distribution_epoch_{epoch}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    return total_loss / len(train_loader), num_sample, train_cls_outputs.detach().cpu()


#%%
def evaluate_model(model, test_loader, cluster_centers, device):
    model.eval()
    total_loss_ = 0
    total_loss_anomaly_ = 0
    all_labels = []
    all_scores = []
    reconstruction_errors = []  # 새로 추가
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
            e_cls_output, x_recon = model(x, edge_index, batch, num_graphs)

            recon_errors = []
            start_node = 0
            
            for i in range(num_graphs):
                num_nodes = (batch == i).sum().item()
                end_node = start_node + num_nodes
                
                # Reconstruction error 계산
                node_loss = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
                node_loss = node_loss.item() * alpha
                
                cls_vec = e_cls_output[i].detach().cpu().numpy()
                distances = cdist([cls_vec], cluster_centers, metric='euclidean')
                min_distance = distances.min().item() * gamma
                
                # 변환된 값들 저장
                reconstruction_errors.append({
                    'reconstruction': node_loss,
                    'clustering': min_distance,
                    'is_anomaly': data.y[i].item() == 1
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
            for error in reconstruction_errors if not error['is_anomaly']
        ],
        'anomaly': [
            {'reconstruction': error['reconstruction'], 
             'clustering': error['clustering']}
            for error in reconstruction_errors if error['is_anomaly']
        ]
    }
    
    # 메트릭 계산
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
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

    return auroc, auprc, precision, recall, f1, total_loss_mean, total_loss_anomaly_mean, visualization_data


#%%
def analyze_cls_embeddings(model, test_loader, epoch, device, n_components=2):
    """
    CLS token embeddings를 추출하고 PCA를 적용하여 더 자세한 분석 수행
    
    Args:
        model: 학습된 모델
        test_loader: 테스트 데이터 로더
        device: 연산 장치
        n_components: PCA 차원 수
    """
    model.eval()
    cls_embeddings = []
    labels = []
    
    # CLS embeddings 추출
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
            
            cls_output, _ = model(x, edge_index, batch, num_graphs)
            cls_embeddings.append(cls_output.cpu())
            labels.extend(data.y.cpu().numpy())
    
    # 텐서를 numpy 배열로 변환
    cls_embeddings = torch.cat(cls_embeddings, dim=0).numpy()
    labels = np.array(labels)
    
    # PCA 적용
    pca = PCA()
    cls_embeddings_pca = pca.fit_transform(cls_embeddings)
    
    # 분산 설명률 계산
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    eigenvalues = pca.explained_variance_
    
    # 시각화
    plt.figure(figsize=(20, 15))
    
    # 1. PCA 산점도 (첫 2개 주성분)
    plt.subplot(221)
    scatter = plt.scatter(cls_embeddings_pca[:, 0], cls_embeddings_pca[:, 1], 
                         c=labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, label='Label (0: Normal, 1: Anomaly)')
    plt.xlabel(f'PC1 (variance ratio: {explained_variance_ratio[0]:.3f})')
    plt.ylabel(f'PC2 (variance ratio: {explained_variance_ratio[1]:.3f})')
    plt.title('PCA of CLS Token Embeddings')
    
    # 2. 누적 분산 설명률
    plt.subplot(222)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), 
             cumulative_variance_ratio, 'bo-')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance Ratio')
    plt.grid(True)
    plt.legend()
    
    # 3. 스크린 플롯
    plt.subplot(223)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.grid(True)
    
    # 4. 로그 스케일 스크린 플롯
    plt.subplot(224)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
    plt.yscale('log')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalue (log scale)')
    plt.title('Scree Plot (Log Scale)')
    plt.grid(True)
    
    plt.tight_layout()
    
    # 분석 결과 저장
    save_path = f'/home1/rldnjs16/graph_anomaly_detection/pca_analysis/cls_pca_analysis_{dataset_name}_fold_{trial}_ep_{epoch}_{current_time}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    # 주요 지표 계산
    n_components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1
    n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
    
    # 분석 결과 출력
    print("\nPCA Analysis Results:")
    print(f"Number of features in original space: {cls_embeddings.shape[1]}")
    print(f"Number of components needed for 90% variance: {n_components_90}")
    print(f"Number of components needed for 95% variance: {n_components_95}")
    print("\nTop 5 Principal Components:")
    for i in range(5):
        print(f"PC{i+1}: {explained_variance_ratio[i]:.4f} "
              f"(cumulative: {cumulative_variance_ratio[i]:.4f})")
    
    # Class separation analysis
    normal_indices = labels == 0
    anomaly_indices = labels == 1
    
    pc1_normal = cls_embeddings_pca[normal_indices, 0]
    pc1_anomaly = cls_embeddings_pca[anomaly_indices, 0]
    pc2_normal = cls_embeddings_pca[normal_indices, 1]
    pc2_anomaly = cls_embeddings_pca[anomaly_indices, 1]
    
    print("\nClass Separation Analysis:")
    print("PC1 - Normal vs Anomaly:")
    print(f"Normal mean: {np.mean(pc1_normal):.4f}, std: {np.std(pc1_normal):.4f}")
    print(f"Anomaly mean: {np.mean(pc1_anomaly):.4f}, std: {np.std(pc1_anomaly):.4f}")
    print("\nPC2 - Normal vs Anomaly:")
    print(f"Normal mean: {np.mean(pc2_normal):.4f}, std: {np.std(pc2_normal):.4f}")
    print(f"Anomaly mean: {np.mean(pc2_anomaly):.4f}, std: {np.std(pc2_anomaly):.4f}")
    
    return cls_embeddings_pca, explained_variance_ratio, eigenvalues


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
parser.add_argument("--BERT-epochs", type=int, default=100)
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
parser.add_argument("--beta", type=float, default=0.05)
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
class BertEncoder(nn.Module):
    def __init__(self, num_features, hidden_dims, d_model, nhead, num_layers, max_nodes, dropout_rate=0.1):
        super().__init__()
        self.gcn_encoder = Encoder(num_features, hidden_dims, dropout_rate)
        self.positional_encoding = GraphBertPositionalEncoding(hidden_dims[-1], max_nodes)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dims[-1], nhead, hidden_dims[-1] * 4, dropout_rate, activation='gelu', batch_first = True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mask_token = nn.Parameter(torch.randn(1, hidden_dims[-1]))
        self.predicter = nn.Linear(hidden_dims[-1], num_features)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dims[-1]))
        self.edge_decoder = BilinearEdgeDecoder(max_nodes)
        
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        self.d_model = d_model
        
        # 가중치 초기화
        self.apply(self._init_weights)

    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, training=False, edge_training=False):
        h = self.gcn_encoder(x, edge_index)
        
        # 배치 처리
        z_list, edge_index_list, max_nodes_in_batch = BatchUtils.process_batch(h, edge_index, batch)
        
        # 3. 각 그래프에 대해 포지셔널 인코딩 계산
        pos_encoded_list = []
        for i, (z_graph, edge_idx) in enumerate(zip(z_list, edge_index_list)):
            # 포지셔널 인코딩 계산
            pos_encoding = self.positional_encoding(edge_idx, z_graph.size(0))
            # 인코딩 적용
            z_graph_with_pos = z_graph + pos_encoding
            pos_encoded_list.append(z_graph_with_pos)
            
        z_with_cls_batch, padding_mask = BatchUtils.add_cls_token(
            pos_encoded_list, self.cls_token, max_nodes_in_batch, x.device
        )
                
        # 5. 마스킹 적용
        if training and mask_indices is not None:
            mask_positions = torch.zeros_like(padding_mask)
            start_idx = 0
            for i in range(len(z_list)):
                num_nodes = z_list[i].size(0)
                graph_mask_indices = mask_indices[start_idx:start_idx + num_nodes]
                mask_positions[i, 1:num_nodes+1] = graph_mask_indices
                node_indices = mask_positions[i].nonzero().squeeze(-1)
                # mask_token = mask_token.to(device)
                z_with_cls_batch[i, node_indices] = self.mask_token
                padding_mask[i, num_nodes+1:] = True
                start_idx += num_nodes
                
        # Transformer 처리
        transformed = self.transformer(
            z_with_cls_batch,
            src_key_padding_mask=padding_mask
        )
        
        if edge_training == False:
            node_embeddings, masked_outputs = self._process_outputs(
                transformed, batch, mask_positions if training and mask_indices is not None else None
                )
        else:
            # 결과 추출
            node_embeddings, _ = self._process_outputs(
                transformed, batch, mask_positions=None
            )
        
        if training and edge_training:
            adj_recon_list = []
            idx = 0
            for i in range(num_graphs):
                num_nodes = z_list[i].size(0)
                z_graph = node_embeddings[idx:idx + num_nodes]
                adj_recon = self.edge_decoder(z_graph)
                adj_recon_list.append(adj_recon)
                idx += num_nodes
            
        if training:
            if edge_training:
                return node_embeddings, adj_recon_list
            else:
                return node_embeddings, masked_outputs

        return node_embeddings

    def _apply_masking(self, z_with_cls_batch, padding_mask, batch, mask_indices):
        batch_size = z_with_cls_batch.size(0)
        mask_positions = torch.zeros_like(padding_mask)
        start_idx = 0
        
        for i in range(batch_size):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            graph_mask_indices = mask_indices[start_idx:start_idx + num_nodes]
            mask_positions[i, 1:num_nodes+1] = graph_mask_indices
            node_indices = mask_positions[i].nonzero().squeeze(-1)
            z_with_cls_batch[i, node_indices] = self.mask_token
            padding_mask[i, num_nodes+1:] = True
            start_idx += num_nodes
            
        return mask_positions

    def _process_outputs(self, transformed, batch, mask_positions=None):
        node_embeddings = []
        masked_outputs = []
        batch_size = transformed.size(0)
        start_idx = 0
        
        for i in range(batch_size):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            # CLS 토큰을 제외한 노드 임베딩 추출
            graph_encoded = transformed[i, 1:num_nodes+1]
            node_embeddings.append(graph_encoded)
            
            # 전체 노드에 대해 예측 수행
            all_predictions = self.predicter(graph_encoded)
            
            # 마스크된 위치의 예측값만 선택
            if mask_positions is not None:
                current_mask_positions = mask_positions[i, 1:num_nodes+1]
                if current_mask_positions.any():
                    masked_predictions = all_predictions[current_mask_positions]
                    masked_outputs.append(masked_predictions)
            
            start_idx += num_nodes
        
        node_embeddings = torch.cat(node_embeddings, dim=0)
        
        # 전체 예측값과 마스크된 위치의 예측값 반환
        if mask_positions is not None and masked_outputs:
            return node_embeddings, torch.cat(masked_outputs, dim=0)
        return node_embeddings, None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.zeros_(module.bias)
    
    
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


class GraphBertPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_nodes):
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes
        
        self.wsp_encoder = nn.Linear(max_nodes, d_model // 2)
        self.le_encoder = nn.Linear(max_nodes, d_model // 2)
        
    def get_wsp_encoding(self, edge_index, num_nodes):
        edge_index_np = edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(zip(edge_index_np[0], edge_index_np[1]))
        
        spl_matrix = torch.zeros((num_nodes, self.max_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    try:
                        path_length = nx.shortest_path_length(G, source=i, target=j)
                    except nx.NetworkXNoPath:
                        path_length = self.max_nodes
                    if j < self.max_nodes:
                        spl_matrix[i, j] = path_length
                        
        return spl_matrix.to(edge_index.device)
    
    def get_laplacian_encoding(self, edge_index, num_nodes):
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
        L = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).to_dense()
        
        L_np = L.cpu().numpy()
        _, eigenvecs = eigh(L_np)
        le_matrix = torch.from_numpy(eigenvecs).float().to(edge_index.device)
        
        padded_le = torch.zeros((num_nodes, self.max_nodes), device=edge_index.device)
        padded_le[:, :num_nodes] = le_matrix
        
        return padded_le
    
    def forward(self, edge_index, num_nodes):
        wsp_matrix = self.get_wsp_encoding(edge_index, num_nodes)
        wsp_encoding = self.wsp_encoder(wsp_matrix)
        
        le_matrix = self.get_laplacian_encoding(edge_index, num_nodes)
        le_encoding = self.le_encoder(le_matrix)
        
        return torch.cat([wsp_encoding, le_encoding], dim=-1)
    

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
class HierarchicalClusterAnalyzer:
    def __init__(self, distance_threshold=None, max_clusters=None):
        """
        Parameters:
        -----------
        distance_threshold : float or None
            클러스터 간 최대 거리 임계값
        max_clusters : int or None
            최대 클러스터 수 제한
        """
        self.distance_threshold = distance_threshold
        self.max_clusters = max_clusters
        
    def fit_predict(self, cls_embeddings, epoch=None):
        """
        계층적 클러스터링 수행 및 분석
        
        Parameters:
        -----------
        cls_embeddings : torch.Tensor or numpy.ndarray
            CLS 임베딩 벡터들
        epoch : int or None
            현재 에폭 (시각화 저장용)
            
        Returns:
        --------
        cluster_centers : numpy.ndarray
            클러스터 중심점들
        clustering_metrics : dict
            클러스터링 관련 메트릭들
        """
        # 텐서를 numpy로 변환
        if torch.is_tensor(cls_embeddings):
            cls_embeddings = cls_embeddings.detach().cpu().numpy()
            
        # 계층적 클러스터링 수행
        linkage_matrix = linkage(cls_embeddings, method='ward')
        
        # 최적의 클러스터 수 또는 거리 임계값 결정
        if self.distance_threshold is None:
            self.distance_threshold = self._find_optimal_threshold(linkage_matrix)
            
        # 클러스터 할당
        cluster_labels = fcluster(linkage_matrix, 
                                t=self.distance_threshold, 
                                criterion='distance')
        
        # 클러스터 중심 계산
        n_clusters = len(np.unique(cluster_labels))
        cluster_centers = np.zeros((n_clusters, cls_embeddings.shape[1]))
        for i in range(n_clusters):
            mask = (cluster_labels == i + 1)  # fcluster는 1부터 시작
            if np.any(mask):
                cluster_centers[i] = cls_embeddings[mask].mean(axis=0)
                
        # 클러스터링 품질 평가
        if n_clusters > 1:
            silhouette_avg = silhouette_score(cls_embeddings, cluster_labels)
        else:
            silhouette_avg = 0
            
        # 덴드로그램 시각화
        self._plot_dendrogram(linkage_matrix, epoch)
        
        # 클러스터 크기 분포 시각화
        self._plot_cluster_sizes(cluster_labels, epoch)
        
        cophenetic_corr, p_value = cophenet(linkage_matrix, pdist(cls_embeddings))

        clustering_metrics = {
            'n_clusters': n_clusters,
            'silhouette_score': silhouette_avg,
            'cluster_sizes': [np.sum(cluster_labels == i + 1) for i in range(n_clusters)],
            'distance_threshold': self.distance_threshold,
            'cophenetic_correlation': cophenetic_corr,
            'cophenetic_p_value': p_value
        }
        
        return cluster_centers, clustering_metrics
        
    def _find_optimal_threshold(self, linkage_matrix):
        """
        클러스터 간 거리의 급격한 변화를 찾아 최적의 임계값 결정
        """
        # 연속된 거리 간의 차이 계산
        distances = linkage_matrix[:, 2]
        diffs = np.diff(distances)
        
        # 거리 차이가 가장 큰 지점 찾기 (elbow method)
        threshold_idx = np.argmax(diffs) + 1
        return distances[threshold_idx]
    
    def _plot_dendrogram(self, linkage_matrix, epoch):
        """덴드로그램 시각화"""
        plt.figure(figsize=(15, 7))
        dendrogram(linkage_matrix)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        
        # 최적 임계값 표시
        plt.axhline(y=self.distance_threshold, color='r', linestyle='--', 
                   label=f'Distance Threshold: {self.distance_threshold:.2f}')
        plt.legend()
        
        # 저장
        save_path = f'/home1/rldnjs16/graph_anomaly_detection/hierarchical_clustering/dendrogram_dataset_name_{dataset_name}_epoch_{epoch}_time_{current_time}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        
    def _plot_cluster_sizes(self, cluster_labels, epoch):
        """클러스터 크기 분포 시각화"""
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(counts)), counts)
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster Index')
        plt.ylabel('Number of Samples')
        
        # 저장
        save_path = f'/home1/rldnjs16/graph_anomaly_detection/hierarchical_clustering/cluster_sizes_epoch_{epoch}.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()


# perform_clustering 함수 수정
def perform_clustering(train_cls_outputs, random_seed, epoch):
    analyzer = HierarchicalClusterAnalyzer()
    cluster_centers, clustering_metrics = analyzer.fit_predict(train_cls_outputs, epoch)
    
    # wandb에 메트릭 기록
    wandb.log({
        'n_clusters': clustering_metrics['n_clusters'],
        'silhouette_score': clustering_metrics['silhouette_score'],
        'distance_threshold': clustering_metrics['distance_threshold'],
        'cluster_sizes': clustering_metrics['cluster_sizes'],
        'cophenetic_corr': clustering_metrics['cophenetic_correlation'],
        'cophenetic_p_value': clustering_metrics['cophenetic_p_value'],
        'epoch': epoch
    })

    return cluster_centers, clustering_metrics

            
#%%
# GRAPH_AUTOENCODER 클래스 수정
class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, nhead_BERT, nhead, num_layers_BERT, num_layers, dropout_rate=0.1):
        super().__init__()
        self.encoder = BertEncoder(
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

    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, training=False, edge_training=False):
        # BERT 인코딩
        if training:
            if edge_training:
                z, adj_recon_list = self.encoder(x, edge_index, batch, num_graphs, mask_indices, training=True, edge_training=True)
            else:
                z, masked_outputs = self.encoder(x, edge_index, batch, num_graphs, mask_indices, training=True, edge_training=False)
        else:
            z = self.encoder(x, edge_index, batch, num_graphs, training=False, edge_training=False)
        
        # 배치 처리
        z_list, edge_index_list, max_nodes_in_batch = BatchUtils.process_batch(z, edge_index, batch, num_graphs)
        z_with_cls_batch, mask = BatchUtils.add_cls_token(
            z_list, self.cls_token, max_nodes_in_batch, x.device
        )
        
        # Transformer 처리 - 수정된 부분
        encoded = self.transformer_d(z_with_cls_batch, mask)  # edge_index_list 제거
        
        # 출력 처리
        cls_output = encoded[:, 0, :]
        node_outputs = [encoded[i, 1:z_list[i].size(0)+1, :] for i in range(num_graphs)]
        u = torch.cat(node_outputs, dim=0)
        
        # 디코딩
        u_prime = self.u_mlp(u)
        x_recon = self.feature_decoder(u_prime)
        
        if training:
            if edge_training:
                return cls_output, x_recon, adj_recon_list
            else:
                return cls_output, x_recon, masked_outputs
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

# %%
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
    bert_save_path = f'/home1/rldnjs16/graph_anomaly_detection/BERT_model/Class/all_pretrained_bert_{dataset_name}_fold{trial}_nhead{n_head_BERT}_seed{random_seed}_BERT_epochs{BERT_epochs}_gcnall{hidden_dims[-1]}_try9.pth'
    
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
            train_loss, num_sample = train_bert_embedding(
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
        train_loss, num_sample, train_cls_outputs = train(model, train_loader, recon_optimizer, max_nodes, device, epoch)
        
        info_train = 'Epoch {:3d}, Loss {:.4f}'.format(epoch, train_loss)

        if epoch % log_interval == 0:
            cluster_centers, clustering_metrics = perform_clustering(train_cls_outputs, random_seed, epoch)

            print(f"\nClustering Analysis Results (Epoch {epoch}):")
            print(f"Number of clusters: {clustering_metrics['n_clusters']}")
            print(f"Silhouette Score: {clustering_metrics['silhouette_score']:.4f}")
            print(f"Distance threshold: {clustering_metrics['distance_threshold']:.4f}")
            print(f"cophenetic_correlation: {clustering_metrics['cophenetic_correlation']:.4f}")
            print("cophenetic_p_value", clustering_metrics['cophenetic_p_value'])
            print("Cluster sizes:", clustering_metrics['cluster_sizes'])

            # auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly, visualization_data = evaluate_model(model, test_loader, max_nodes, cluster_centers, device)
            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly, visualization_data = evaluate_model(model, test_loader, cluster_centers, device)

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

            # run() 함수 내에서 평가 시점에 다음 코드 추가
            cls_embeddings_pca, explained_variance_ratio, eigenvalues = analyze_cls_embeddings(model, test_loader, epoch, device)

    return auroc, epoch_results, cls_embeddings_pca, explained_variance_ratio, eigenvalues


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
        ad_auc, epoch_results, cls_embeddings_pca, explained_variance_ratio, eigenvalues = run(dataset_name, random_seed, dataset_AN, trial, device=device, epoch_results=epoch_results)
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
    results_path = f'/home1/rldnjs16/graph_anomaly_detection/cross_val_results/all_pretrained_bert_{dataset_name}_time_{current_time}_nhead{n_head_BERT}_seed{random_seed}_BERT_epochs{BERT_epochs}_gcnall{hidden_dims[-1]}_try9_2.json'
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump({
            'epoch_means': epoch_means,
            'epoch_stds': epoch_stds,
            'best_epoch': int(best_epoch),
            'final_auroc_mean': float(np.mean(ad_aucs)),
            'final_auroc_std': float(np.std(ad_aucs))
        }, f, indent=4)


#%%
base_dir = f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/json/{dataset_name}_time_{current_time}/'

for trial in range(n_cross_val):    
    for epoch in range(1, epochs + 1):    
        if epoch % log_interval == 0:
            with open(base_dir + f'error_distribution_epoch_{epoch}_fold_{trial}.json', 'r') as f:
                data = json.load(f)

            # 데이터 추출
            normal_recon = [point['reconstruction'] for point in data['normal']]
            normal_cluster = [point['clustering'] for point in data['normal']]
            anomaly_recon = [point['reconstruction'] for point in data['anomaly']]
            anomaly_cluster = [point['clustering'] for point in data['anomaly']]

            # 두 개의 서브플롯 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
            
            # 일반 스케일 플롯
            ax1.scatter(normal_recon, normal_cluster, c='blue', label='Normal', alpha=0.6)
            ax1.scatter(anomaly_recon, anomaly_cluster, c='red', label='Anomaly', alpha=0.6)
            ax1.set_xlabel('Reconstruction Error (node_loss * alpha)')
            ax1.set_ylabel('Clustering Distance (min_distance * gamma)')
            ax1.set_title('Error Distribution')
            ax1.legend()
            ax1.grid(True)

            # 로그 스케일 플롯
            ax2.scatter(normal_recon, normal_cluster, c='blue', label='Normal', alpha=0.6)
            ax2.scatter(anomaly_recon, anomaly_cluster, c='red', label='Anomaly', alpha=0.6)
            ax2.set_xlabel('Reconstruction Error (node_loss * alpha)')
            ax2.set_ylabel('Clustering Distance (min_distance * gamma)')
            ax2.set_title('Error Distribution (Log Scale)')
            ax2.set_xscale('log')
            ax2.set_yscale('log')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plt.savefig(f'/home1/rldnjs16/graph_anomaly_detection/error_distribution_plot/plot/{dataset_name}_time_{current_time}/error_distribution_plot_epoch_{epoch}_fold_{trial}.png')
            plt.close()


wandb.finish()

# %%
