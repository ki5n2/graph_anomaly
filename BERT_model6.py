#%%
'''PRESENT'''
print('이번 BERT 모델 6은 AIDS, BZR, COX2, DHFR에 대한 실험 파일입니다. 마스크 토큰 재구성을 통한 프리-트레인이 이루어지며, 이후 기존과 같이 노드 특성을 재구성하는 모델을 통해 이상을 탐지합니다.')

#%%
'''IMPORTS'''
import os
import re
import sys
import math
import time
import wandb
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch_geometric.utils as utils

from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict, deque

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, to_dense_batch, to_networkx, get_laplacian
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from functools import partial
from scipy.linalg import eigh
from multiprocessing import Pool

from modules.loss import loss_cal
from modules.utils import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original

from sklearn.metrics import silhouette_score
from scipy.linalg import eigh

import networkx as nx


#%%
def train_bert_embedding(model, train_loader, bert_optimizer, device):
    model.train()
    total_loss = 0
    num_sample = 0
    
    for data in train_loader:
        bert_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs, node_label = data.x, data.edge_index, data.batch, data.num_graphs, data.node_label
        node_label = torch.round(node_label).long()

        # 15% 노드 마스킹
        mask_indices = torch.rand(x.size(0), device=device) < 0.15
        
        # BERT 인코딩 및 마스크 토큰 예측
        _, z, _, masked_outputs = model(
            x, edge_index, batch, num_graphs, 
            mask_indices=mask_indices, training=True
        )
        
        # 마스크 예측 손실 계산
        mask_loss = torch.norm(masked_outputs - x[mask_indices], p='fro')**2 / mask_indices.sum()
        
        loss = mask_loss
        print(f'mask_node_feature:{mask_loss}')

        loss.backward()
        bert_optimizer.step()
        
        total_loss += loss.item()
        num_sample += num_graphs
    
    return total_loss / len(train_loader), num_sample, z.detach().cpu()


#%%
def train(model, train_loader, recon_optimizer, max_nodes, device):
    model.train()
    total_loss = 0
    num_sample = 0
    
    for data in train_loader:
        recon_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index = data.x, data.edge_index
        batch, num_graphs = data.batch, data.num_graphs
        
        # 모델 forward pass
        train_cls_outputs, z, z_ = model(x, edge_index, batch, num_graphs)
        
        # 손실 계산
        loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            node_loss = torch.norm(z[start_node:end_node] - z_[start_node:end_node], p='fro')**2 / num_nodes
            loss += node_loss
            start_node = end_node

        print(f'train_node_loss: {loss}')

        num_sample += num_graphs

        loss.backward()
        recon_optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader), num_sample, train_cls_outputs.detach().cpu()


def evaluate_model(model, test_loader, max_nodes, cluster_centers, device):
    model.eval()
    total_loss_ = 0
    total_loss_anomaly_ = 0
    total_loss_mean = 0
    total_loss_anomaly_mean = 0

    all_labels = []
    all_scores = []

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            x, edge_index = data.x, data.edge_index
            batch, num_graphs = data.batch, data.num_graphs

            e_cls_output, z, z_ = model(x, edge_index, batch, num_graphs)

            recon_errors = []
            start_node = 0
            for i in range(num_graphs):
                num_nodes = (batch == i).sum().item()
                end_node = start_node + num_nodes
                
                # 노드 재구성 손실
                node_loss = torch.norm(z[start_node:end_node] - z_[start_node:end_node], p='fro')**2 / num_nodes
                
                # 클러스터 중심과의 거리 계산
                cls_vec = e_cls_output[i].detach().cpu().numpy()
                distances = cdist([cls_vec], cluster_centers, metric='euclidean')
                min_distance = distances.min()

                # 최종 이상 점수 계산
                recon_error = (node_loss.item() * alpha) + (min_distance.item() * gamma)              
                recon_errors.append(recon_error)

                print(f'test_node_loss: {node_loss.item() * alpha}')
                print(f'test_min_distance: {min_distance.item() * gamma}')
                
                if data.y[i].item() == 0:
                    total_loss_ += recon_error
                else:
                    total_loss_anomaly_ += recon_error

                start_node = end_node
            
            # 평균 손실 계산
            total_loss = total_loss_ / max(sum(data.y == 0).item(), 1)
            total_loss_anomaly = total_loss_anomaly_ / max(sum(data.y == 1).item(), 1)
            
            total_loss_mean += total_loss
            total_loss_anomaly_mean += total_loss_anomaly
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())

    # 평가 지표 계산
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

    # 성능 지표 계산
    precision = precision_score(all_labels, pred_labels)
    recall = recall_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)

    return auroc, auprc, precision, recall, f1, total_loss_mean / len(test_loader), total_loss_anomaly_mean / len(test_loader)


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='COX2')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--n-head", type=int, default=8)
parser.add_argument("--n-layer", type=int, default=8)
parser.add_argument("--BERT-epochs", type=int, default=200)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--n-cluster", type=int, default=3)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--random-seed", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=300)
parser.add_argument("--log-interval", type=int, default=5)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])

parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--dropout-rate", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.0001)
parser.add_argument("--learning-rate", type=float, default=0.0001)

parser.add_argument("--alpha", type=float, default=10.0)
parser.add_argument("--beta", type=float, default=0.05)
parser.add_argument("--gamma", type=float, default=0.1)
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


#%%
# class GraphBertNodeEmbedding(nn.Module):
#     """Complete node embedding for GRAPH-BERT"""
#     def __init__(
#         self,
#         feature_dim: int,
#         hidden_dim: int,
#         max_degree: int = 50
#     ):
#         super().__init__()
#         self.hidden_dim = hidden_dim
        
#         # Raw Feature Embedding
#         self.feature_embedding = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim),
#             nn.LayerNorm(hidden_dim),
#             nn.ReLU()
#         )
        
#         # WL Role Embedding
#         self.wl_embedding = nn.Embedding(max_degree, hidden_dim)
        
#         # Position Embedding
#         self.position_embedding = nn.Embedding(1000, hidden_dim)
        
#         # Hop Distance Embedding
#         self.hop_embedding = nn.Embedding(20, hidden_dim)
        
#         # Final projection
#         self.output_projection = nn.Linear(4 * hidden_dim, hidden_dim)
        
#         self.reset_parameters()
        
#     def reset_parameters(self):
#         # 가중치 초기화
#         nn.init.xavier_uniform_(self.feature_embedding[0].weight)
#         nn.init.zeros_(self.feature_embedding[0].bias)
#         nn.init.xavier_uniform_(self.output_projection.weight)
#         nn.init.zeros_(self.output_projection.bias)
        
#     def forward(self, node_features, wl_labels, positions, hop_distances):
#         # 각 임베딩 계산
#         feat_emb = self.feature_embedding(node_features)
#         wl_emb = self.wl_embedding(wl_labels)
#         pos_emb = self.position_embedding(positions)
#         hop_emb = self.hop_embedding(hop_distances)
        
#         # 모든 임베딩 결합
#         combined = torch.cat([feat_emb, wl_emb, pos_emb, hop_emb], dim=-1)
#         return self.output_projection(combined)


#%%
def position_embed(pos, d_model):
    """
    Position-Embed 함수 구현 (논문의 수식 (3) 구현)
    """
    pe = torch.zeros(pos.size(0), d_model, device=pos.device)
    div_term = torch.pow(10000, torch.arange(0, d_model, 2, device=pos.device).float() / d_model)
    
    pe[:, 0::2] = torch.sin(pos.float().unsqueeze(-1) / div_term)
    pe[:, 1::2] = torch.cos(pos.float().unsqueeze(-1) / div_term)
    return pe

class NodeInputEmbedding(nn.Module):
    """Raw feature vector embedding (논문의 수식 (2) 구현)"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.embed = nn.Linear(input_dim, output_dim)  # 원본 특성 -> hidden_dim
    
    def forward(self, x):
        return self.embed(x)


class GRAPH_BERT_EMBEDDINGS(nn.Module):
    def __init__(self, num_features, hidden_dim, max_nodes):
        super().__init__()
        self.max_nodes = max_nodes
        self.hidden_dim = hidden_dim
        
        # (1) Raw feature vector embedding
        self.feature_embedding = NodeInputEmbedding(num_features, hidden_dim)
        
        # (2) WL based absolute role embedding
        self.wl_embed = partial(position_embed, d_model=hidden_dim)
        
        # (3) Intimacy based relative positional embedding
        self.intimacy_embed = partial(position_embed, d_model=hidden_dim)
        
        # (4) Hop based relative distance embedding
        self.hop_embed = partial(position_embed, d_model=hidden_dim)

        # Final projection layer to combine all embeddings
        self.output_projection = nn.Linear(4 * hidden_dim, hidden_dim)

    def get_wl_labels(self, edge_index, num_nodes):
        """WL 알고리즘을 통한 노드 라벨링"""
        adj_list = defaultdict(list)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            adj_list[src].append(dst)
            adj_list[dst].append(src)
        
        # 초기 라벨은 노드의 차수
        labels = torch.tensor([len(adj_list[i]) for i in range(num_nodes)], 
                            device=edge_index.device)
        return labels

    def get_intimacy_positions(self, edge_index, num_nodes):
        """친밀도 기반 상대적 위치 계산"""
        positions = torch.zeros(num_nodes, device=edge_index.device)
        for i in range(num_nodes):
            # 각 노드에 대해 직렬화된 리스트에서의 위치 할당
            positions[i] = i
        return positions

    def get_hop_distances(self, edge_index, num_nodes, source):
        """홉 기반 상대적 거리 계산 H(vj; vi)"""
        adj_list = defaultdict(list)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            adj_list[src].append(dst)
            adj_list[dst].append(src)
            
        distances = torch.full((num_nodes,), float('inf'), device=edge_index.device)
        distances[source] = 0
        queue = deque([source])
        visited = {source}
        
        while queue:
            node = queue.popleft()
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        
        # 무한대 값을 최대 거리로 대체
        max_dist = distances[distances != float('inf')].max().item()
        distances[distances == float('inf')] = max_dist + 1
        return distances

    def forward(self, x, edge_index, batch):
        num_nodes = x.size(0)
        device = x.device
        
        # 1. Raw feature embedding
        e_x = self.feature_embedding(x)
        
        # 2. WL based absolute role embedding
        wl_labels = self.get_wl_labels(edge_index, num_nodes)
        e_wl = self.wl_embed(wl_labels)
        
        # 3. Intimacy based relative positional embedding
        positions = self.get_intimacy_positions(edge_index, num_nodes)
        e_pos = self.intimacy_embed(positions)
        
        # 4. Hop based relative distance embedding
        all_hop_embeddings = []
        unique_batches = torch.unique(batch)
        for b in unique_batches:
            batch_mask = (batch == b)
            batch_nodes = torch.where(batch_mask)[0]
            batch_size = batch_nodes.size(0)
            
            hop_matrix = torch.zeros(batch_size, self.hidden_dim, device=device)
            for i, source in enumerate(batch_nodes):
                hop_distances = self.get_hop_distances(edge_index, num_nodes, source)
                hop_matrix[i] = self.hop_embed(hop_distances[batch_mask]).mean(0)
            
            all_hop_embeddings.append(hop_matrix)
        
        e_hop = torch.cat(all_hop_embeddings, dim=0)
        
        # 모든 임베딩을 결합
        combined = torch.cat([e_x, e_wl, e_pos, e_hop], dim=-1)

        # 최종 프로젝션
        h = self.output_projection(combined)
                
        return h
        
class BertEncoder(nn.Module):
    def __init__(self, num_features, hidden_dims, nhead, num_layers, max_nodes, num_node_classes, dropout_rate=0.1):
        super(BertEncoder, self).__init__()
        self.hidden_dim = hidden_dims[-1]  # 최종 hidden dimension
        
        self.node_embedding = GRAPH_BERT_EMBEDDINGS(
            num_features=num_features,  # 원본 특성 차원
            hidden_dim=self.hidden_dim,  # 최종 hidden dimension
            max_nodes=max_nodes
        )
        
        # Transformer 차원을 hidden_dim으로 설정
        encoder_layer = nn.TransformerEncoderLayer(
            self.hidden_dim, nhead, self.hidden_dim * 4, dropout_rate, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 마스크 토큰도 hidden_dim 크기로
        self.mask_token = nn.Parameter(torch.randn(1, self.hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # hidden_dim에서 원본 특성으로 변환하는 예측기
        self.predicter = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.LayerNorm(self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_dim, num_features)  # 원본 특성 차원으로 출력
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes


    def preprocess_graph(self, edge_index, num_nodes):
        # WL 라벨링 계산
        adj_list = defaultdict(list)
        # 노드 인덱스 재매핑
        unique_nodes = torch.unique(edge_index)
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(unique_nodes)}
        
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            # 재매핑된 인덱스 사용
            src_new = node_map[src]
            dst_new = node_map[dst]
            adj_list[src_new].append(dst_new)
            adj_list[dst_new].append(src_new)
        
        # 초기 라벨은 노드의 차수
        wl_labels = torch.tensor([len(adj_list.get(i, [])) for i in range(num_nodes)], 
                            device=edge_index.device)
        
        # 위치 정보 계산 (BFS 기반)
        positions = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        visited = set()
        if num_nodes > 0:  # 노드가 있는 경우에만 BFS 수행
            queue = deque([0])  # 시작 노드
            pos = 0
            
            while queue and len(visited) < num_nodes:
                node = queue.popleft()
                if node not in visited:
                    positions[node] = pos
                    pos += 1
                    visited.add(node)
                    queue.extend(n for n in adj_list.get(node, []) if n not in visited)
        
        # 홉 거리 계산
        hop_distances = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
        for i in range(num_nodes):
            visited = set()
            queue = deque([(i, 0)])
            while queue:
                node, dist = queue.popleft()
                if node not in visited and node < num_nodes:  # 인덱스 범위 체크 추가
                    hop_distances[node] = min(dist, 19)  # 최대 19 홉으로 제한
                    visited.add(node)
                    # 유효한 이웃 노드만 추가
                    queue.extend((n, dist + 1) for n in adj_list.get(node, []) 
                            if n < num_nodes and n not in visited)
    
        return wl_labels, positions, hop_distances
    

    def forward(self, x, edge_index, batch, mask_indices=None, training=True):
        batch_size = batch.max().item() + 1
        
        # padded_tensor를 hidden_dim 크기로 초기화
        padded_tensor = torch.zeros(batch_size, self.max_nodes, self.hidden_dim, device=x.device)
        padding_mask = torch.ones(batch_size, self.max_nodes, dtype=torch.bool, device=x.device)
        
        if training and mask_indices is not None:
            mask_positions = torch.zeros_like(padding_mask)
        
        node_embeddings = []
        outputs = []
        
        for i in range(batch_size):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            
            # 현재 그래프의 노드와 엣지
            graph_x = x[mask]
            graph_edge_index = get_subgraph_edges(edge_index, mask)
            
            # 노드 임베딩 계산 (hidden_dim 크기의 출력)
            graph_h = self.node_embedding(
                x=graph_x,
                edge_index=graph_edge_index,
                batch=batch[mask]
            )
            
            if training and mask_indices is not None:
                graph_mask_indices = mask_indices[mask]
                graph_h[graph_mask_indices] = self.mask_token
                mask_positions[i, :num_nodes][graph_mask_indices] = True
                
                if graph_mask_indices.any():
                    masked_nodes = graph_h[graph_mask_indices]
                    masked_preds = self.predicter(masked_nodes)
                    outputs.append(masked_preds)
            
            padded_tensor[i, :num_nodes] = graph_h
            padding_mask[i, :num_nodes] = False
            node_embeddings.append(graph_h)
        
        # Transformer 처리
        transformed = self.transformer(
            padded_tensor.transpose(0, 1),
            src_key_padding_mask=padding_mask
        ).transpose(0, 1)
        
        # 결과 수집
        final_embeddings = []
        for i in range(batch_size):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            final_embeddings.append(transformed[i, :num_nodes])
        
        final_embeddings = torch.cat(final_embeddings, dim=0)
        
        if training and mask_indices is not None and outputs:
            outputs = torch.cat(outputs, dim=0)
            return final_embeddings, outputs
        
        return final_embeddings



class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_nodes, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask):
        src = src.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = output.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        return output


def get_subgraph_edges(edge_index, mask):
    """서브그래프의 엣지 인덱스 추출"""
    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    return edge_index[:, edge_mask]


def perform_clustering(train_cls_outputs, random_seed, n_clusters):
    # train_cls_outputs가 이미 텐서이므로, 그대로 사용
    cls_outputs_tensor = train_cls_outputs  # [total_num_graphs, hidden_dim]
    cls_outputs_np = cls_outputs_tensor.detach().cpu().numpy()
    
    # K-Means 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto").fit(cls_outputs_np)

    # 클러스터 중심 저장
    cluster_centers = kmeans.cluster_centers_

    return kmeans, cluster_centers


#%%
class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, nhead, num_layers, num_node_labels, dropout_rate=0.1):
        super(GRAPH_AUTOENCODER, self).__init__()
        # self.gcn_encoder = Encoder(num_features, hidden_dims, dropout_rate)
        
        # BERT 인코더
        self.encoder = BertEncoder(
            num_features=num_features,
            hidden_dims=hidden_dims,
            nhead=nhead,
            num_layers=num_layers,
            max_nodes=max_nodes,
            num_node_classes=num_node_labels,
            dropout_rate=dropout_rate
        )
        
        self.transformer = TransformerEncoder(
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
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        self.sigmoid = nn.Sigmoid()
        
        # 가중치 초기화
        self.apply(self._init_weights)

    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, training=True):
        # # GCN 인코딩
        # h = self.gcn_encoder(x, edge_index)
        # h = self.dropout(h)
        
        # BERT 인코딩
        if training and mask_indices is not None:
            z, masked_outputs = self.encoder(
                x, edge_index, batch, mask_indices, training=True
            )
        else:
            z = self.encoder(
                x, edge_index, batch, training=False
            )
 
        print(mask_indices)
        print(training)
        
        # 그래프별 노드 임베딩 리스트 생성
        z_list = [z[batch == i] for i in range(num_graphs)]
        edge_index_list = []
        start_idx = 0
        
        # 그래프별 엣지 인덱스 처리
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            mask = (batch == i)
            graph_edges = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < start_idx + num_nodes)]
            graph_edges = graph_edges - start_idx
            edge_index_list.append(graph_edges)
            start_idx += num_nodes
        
        # CLS 토큰 추가 및 패딩 처리
        z_with_cls_list = []
        mask_list = []
        max_nodes_in_batch = max(z_graph.size(0) for z_graph in z_list)
        
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            cls_token = self.cls_token.repeat(1, 1, 1)
            cls_token = cls_token.to(x.device)
            z_graph = z_list[i].unsqueeze(1)
            
            pad_size = max_nodes_in_batch - num_nodes
            z_graph_padded = F.pad(z_graph, (0, 0, 0, 0, 0, pad_size), 'constant', 0)
            z_with_cls = torch.cat([cls_token, z_graph_padded.transpose(0, 1)], dim=1)
            z_with_cls_list.append(z_with_cls)

            graph_mask = torch.cat([torch.tensor([False]), torch.tensor([False]*num_nodes + [True]*pad_size)])
            mask_list.append(graph_mask)

        # 배치 처리
        z_with_cls_batch = torch.cat(z_with_cls_list, dim=0)
        mask = torch.stack(mask_list).to(x.device)
        
        # Transformer 인코딩
        encoded = self.transformer(z_with_cls_batch, mask)

        # CLS 토큰과 노드 출력 분리
        cls_output = encoded[:, 0, :]
        node_output = encoded[:, 1:, :]
        
        # 노드별 출력 처리
        node_output_list = []
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            node_output_list.append(node_output[i, :num_nodes, :])

        u = torch.cat(node_output_list, dim=0)
        u_prime = self.u_mlp(u)
        
        # 특성 재구성
        x_recon = self.feature_decoder(u_prime)
        
        # 최종 임베딩 생성
        # h_ = self.gcn_encoder(x_recon, edge_index)
        # h_ = self.dropout(h_)
        z_ = self.encoder(x_recon, edge_index, batch, training=False)
        
        if training and mask_indices is not None:
            return cls_output, z, z_, masked_outputs
        
        return cls_output, z, z_

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


#%%
'''RUN'''
def run(dataset_name, random_seed, dataset_AN, trial, device=device):
    all_results = []
    set_seed(random_seed)
    split = splits[trial]
    
    # 데이터 로더 초기화
    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN)
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']
    max_node_label = meta['max_node_label']
    
    # BERT 모델 저장 경로
    bert_save_path = f'/home1/rldnjs16/graph_anomaly_detection/BERT_model/pretrained_bert_{dataset_name}_fold{trial}_n_head{n_head}_seed{random_seed}_BERT_epochs{BERT_epochs}_try4.pth'
    
    # 모델 초기화
    model = GRAPH_AUTOENCODER(
        num_features=num_features, 
        hidden_dims=hidden_dims, 
        max_nodes=max_nodes,
        nhead=n_head,
        num_layers=n_layer,
        num_node_labels=max_node_label,
        dropout_rate=dropout_rate
    ).to(device)
    
    train_loader = loaders['train']
    test_loader = loaders['test']
    
    global train_cls_outputs
    train_cls_outputs = []
    
    # 1단계: BERT 임베딩 학습
    if os.path.exists(bert_save_path):
        print("Loading pretrained BERT...")
        model.encoder.load_state_dict(torch.load(bert_save_path))
    else:
        print("Training BERT from scratch...")
        print("Stage 1: Training BERT embeddings...")
        pretrain_params = list(model.encoder.parameters())
        bert_optimizer = torch.optim.Adam(pretrain_params, lr=learning_rate)
        
        for epoch in range(1, BERT_epochs+1):
            train_loss, num_sample, node_embeddings = train_bert_embedding(
                model, train_loader, bert_optimizer, device
            )
            
            if epoch % log_interval == 0:
                print(f'BERT Training Epoch {epoch}: Loss = {train_loss:.4f}')
                
        print("Saving pretrained BERT...")
        torch.save(model.encoder.state_dict(), bert_save_path)
    
    # 2단계: 재구성 학습
    print("\nStage 2: Training reconstruction...")
    recon_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(1, epochs+1):
        fold_start = time.time()
        train_loss, num_sample, train_cls_outputs = train(
            model, train_loader, recon_optimizer, max_nodes, device
        )
        
        info_train = 'Epoch {:3d}, Loss {:.4f}'.format(epoch, train_loss)
        
        if epoch % log_interval == 0:
            cluster_centers = train_cls_outputs.mean(dim=0)
            cluster_centers = cluster_centers.detach().cpu().numpy()
            cluster_centers = cluster_centers.reshape(-1, hidden_dims[-1])

            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly = evaluate_model(
                model, test_loader, max_nodes, cluster_centers, device
            )
            
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

    return auroc


#%%
'''MAIN'''
if __name__ == '__main__':
    ad_aucs = []
    fold_times = []
    splits = get_ad_split_TU(dataset_name, n_cross_val)    

    start_time = time.time()  # 전체 실행 시작 시간

    for trial in range(n_cross_val):
        fold_start = time.time()  # 현재 폴드 시작 시간

        print(f"Starting fold {trial + 1}/{n_cross_val}")
        ad_auc = run(dataset_name, random_seed, dataset_AN, trial)
        ad_aucs.append(ad_auc)
        
        fold_end = time.time()  # 현재 폴드 종료 시간
        fold_duration = fold_end - fold_start  # 현재 폴드 실행 시간
        fold_times.append(fold_duration)
        
        print(f"Fold {trial + 1} finished in {fold_duration:.2f} seconds.")
        
    total_time = time.time() - start_time  # 전체 실행 시간
    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print(len(ad_aucs))
    print('[FINAL RESULTS] ' + results)
    print(f"Total execution time: {total_time:.2f} seconds")


wandb.finish()

# %%
