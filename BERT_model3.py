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

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, OneCycleLR
from torch_geometric.utils import to_networkx, get_laplacian, to_dense_adj, to_dense_batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from functools import partial
from scipy.linalg import eigh
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
        x, edge_index, batch, num_graphs, node_label = data.x, data.edge_index, data.batch, data.num_graphs, data.node_label
        node_label = torch.round(node_label).long()

        mask_indices = torch.rand(x.size(0), device=device) < 0.15  # 15% 노드 마스킹
        
        _, _, _, z, masked_outputs_ = model(
            x, edge_index, batch, num_graphs, mask_indices, training=True
        )
        
        # mask_loss = F.mse_loss(masked_outputs, x[mask_indices])
        mask_loss_ = F.cross_entropy(masked_outputs_, node_label[mask_indices])
        loss = mask_loss_
        
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
    
    # # BERT 인코더의 파라미터는 고정
    # model.encoder.eval()
    # for param in model.encoder.parameters():
    #     param.requires_grad = True
        
    for data in train_loader:
        recon_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs, node_label = data.x, data.edge_index, data.batch, data.num_graphs, data.node_label
        
        x_recon, adj_recon_list, train_cls_outputs, z_ = model(x, edge_index, batch, num_graphs)

        adj = adj_original(edge_index, batch, max_nodes)
        loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            adj_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2 / num_nodes
            adj_loss = adj_loss * adj_theta
            
            loss += adj_loss
            
            start_node = end_node
        
        print(f'train_adj_loss: {loss}')
        
        node_loss = (torch.norm(x_recon - x, p='fro')**2) / max_nodes
        node_loss = node_loss * node_theta
        print(f'train_node loss: {node_loss}')
        
        loss += node_loss
        num_sample += num_graphs

        loss.backward()
        recon_optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader), num_sample, train_cls_outputs.detach().cpu()


#%%
'''EVALUATION'''
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
            x, edge_index, batch, num_graphs, node_label = data.x, data.edge_index, data.batch, data.num_graphs, data.node_label

            adj = adj_original(edge_index, batch, max_nodes)

            x_recon, adj_recon_list, e_cls_output, z_ = model(x, edge_index, batch, num_graphs)

            recon_errors = []
            start_node = 0
            for i in range(num_graphs):
                num_nodes = (batch == i).sum().item()
                end_node = start_node + num_nodes

                node_loss = (torch.norm(x_recon[start_node:end_node] - x[start_node:end_node], p='fro')**2) / num_nodes
                
                adj_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2 / num_nodes
                
                cls_vec = e_cls_output[i].detach().cpu().numpy()  # [hidden_dim]
                distances = cdist([cls_vec], cluster_centers, metric='euclidean')  # [1, n_clusters]
                min_distance = distances.min()

                recon_error = node_loss.item() * alpha + adj_loss.item() * beta + min_distance.item() * gamma
                recon_errors.append(recon_error)
                
                print(f'test_node_loss: {node_loss.item() * alpha }')
                print(f'test_adj_loss: {adj_loss.item() * beta }')
                print(f'test_min_distance: {min_distance * gamma }')

                if data.y[i].item() == 0:
                    total_loss_ += recon_error
                else:
                    total_loss_anomaly_ += recon_error

                start_node = end_node
            
            total_loss = total_loss_ / sum(data.y == 0)
            total_loss_anomaly = total_loss_anomaly_ / sum(data.y == 1)
            
            total_loss_mean += total_loss
            total_loss_anomaly_mean += total_loss_anomaly
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)

    # Compute metrics
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

    return auroc, auprc, precision, recall, f1, total_loss_mean / len(test_loader), total_loss_anomaly_mean / len(test_loader)


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='DHFR')
parser.add_argument("--data-root", type=str, default='./dataset')
parser.add_argument("--assets-root", type=str, default="./assets")

parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--BERT-epochs", type=int, default=100)

parser.add_argument("--n-head", type=int, default=8)
parser.add_argument("--n-layer", type=int, default=8)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--n-cluster", type=int, default=3)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--random-seed", type=int, default=1)
parser.add_argument("--log-interval", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=300)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--dim-feed-layers", type=int, default=2048)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 768])

parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--dropout-rate", type=float, default=0.1)
parser.add_argument("--weight-decay", type=float, default=0.0001)
parser.add_argument("--learning-rate", type=float, default=0.0001)

parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--beta", type=float, default=0.0)
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

epochs: int = args.epochs
BERT_epochs: int = args.BERT_epochs

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
dim_feed_layers: int = args.dim_feed_layers
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


# %%
'''MODEL CONSTRUCTION'''
class BertEncoder(nn.Module):
    def __init__(self, num_features, hidden_dims, nhead, num_layers, max_nodes, num_node_classes, dropout_rate=0.1):
        super(BertEncoder, self).__init__()
        self.input_projection = nn.Linear(num_features, hidden_dims[-1])
        self.positional_encoding = GraphBertPositionalEncoding(hidden_dims[-1], max_nodes)
        
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dims[-1], nhead, dim_feed_layers, dropout_rate, activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.mask_token = nn.Parameter(torch.randn(1, hidden_dims[-1]))
        
        # self.predicter = nn.Linear(hidden_dims[-1], num_features)  # 원래 feature space로 projection
        self.classifier = nn.Linear(hidden_dims[-1], num_node_classes)  # 노드 라벨 수
        
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes

    def forward(self, x, edge_index, batch, mask_indices=None, training=True):
        h = self.input_projection(x)  # [num_nodes, hidden_dim]
        
        # 그래프별 처리
        batch_size = batch.max().item() + 1
        node_embeddings = []
        outputs = []
        outputs_ = []
        
        start_idx = 0
        for i in range(batch_size):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            graph_h = h[mask]  # [num_nodes_i, hidden_dim]
            
            # 포지셔널 인코딩 추가
            graph_edge_index = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < start_idx + num_nodes)]
            graph_edge_index = graph_edge_index - start_idx
            pos_encoding = self.positional_encoding(graph_edge_index, num_nodes)
            graph_h = graph_h + pos_encoding
            
            if training and mask_indices is not None:
                graph_mask_indices = mask_indices[mask]
                graph_h[graph_mask_indices] = self.mask_token
               
            # 패딩
            padded_h = F.pad(graph_h, (0, 0, 0, self.max_nodes - num_nodes), "constant", 0)
            padding_mask = torch.zeros(1, self.max_nodes, dtype=torch.bool, device=x.device)  # [1, max_nodes]
            padding_mask[0, num_nodes:] = True
            
            transformed_h = padded_h.unsqueeze(1)
            encoded = self.transformer(transformed_h, src_key_padding_mask=padding_mask)
            encoded = encoded.squeeze(1)[:num_nodes]  # 패딩 제거
           
            node_embeddings.append(encoded)
           
            # 학습 중이면 마스크된 노드의 원본 특성 예측
            if training and mask_indices is not None:
                # masked_output = self.predicter(encoded[graph_mask_indices])
                # outputs.append(masked_output)
                masked_output_ = self.classifier(encoded[graph_mask_indices])
                outputs_.append(masked_output_)
            
            start_idx += num_nodes
        
        # 모든 그래프의 노드 임베딩 합치기
        node_embeddings = torch.cat(node_embeddings, dim=0)
        
        if training and mask_indices is not None:
            # outputs = torch.cat(outputs, dim=0)
            outputs_ = torch.cat(outputs_, dim=0)
            return node_embeddings, outputs_
        
        return node_embeddings
    

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
    

class BilinearEdgeDecoder(nn.Module):
    def __init__(self, max_nodes):
        super(BilinearEdgeDecoder, self).__init__()
        self.max_nodes = max_nodes
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, z):
        actual_nodes = z.size(0)
        
        adj = torch.mm(z, z.t())
        adj = self.sigmoid(adj)
        adj = adj * (1 - torch.eye(actual_nodes, device=z.device))
        
        padded_adj = torch.zeros(self.max_nodes, self.max_nodes, device=z.device)
        padded_adj[:actual_nodes, :actual_nodes] = adj
        
        return padded_adj
        

#%%
class GraphBertPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_nodes):
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes
        
        # WSP와 LE 각각에 d_model/2 차원을 할당
        self.wsp_encoder = nn.Linear(max_nodes, d_model // 2)
        self.le_encoder = nn.Linear(max_nodes, d_model // 2)
        
    def get_wsp_encoding(self, edge_index, num_nodes):
        # Weighted Shortest Path 계산
        edge_index_np = edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = list(zip(edge_index_np[0], edge_index_np[1]))
        G.add_edges_from(edges)
        
        spl_matrix = torch.zeros((num_nodes, self.max_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    try:
                        path_length = nx.shortest_path_length(G, source=i, target=j)
                    except nx.NetworkXNoPath:
                        path_length = self.max_nodes  # 연결되지 않은 경우 최대 거리 할당
                    spl_matrix[i, j] = path_length

        return spl_matrix.to(edge_index.device)
    
    def get_laplacian_encoding(self, edge_index, num_nodes):
        # Laplacian Eigenvector 계산
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', 
                                            num_nodes=num_nodes)
        L = torch.sparse_coo_tensor(edge_index, edge_weight, 
                                (num_nodes, num_nodes)).to_dense()
        
        # CUDA 텐서를 CPU로 이동 후 NumPy로 변환
        L_np = L.cpu().numpy()
        eigenvals, eigenvecs = eigh(L_np)
        
        # 결과를 다시 텐서로 변환하고 원래 디바이스로 이동
        le_matrix = torch.from_numpy(eigenvecs).float().to(edge_index.device)
        
        padded_le = torch.zeros((num_nodes, self.max_nodes), device=edge_index.device)
        padded_le[:, :num_nodes] = le_matrix
        
        return padded_le
    
    def forward(self, edge_index, num_nodes):
        # WSP 인코딩
        wsp_matrix = self.get_wsp_encoding(edge_index, num_nodes)
        wsp_encoding = self.wsp_encoder(wsp_matrix)
        
        # LE 인코딩
        le_matrix = self.get_laplacian_encoding(edge_index, num_nodes)
        le_encoding = self.le_encoder(le_matrix)
        
        # WSP와 LE 결합
        pos_encoding = torch.cat([wsp_encoding, le_encoding], dim=-1)
        
        return pos_encoding
    

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_nodes, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.positional_encoding = GraphBertPositionalEncoding(d_model, max_nodes)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model

    def forward(self, src, edge_index_list, src_key_padding_mask):
        batch_size = src.size(0)
        max_seq_len = src.size(1)
        
        pos_encodings = []
        for i in range(batch_size):
            cls_pos_encoding = torch.zeros(1, self.d_model).to(src.device)
            
            num_nodes = (~src_key_padding_mask[i][1:]).sum().item()
            
            if num_nodes > 0:
                graph_pos_encoding = self.positional_encoding( 
                    edge_index_list[i], num_nodes
                )
                padded_pos_encoding = F.pad(
                    graph_pos_encoding, 
                    (0, 0, 0, max_seq_len - num_nodes - 1), 
                    'constant', 0
                )
            else:
                padded_pos_encoding = torch.zeros(max_seq_len - 1, self.d_model).to(src.device)
            
            full_pos_encoding = torch.cat([cls_pos_encoding, padded_pos_encoding], dim=0)
            pos_encodings.append(full_pos_encoding)
        
        pos_encoding_batch = torch.stack(pos_encodings)
        src_ = src + pos_encoding_batch
        
        src_ = src_.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        output_ = self.transformer_encoder(src_, src_key_padding_mask=src_key_padding_mask)
        output = output_.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]

        return output


def perform_clustering(train_cls_outputs, random_seed, n_clusters):
    cls_outputs_tensor = train_cls_outputs  # [total_num_graphs, hidden_dim]
    cls_outputs_np = cls_outputs_tensor.detach().cpu().numpy()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init="auto").fit(cls_outputs_np)

    cluster_centers = kmeans.cluster_centers_

    return kmeans, cluster_centers


#%%
# GRAPH_AUTOENCODER 클래스 수정
class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, nhead, num_layers, num_node_labels, dropout_rate=0.1):
        super(GRAPH_AUTOENCODER, self).__init__()
        # BERT 인코더로 변경
        self.encoder = BertEncoder(
            num_features=num_features,
            hidden_dims=hidden_dims,
            nhead=n_head,
            num_layers=n_layer,
            max_nodes=max_nodes,
            num_node_classes=num_node_labels,
            dropout_rate=dropout_rate
        )
        self.transformer = TransformerEncoder(
            d_model=hidden_dims[-1],
            nhead=8,
            num_layers=2,
            dim_feedforward=dim_feed_layers,
            max_nodes=max_nodes,
            dropout=dropout_rate
        )
        self.u_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        self.feature_decoder = FeatureDecoder(hidden_dims[-1], num_features)
        self.edge_decoder = BilinearEdgeDecoder(max_nodes)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dims[-1]))
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        self.sigmoid = nn.Sigmoid()
        
        # 가중치 초기화
        self.apply(self._init_weights)


    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, training=True):
        # BERT 인코딩
        if training and mask_indices is not None:
            z, masked_outputs_ = self.encoder(
                x, edge_index, batch, mask_indices, training=True
            )
        else:
            z = self.encoder(
                x, edge_index, batch, training=False
            )
        print(mask_indices)
        print(training)
        z_list = [z[batch == i] for i in range(num_graphs)] # 그래프 별 z 저장 (batch_size, num nodes, feature dim)
        edge_index_list = [] # 그래프 별 엣지 인덱스 저장 (batch_size), edge_index_list[0] = (2 x m), m is # of edges
        start_idx = 0
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            mask = (batch == i)
            graph_edges = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < start_idx + num_nodes)]
            graph_edges = graph_edges - start_idx
            edge_index_list.append(graph_edges)
            start_idx += num_nodes
        
        z_with_cls_list = []
        mask_list = []
        max_nodes_in_batch = max(z_graph.size(0) for z_graph in z_list) # 배치 내 최대 노드 수
        
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            cls_token = self.cls_token.repeat(1, 1, 1)  # [1, 1, hidden_dim]
            cls_token = cls_token.to(device)
            z_graph = z_list[i].unsqueeze(1)  # [num_nodes, 1, hidden_dim]
            
            pad_size = max_nodes_in_batch - num_nodes
            z_graph_padded = F.pad(z_graph, (0, 0, 0, 0, 0, pad_size), 'constant', 0)  # [max_nodes, 1, hidden_dim] -> 나머지는 패딩
            
            z_with_cls = torch.cat([cls_token, z_graph_padded.transpose(0, 1)], dim=1)  # [1, max_nodes+1, hidden_dim] -> CLS 추가
            z_with_cls_list.append(z_with_cls)

            graph_mask = torch.cat([torch.tensor([False]), torch.tensor([False]*num_nodes + [True]*pad_size)])
            mask_list.append(graph_mask)

        z_with_cls_batch = torch.cat(z_with_cls_list, dim=0)  # [batch_size, max_nodes+1, hidden_dim] -> 모든 그래프에 대한 CLS 추가
        mask = torch.stack(mask_list).to(z.device)  # [batch_size, max_nodes+1]

        encoded = self.transformer(z_with_cls_batch, edge_index_list, mask)

        cls_output = encoded[:, 0, :]       # [batch_size, hidden_dim]
        node_output = encoded[:, 1:, :]     # [batch_size, max_nodes, hidden_dim]
        
        node_output_list = []
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            node_output_list.append(node_output[i, :num_nodes, :])

        u = torch.cat(node_output_list, dim=0)  # [total_num_nodes, hidden_dim]

        u_prime = self.u_mlp(u)
        
        x_recon = self.feature_decoder(u_prime)
                
        adj_recon_list = []
        idx = 0
        for i in range(num_graphs):
            num_nodes = z_list[i].size(0)
            z_graph = u_prime[idx:idx + num_nodes]
            adj_recon = self.edge_decoder(z_graph)
            adj_recon_list.append(adj_recon)
            idx += num_nodes
        
        if training and mask_indices is not None:
            return x_recon, adj_recon_list, cls_output, z, masked_outputs_
        
        return x_recon, adj_recon_list, cls_output, z

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


# %%
'''RUN'''
def run(dataset_name, random_seed, dataset_AN, trial, device=device):
    split=splits[trial]
    
    all_results = []
    set_seed(random_seed)

    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN)
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']
    max_node_label = meta['max_node_label']
    
    # BERT 모델 저장 경로
    bert_save_path = f'/home1/rldnjs16/graph_anomaly_detection/BERT_model/pretrained_bert_{dataset_name}_fold{trial}_seed{random_seed}_BERT_epochs{BERT_epochs}_try0.pth'
    
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
    
    # 훈련 단계에서 cls_outputs 저장할 리스트 초기화
    global train_cls_outputs
    train_cls_outputs = []
    
    # 1단계: BERT 임베딩 학습
    if os.path.exists(bert_save_path):
        print("Loading pretrained BERT...")
        # BERT 인코더의 가중치만 로드
        model.encoder.load_state_dict(torch.load(bert_save_path))
    else:
        print("Training BERT from scratch...")
        # 1단계: BERT 임베딩 학습
        print("Stage 1: Training BERT embeddings...")

        bert_optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        bert_scheduler = ReduceLROnPlateau(bert_optimizer, mode='min', factor=factor, patience=patience)
    
        for epoch in range(1, BERT_epochs+1):
            train_loss, num_sample, node_embeddings = train_bert_embedding(
                model, train_loader, bert_optimizer, device
            )
            bert_scheduler.step(train_loss)
            
            if epoch % log_interval == 0:
                print(f'BERT Training Epoch {epoch}: Loss = {train_loss:.4f}')
                
        # 학습된 BERT 저장
        print("Saving pretrained BERT...")
        torch.save(model.encoder.state_dict(), bert_save_path)
        
    # 2단계: 재구성 학습
    print("\nStage 2: Training reconstruction...")
    recon_optimizer = torch.optim.Adam(model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    for epoch in range(1, epochs+1):
        fold_start = time.time()  # 현재 폴드 시작 시간
        train_loss, num_sample, train_cls_outputs = train(model, train_loader, recon_optimizer, max_nodes, device)
        
        info_train = 'Epoch {:3d}, Loss {:.4f}'.format(epoch, train_loss)

        if epoch % log_interval == 0:
            
            cluster_centers = train_cls_outputs.mean(dim=0)
            cluster_centers = cluster_centers.detach().cpu().numpy()
            cluster_centers = cluster_centers.reshape(-1, hidden_dims[-1])

            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly = evaluate_model(model, test_loader, max_nodes, cluster_centers, device)
            # scheduler.step(auroc)
            
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
