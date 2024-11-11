#%%
'''IMPORTS'''
import os
import re
import sys
import wandb
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj, to_undirected, to_networkx, k_hop_subgraph
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from scipy.stats import wasserstein_distance
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from functools import partial
from multiprocessing import Pool

from module.loss import info_nce_loss, Triplet_loss, loss_cal
from module.utils import set_seed, set_device, EarlyStopping, get_ad_split_TU, get_data_loaders_TU, adj_original, batch_nodes_subgraphs

import networkx as nx
import dgl
        
#%%
# def adj_to_dgl_graph(adj):
#     """Convert adjacency matrix to dgl format."""
#     nx_graph = nx.from_scipy_sparse_array(adj)
#     dgl_graph = dgl.DGLGraph(nx_graph)
#     return dgl_graph

def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    # torch.Tensor를 scipy 희소 행렬로 변환
    adj_sparse = sp.csr_matrix(adj.cpu().numpy())
    
    # scipy 희소 행렬을 networkx 그래프로 변환
    nx_graph = nx.from_scipy_sparse_array(adj_sparse)
    
    # networkx 그래프를 DGL 그래프로 변환
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph


def random_walk_with_restart(graph, start_node, num_steps, restart_prob):
    current_node = start_node
    walk = [current_node]
    for _ in range(num_steps - 1):
        if random.random() < restart_prob:
            current_node = start_node
        else:
            neighbors = graph.successors(current_node).numpy().tolist()
            if neighbors:
                current_node = random.choice(neighbors)
            else:
                current_node = start_node
        walk.append(current_node)
    return walk


def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    subv = []
    for i in all_idx:
        walk = random_walk_with_restart(dgl_graph, i, subgraph_size * 3, restart_prob=0.1)
        subv.append(list(set(walk)))
        retry_time = 0
        while len(subv[-1]) < reduced_size:
            walk = random_walk_with_restart(dgl_graph, i, subgraph_size * 5, restart_prob=0.1)
            subv[-1] = list(set(walk))
            retry_time += 1
            if (len(subv[-1]) <= 2) and (retry_time > 10):
                subv[-1] = (subv[-1] * reduced_size)
        subv[-1] = subv[-1][:reduced_size]
        subv[-1].append(i)
    return subv

def batch_nodes_subgraphs(data, subgraph_size=20):
    device = data.x.device
    batched_pos_subgraphs = []
    batched_neg_subgraphs = []
    batched_target_node_features = []
    start_node = 0
    for i in range(data.num_graphs):
        num_nodes = (data.batch == i).sum().item()
        end_node = start_node + num_nodes
        
        # 현재 그래프에 대한 부분 그래프 추출
        sub_x = data.x[start_node:end_node]
        sub_edge_index = data.edge_index[:, (data.edge_index[0] >= start_node) & (data.edge_index[1] < end_node)]
        sub_edge_index = sub_edge_index - start_node  # 노드 인덱스를 0부터 시작하도록 조정
        
        # adjacency matrix 생성
        adj = to_dense_adj(sub_edge_index, max_num_nodes=num_nodes)[0].to(device)
        
        # DGL 그래프로 변환
        dgl_graph = adj_to_dgl_graph(adj)
        
        # RWR을 사용하여 서브그래프 생성
        subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
        
        # 양성 및 음성 서브그래프 생성
        target_node = torch.randint(0, num_nodes, size=(1,), device=device).item()
        pos_subgraph_nodes = subgraphs[target_node]
        neg_subgraph_nodes = subgraphs[torch.randint(0, num_nodes, size=(1,), device=device).item()]
        
        # 서브그래프에 대한 Data 객체 생성
        pos_subgraph = Data(x=sub_x[pos_subgraph_nodes],
                            edge_index=subgraph_edge_index(sub_edge_index, pos_subgraph_nodes, device))
        neg_subgraph = Data(x=sub_x[neg_subgraph_nodes],
                            edge_index=subgraph_edge_index(sub_edge_index, neg_subgraph_nodes, device))
        
        batched_pos_subgraphs.append(pos_subgraph)
        batched_neg_subgraphs.append(neg_subgraph)
        batched_target_node_features.append(sub_x[target_node])
        
        start_node = end_node
    
    batched_pos_subgraphs = Batch.from_data_list(batched_pos_subgraphs)
    batched_neg_subgraphs = Batch.from_data_list(batched_neg_subgraphs)
    batched_target_node_features = torch.stack(batched_target_node_features)
    
    return batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features

def subgraph_edge_index(edge_index, subgraph_nodes, device):
    subgraph_nodes_tensor = torch.tensor(subgraph_nodes, device=device)
    mask = torch.isin(edge_index[0], subgraph_nodes_tensor) & torch.isin(edge_index[1], subgraph_nodes_tensor)
    subgraph_edge_index = edge_index[:, mask]
    node_idx = {old: new for new, old in enumerate(subgraph_nodes)}
    new_edge_index = torch.tensor([[node_idx[i.item()] for i in subgraph_edge_index[0]],
                                   [node_idx[i.item()] for i in subgraph_edge_index[1]]], device=device)
    return new_edge_index


#%%
'''TARIN'''
def train(model, train_loader, optimizer, max_nodes, device):
    model.train()
    total_loss = 0
    num_sample = 0
    for data in train_loader:
        optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features = batch_nodes_subgraphs(data)
        
        adj = adj_original(edge_index, batch, max_nodes)
        
        z, z_g_mlp, z_noisy, x_recon, adj_recon_list, z_tilde, z_tilde_g_mlp, pos_sub_z_g, neg_sub_z_g, target_z = model(x, edge_index, batch, num_graphs, batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features)
        
        z_noisy_g = global_max_pool(z_noisy, batch)
        z_noisy_g_mlp = model.projection_head(z_noisy_g)
        
        # Contrastive loss
        contrastive_loss = loss_cal(z_g_mlp, z_tilde_g_mlp)
        
        # Triplet loss
        triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g))
        
        # Info NCE loss
        info_nce = info_nce_loss(z_g_mlp, z_noisy_g_mlp)
        
        loss = 0
        start_node = 0
        for i in range(data.num_graphs): 
            num_nodes = (data.batch == i).sum().item() 
            end_node = start_node + num_nodes
            
            adj_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i])
            node_loss = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2 / num_nodes
            
            z_dist = torch.pdist(z[start_node:end_node])
            z_tilde_dist = torch.pdist(z_tilde[start_node:end_node])
            w_distance = torch.tensor(wasserstein_distance(z_dist.detach().cpu().numpy(), z_tilde_dist.detach().cpu().numpy()), device=device) / 2
            
            loss += adj_loss + node_loss + w_distance
            
            start_node = end_node
        
        # Add new loss components
        loss += contrastive_loss + triplet_loss + info_nce
        
        num_sample += data.num_graphs
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader), num_sample

def evaluate_model(model, val_loader, max_nodes, device):
    model.eval()
    total_loss = 0
    total_loss_anomaly = 0
    
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)  
            batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features = batch_nodes_subgraphs(data)
            x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        
            adj = adj_original(edge_index, batch, max_nodes)
            
            z, z_g_mlp, z_noisy, x_recon, adj_recon_list, z_tilde, z_tilde_g_mlp, pos_sub_z_g, neg_sub_z_g, target_z = model(x, edge_index, batch, num_graphs, batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features)
            
            recon_errors = []
            start_node = 0
            for i in range(data.num_graphs): 
                recon_error = 0
                num_nodes = (data.batch == i).sum().item() 
                end_node = start_node + num_nodes
                
                node_loss = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2 / num_nodes
                node_recon_error = node_loss * 2
                
                adj_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i])
                edge_recon_error = adj_loss / 50
                
                z_dist = torch.pdist(z[start_node:end_node])
                z_tilde_dist = torch.pdist(z_tilde[start_node:end_node])
                w_distance = torch.tensor(wasserstein_distance(z_dist.detach().cpu().numpy(), z_tilde_dist.detach().cpu().numpy()), device=device) 
                
                recon_error += node_recon_error + edge_recon_error + w_distance
                recon_errors.append(recon_error.item())

                if data.y[i].item() == 0:
                    total_loss += adj_loss / 400 + node_loss + w_distance / 2
                else:
                    total_loss_anomaly += adj_loss / 400 + node_loss + w_distance / 2
                
                start_node = end_node
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
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
    
    return auroc, auprc, precision, recall, f1, total_loss / len(val_loader), total_loss_anomaly / len(val_loader)


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='COX2')
parser.add_argument("--assets-root", type=str, default="./assets")
parser.add_argument("--data-root", type=str, default='./dataset')

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=300)
parser.add_argument("--random-seed", type=int, default=42)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])

parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--learning-rate", type=float, default=0.0001)
parser.add_argument("--dropout-rate", type=float, default=0.1)

parser.add_argument("--dataset-AN", action="store_false")

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
log_interval: int = args.log_interval
step_size: int = args.step_size
batch_size: int = args.batch_size
n_cross_val: int = args.n_cross_val
random_seed: int = args.random_seed
hidden_dims: list = args.hidden_dims
n_test_anomaly: int = args.n_test_anomaly
test_batch_size: int = args.test_batch_size

factor: float = args.factor
test_size: float = args.test_size
weight_decay: float = args.weight_decay
learning_rate: float = args.learning_rate
dropout_rate: float = args.dropout_rate

dataset_AN: bool = args.dataset_AN

set_seed(random_seed)

device = set_device()
# device = torch.device("cpu")
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
      
# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.conv.lin.weight, gain=gain)
        nn.init.zeros_(self.conv.bias)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.xavier_uniform_(self.shortcut.weight, gain=1.0)
            nn.init.zeros_(self.shortcut.bias)

    def forward(self, x, edge_index):
        residual = self.shortcut(x)
        x = F.relu(self.bn(self.conv(x, edge_index)))
        x = self.dropout(x)
        return F.relu(x + residual)

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
    def __init__(self, embed_dim, hidden_dims, num_features):
        super(FeatureDecoder, self).__init__()
        dims = [embed_dim] + list(reversed(hidden_dims)) + [num_features]
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                self.layers.append(nn.ReLU())
                self.layers.append(nn.BatchNorm1d(dims[i+1]))

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
        return z

class BilinearEdgeDecoder(nn.Module):
    def __init__(self, input_dim, max_nodes, threshold=0.5):
        super(BilinearEdgeDecoder, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.threshold = threshold
        self.max_nodes = max_nodes
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight, gain=0.01)

    def forward(self, z):
        actual_nodes = z.size(0)
        adj = torch.sigmoid(torch.mm(torch.mm(z, self.weight), z.t()))
        adj_binary = (adj > self.threshold).float()
        padded_adj = F.pad(adj_binary, (0, self.max_nodes - actual_nodes, 0, self.max_nodes - actual_nodes))
        return padded_adj
    

# %%
class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, dropout_rate=0.1, noise_std=0.1):
        super(GRAPH_AUTOENCODER, self).__init__()
        self.encoder = Encoder(num_features, hidden_dims, dropout_rate)
        self.encoder_sub = Encoder(num_features, hidden_dims, dropout_rate)
        self.encoder_node_blocks = nn.ModuleList()        
        
        self.edge_decoder = BilinearEdgeDecoder(hidden_dims[-1], max_nodes, threshold=0.5)
        self.feature_decoder = FeatureDecoder(hidden_dims[-1], hidden_dims, num_features)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        self.noise_std = noise_std
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_node_blocks.append(nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            current_dim = hidden_dim  

        # 가중치 초기화
        self.apply(self._init_weights)
        
    def apply_noise_to_encoder(self):
        for module in self.encoder.modules():
            if isinstance(module, GCNConv):
                module.lin.weight.data += torch.randn_like(module.lin.weight) * self.noise_std
                if module.lin.bias is not None:
                    module.lin.bias.data += torch.randn_like(module.lin.bias) * self.noise_std

    def forward(self, x, edge_index, batch, num_graphs, batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features):
        z = self.encoder(x, edge_index)
        z = self.dropout(z)
        
        adj_recon_list = []
        for i in range(num_graphs):
            mask = (batch == i)
            z_graph = z[mask]
            adj_recon = self.edge_decoder(z_graph)
            adj_recon_list.append(adj_recon)
        
        new_edge_index = self.get_edge_index_from_adj_list(adj_recon_list, batch).to(x.device)
        
        # Apply noise to encoder
        self.apply_noise_to_encoder()
        
        # Noisy encoding
        z_noisy = self.encoder(x, edge_index)
        z_noisy = self.dropout(z_noisy)
        
        x_recon = self.feature_decoder(z)

        z_g = global_max_pool(z, batch)
        z_g_mlp = self.projection_head(z_g)
        
        z_tilde = self.encoder(x_recon, new_edge_index)
        z_tilde_g = global_max_pool(z_tilde, batch)
        z_tilde_g_mlp = self.projection_head(z_tilde_g)

        # subgraph        
        pos_x, pos_edge_index, pos_batch = batched_pos_subgraphs.x, batched_pos_subgraphs.edge_index, batched_pos_subgraphs.batch
        pos_sub_z, pos_new_edge_index = self.process_subgraphs(batched_pos_subgraphs)
        pos_sub_z = torch.cat(pos_sub_z) # (number of nodes, embedded size)
        
        unique_pos_batch, new_pos_batch = torch.unique(pos_batch, return_inverse=True)
        pos_sub_z_g = global_mean_pool(pos_sub_z, new_pos_batch)
        
        neg_x, neg_edge_index, neg_batch = batched_neg_subgraphs.x, batched_neg_subgraphs.edge_index, batched_neg_subgraphs.batch
        neg_sub_z, neg_new_edge_index = self.process_subgraphs(batched_neg_subgraphs)
        neg_sub_z = torch.cat(neg_sub_z)
        
        unique_neg_batch, new_neg_batch = torch.unique(neg_batch, return_inverse=True)
        neg_sub_z_g = global_mean_pool(neg_sub_z, new_neg_batch)
        
        target_z = self.encode_node(batched_target_node_features) # (batch_size, feature_size)
        
        return z, z_g_mlp, z_noisy, x_recon, adj_recon_list, z_tilde, z_tilde_g_mlp, pos_sub_z_g, neg_sub_z_g, target_z

    def get_edge_index_from_adj_list(self, adj_recon_list, batch, threshold=0.5):
        edge_index_list = []
        start_idx = 0
        for i, adj in enumerate(adj_recon_list):
            num_nodes = (batch == i).sum().item()
            adj_binary = (adj[:num_nodes, :num_nodes] > threshold).float()
            edge_index = adj_binary.nonzero().t() + start_idx
            edge_index_list.append(edge_index)
            start_idx += num_nodes
        return torch.cat(edge_index_list, dim=1)

    def encode(self, x, edge_index):
        return self.encoder(x, edge_index)

    def encode_subgraph(self, x, edge_index):
        return self.encoder_sub(x, edge_index)

    def encode_node(self, x):
        for encoder in self.encoder_node_blocks[:-1]:
            x = self.act(encoder(x))
            bn_module = nn.BatchNorm1d(x.size()[1]).to(x.device)
            x = bn_module(x)
        x = self.encoder_node_blocks[-1](x)
        x = F.normalize(x, p=2, dim=1)
        return x
    
    def process_subgraphs(self, subgraphs):
        # 각 서브그래프에 대해 인코딩을 실행
        subgraph_embeddings = []
        for i in range(subgraphs.num_graphs):
            subgraph = subgraphs[i]
            x = subgraph.x
            edge_index = subgraph.edge_index

            # 로컬 인덱스로 edge_index 재조정
            unique_nodes, new_edge_index = torch.unique(edge_index, return_inverse=True)
            new_edge_index = new_edge_index.reshape(edge_index.shape)

            # 서브그래프 인코딩
            z = self.encode_subgraph(x, new_edge_index)
            subgraph_embeddings.append(z)

        return subgraph_embeddings, new_edge_index
    
    def conservative_augment_molecular_graph(self, graph, node_attr_noise_std=0.01, edge_mask_prob=0.03):
        augmented_graph = graph.clone()
        
        if graph.x is not None:
            augmented_graph.x = graph.x + torch.randn_like(graph.x) * node_attr_noise_std
        
        if random.random() < edge_mask_prob:
            edge_index = augmented_graph.edge_index
            num_edges = edge_index.size(1)
            mask = torch.rand(num_edges) > 0.1
            augmented_graph.masked_edges = edge_index[:, ~mask]
            augmented_graph.edge_index = edge_index[:, mask]
        
        return augmented_graph

    def _init_weights(self, module):
        if isinstance(module, ResidualBlock):
            module.reset_parameters()
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, BilinearEdgeDecoder):
            nn.init.xavier_uniform_(module.weight, gain=0.01)
            

# %%
'''DATASETS'''
splits = get_ad_split_TU(dataset_name, n_cross_val, random_seed)
loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, splits[0])
num_train = meta['num_train']
num_features = meta['num_feat']
num_edge_features = meta['num_edge_feat']
max_nodes = meta['max_nodes']

print(f'Number of graphs: {num_train}')
print(f'Number of features: {num_features}')
print(f'Number of edge features: {num_edge_features}')
print(f'Max nodes: {max_nodes}')


#%%
'''MODEL AND OPTIMIZER DEFINE'''
model = GRAPH_AUTOENCODER(num_features, hidden_dims, max_nodes, dropout_rate=dropout_rate).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2 regularization
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)


#%%
def run(dataset_name, random_seed, split=None, device=device):
    set_seed(random_seed)

    loaders, meta = get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split)    
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']

    model = GRAPH_AUTOENCODER(num_features, hidden_dims, max_nodes, dropout_rate=dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_loader = loaders['train']
    test_loader = loaders['test']

    for epoch in range(1, epochs+1):
        train_loss, num_sample = train(model, train_loader, optimizer, max_nodes, device)
        
        info_train = 'Epoch {:3d}, Loss CL {:.4f}'.format(epoch, train_loss / num_sample)

        if epoch % log_interval == 0:
            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly = evaluate_model(model, test_loader, max_nodes, device)

            info_test = 'AD_AUC:{:.4f}'.format(auroc)

            print(info_train + '   ' + info_test)
            
            early_stopping(test_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    return auroc


#%%
if __name__ == '__main__':
    ad_aucs = []
    splits = get_ad_split_TU(dataset_name, n_cross_val, random_seed)
    key_auc_list = []

    for trial in range(n_cross_val):
        results = run(dataset_name, random_seed, split=splits[trial])
        ad_auc = results
        ad_aucs.append(ad_auc)

    results = 'AUC: {:.2f}+-{:.2f}'.format(np.mean(ad_aucs) * 100, np.std(ad_aucs) * 100)
    print(len(ad_aucs))

    print('[FINAL RESULTS] ' + results)
    
    
# %%
