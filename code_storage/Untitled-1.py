#%%
'''IMPORTS'''
import os
import wandb
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.nn import init
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

from module.model import ResidualBlock, GRAPH_AUTOENCODER_
from module.loss import Triplet_loss, loss_cal, focal_loss
from module.utils import set_device
    

#%%
'''TARIN'''
def train(model, train_loader, optimizer, threshold=0.5):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        adj, z, z_g, batch, x_recon, adj_recon_list, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z = model(data)
        
        loss = 0
        start_node = 0
        
        for i in range(data.num_graphs): 
            num_nodes = (data.batch == i).sum().item() 
            end_node = start_node + num_nodes
            graph_num_nodes = end_node - start_node        
            
            # focal_loss_value = focal_loss(adj_recon_list[i], adj[i], gamma=2, alpha=0.25)
            # l1_loss = focal_loss_value

            edge_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2
            graph_edge_loss = edge_loss/graph_num_nodes
            l1_loss = graph_edge_loss / 30
            
            edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
            edge_index = edges.t()
            
            z_tilde =  model.encode(x_recon, edge_index).to('cuda')
            z_tilde_g = global_max_pool(z_tilde, batch)
            
            recon_z_node_loss = torch.norm(z[start_node:end_node] - z_tilde[start_node:end_node], p='fro')**2
            graph_z_node_loss = recon_z_node_loss/graph_num_nodes
            
            recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
            l3_loss = (graph_z_node_loss / 10) + (recon_z_graph_loss / 10)
                        
            loss += l1_loss + l3_loss
            
            start_node = end_node
        
        node_loss = torch.norm(x_recon - data.x, p='fro')**2
        node_loss = (node_loss/x_recon.size(0))
        
        triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) / 10
        # l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) * 3
        loss += node_loss + triplet_loss
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)


#%%
'''EVALUATION'''
def evaluate_model(model, val_loader, threshold = 0.5):
    model.eval()
    total_loss = 0
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)  
            adj, z, z_g, batch, x_recon, adj_recon_list, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z = model(data)
            
            recon_errors = []
            loss = 0
            start_node = 0
            
            for i in range(data.num_graphs): 
                recon_error = 0
                num_nodes = (data.batch == i).sum().item() 
                end_node = start_node + num_nodes
                graph_num_nodes = end_node - start_node        
                
                node_loss = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2
                graph_node_loss = node_loss/graph_num_nodes
                node_recon_error = graph_node_loss / 30

                edge_loss = torch.norm(adj_recon_list[i] - adj[i], p='fro')**2
                graph_edge_loss = edge_loss/graph_num_nodes
                edge_recon_error = graph_edge_loss / 30
                
                edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
                edge_index = edges.t()
                
                z_tilde =  model.encode(x_recon, edge_index).to('cuda')
                z_tilde_g = global_max_pool(z_tilde, batch)
                
                recon_z_node_loss = torch.norm(z[start_node:end_node] - z_tilde[start_node:end_node], p='fro')**2
                graph_z_node_loss = recon_z_node_loss/graph_num_nodes

                recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
                graph_recon_loss = (graph_z_node_loss / 10) + (recon_z_graph_loss / 10)
            
                recon_error += node_recon_error + edge_recon_error + graph_recon_loss
                recon_errors.append(recon_error.item())
            
                loss += node_recon_error + edge_recon_error + graph_recon_loss

                start_node = end_node
            
            triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) / 10
            l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) * 3
            loss += triplet_loss + l2_loss
            total_loss += loss.item()
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    auroc = auc(fpr, tpr)
        
    return auroc, total_loss / len(train_loader)


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--data-root", type=str, default='./dataset/data')
parser.add_argument("--assets-root", type=str, default="./assets")
parser.add_argument("--dataset-name", type=str, default='NCI1')

parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])
parser.add_argument("--test-batch-size", type=int, default=32)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--random-seed", type=int, default=42)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--epochs", type=int, default=100)


parser.add_argument("--learning-rate", type=float, default=0.0001)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--factor", type=float, default=0.1)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--patience", type=int, default=10)

parser.add_argument("--dataset-AN", action="store_false")

try:
    args = parser.parse_args()
except:
    args = parser.parse_args([])


#%%
'''OPTIONS'''
dataset_name: str = args.dataset_name
assets_root: str = args.assets_root
data_root: str = args.data_root

test_batch_size: int = args.test_batch_size
n_test_anomaly: int = args.n_test_anomaly
hidden_dims: list = args.hidden_dims
random_seed: int = args.random_seed
n_cross_val: int = args.n_cross_val
batch_size: int = args.batch_sizes
step_size: int = args.step_size
patience: int = args.patience
epochs: int = args.epochs

learning_rate: float = args.learning_rate
weight_decay: float = args.weight_decay
test_size: float = args.test_size
factor: float = args.factor

dataset_AN: bool = args.dataset_AN

# device = torch.device('cpu')
device = set_device()
print(f"Using device: {device}")

torch.set_printoptions(edgeitems=3)  # 텐서 출력시 표시되는 요소 수 조정
torch.backends.cuda.matmul.allow_tf32 = False  # 더 정확한 연산을 위해 False 설정

# CUDA 디버깅 활성화
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

wandb.init(project="graph anomaly detection", entity="ki5n2")

wandb.config.update(args)

wandb.config = {
  "random_seed": random_seed,
  "learning_rate": 0.0001,
  "epochs": 100
}

    
#%%
'''DATASETS'''
graph_dataset = TUDataset(root='./dataset/data', name='DHFR').shuffle()
labels = np.array([data.y.item() for data in graph_dataset])

print(f'Number of graphs: {len(graph_dataset)}')
print(f'Number of features: {graph_dataset.num_features}')
print(f'Number of edge features: {graph_dataset.num_edge_features}')
print(f'labels: {labels}')

# 5-fold cross-validation 설정
skf = StratifiedKFold(n_splits=n_cross_val, shuffle=True, random_state=random_seed)


#%%
# graph_dataset = TUDataset(root='./dataset', name='NCI1')
# graph_dataset = graph_dataset.shuffle()

# print(f'Number of graphs: {len(graph_dataset)}')
# print(f'Number of features: {graph_dataset.num_features}')
# print(f'Number of edge features: {graph_dataset.num_edge_features}')

# dataset_normal = [data for data in graph_dataset if data.y.item() == 1]
# dataset_anomaly = [data for data in graph_dataset if data.y.item() == 0]
# train_data, test_data = train_test_split(dataset_normal, test_size=test_size, random_state=random_seed)
# evaluation_data = test_data + dataset_anomaly[:n_test_anomaly]

# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(evaluation_data, batch_size=test_batch_size, shuffle=True)

# print(f"Number of normal samples: {len(dataset_normal)}")
# print(f"Number of anomaly samples: {len(dataset_anomaly)}")
# print(f"Number of test normal data: {len(test_data)}")
# print(f"Number of test anomaly samples: {len(dataset_anomaly[:n_test_anomaly])}")
# print(f"Number of samples in the evaluation dataset: {len(evaluation_data)}")


#%%
'''MODEL AND OPTIMIZER DEFINE'''
num_features = graph_dataset.num_features
model = GRAPH_AUTOENCODER_(num_features, hidden_dims).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2 regularization
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)


#%%
'''TRAIN PROCESS'''
torch.autograd.set_detect_anomaly(True)  

for fold, (train_idx, val_idx) in enumerate(skf.split(graph_dataset, labels)):
    print(f"Fold {fold + 1}")
    
    if dataset_AN:
        train_normal_idx = [idx for idx in train_idx if labels[idx] == 1]
        train_dataset = [graph_dataset[i] for i in train_normal_idx]
        val_dataset = [graph_dataset[i] for i in val_idx]

        idx = 0
        for data in train_dataset:
            data.y = 0
            data['idx'] = idx
            idx += 1

        for data in val_dataset:
            data.y = 1 if data.y == 0 else 0
            
    else:
        train_normal_idx = [idx for idx in train_idx if labels[idx] == 0]
        train_dataset = [graph_dataset[i] for i in train_normal_idx]
        val_dataset = [graph_dataset[i] for i in val_idx]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=True)
    
    print(f"  Training set size (normal only): {len(train_dataset)}")
    print(f"  Validation set size (normal + abnormal): {len(val_dataset)}")
    
    model = GRAPH_AUTOENCODER_(num_features, hidden_dims).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2 regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer)
        auroc, test_loss = evaluate_model(model, val_loader)
        scheduler.step(auroc)  # AUC 기반으로 학습률 조정
        print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation loss = {test_loss:.4f}, Validation AUC = {auroc:.4f}')
        wandb.log({"epoch": epoch, "train loss": train_loss, "test loss": test_loss, "test AUC": auroc})

    print("\n")


wandb.finish()
