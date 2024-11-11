#%%
'''IMPORTS'''
import os
import re
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

from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from scipy.stats import wasserstein_distance
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold

from module.model import ResidualBlock, GRAPH_AUTOENCODER2
from module.loss import Triplet_loss, loss_cal, info_nce_loss
from module.utils import set_seed, set_device, EarlyStopping
        
        
#%%
'''TARIN'''
def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        adj, z, z_g, batch, x_recon, adj_recon_list, z_tilde, z_tilde_g, aug_z_g = model(data)
        
        z_g_mean = global_mean_pool(z, batch)
        z_g_train = z_g.mean(dim=0, keepdim=True)
        z_g_mean_train = z_g_mean.mean(dim=0, keepdim=True)
        
        print(len(adj))
        print(len(adj_recon_list))
        print(adj[0].shape)
        print(adj_recon_list[0].shape)
        print(adj[0])
        print(adj_recon_list[0])

        loss = 0
        loss1 = 0
        loss2 = 0
        start_node = 0
        
        for i in range(data.num_graphs): 
            num_nodes = (data.batch == i).sum().item() 
            end_node = start_node + num_nodes
            # graph_num_nodes = end_node - start_node        
            
            # node_loss_2 = criterion_node(x_recon_prime[start_node:end_node], data.x[start_node:end_node])
            # node_loss = node_loss_1 / 200

            adj_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i])
            l1_loss = adj_loss / 400
            loss1 += l1_loss 
            
            z_dist = torch.pdist(z[start_node:end_node])
            z_tilde_dist = torch.pdist(z_tilde[start_node:end_node])
            w_distance = torch.tensor(wasserstein_distance(z_dist.detach().cpu().numpy(), z_tilde_dist.detach().cpu().numpy()), device='cpu') /2
            loss2 += w_distance
            
            loss += l1_loss + w_distance
            # edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
            # edge_index = edges.t()
    
            # recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
            
            start_node = end_node

        print(f'Train adj_loss : {loss1}')
        print(f'Train w_distance : {loss2}')
        
        node_loss = torch.norm(x_recon - data.x, p='fro')**2
        node_loss = (node_loss/x_recon.size(0))
  
        # recon_z_node_loss = torch.norm(z - z_tilde, p='fro')**2
        # graph_z_node_loss = recon_z_node_loss / (z.size(1) * 2)
        
        # z_g_dist = torch.pdist(z_g)
        # z_tilde_g_dist = torch.pdist(z_tilde_g)
        # w_distance = torch.tensor(wasserstein_distance(z_g_dist.detach().cpu().numpy(), z_tilde_g_dist.detach().cpu().numpy()), device='cuda')
        # w_distance = w_distance * 50
        
        info_nce_loss_value = info_nce_loss(z_g, aug_z_g) * 4

        print(f'Train node loss: {node_loss}')
        print(f'Train info_nce_loss :{info_nce_loss_value}')
        
        # triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) / 10
        # l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) * 3
        loss += node_loss + w_distance + info_nce_loss_value
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader), z_g_train, z_g_mean_train


#%%
'''EVALUATION'''
def evaluate_model(model, val_loader, z_g_train, z_g_mean_train):
    model.eval()
    total_loss = 0
    total_loss_anomaly = 0
    max_AUC=0
    
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)  
            adj, z, z_g, batch, x_recon, adj_recon_list, z_tilde, z_tilde_g, aug_z_g = model(data)
            
            z_g_test = z_g
            z_g_mean_test = global_mean_pool(z, batch)
            
            recon_errors = []
            recon_errors_ = []
            
            loss = 0
            loss_anomaly = 0
            
            start_node = 0
            for i in range(data.num_graphs): 
                recon_error = 0
                recon_error_ = 0
                num_nodes = (data.batch == i).sum().item() 
                end_node = start_node + num_nodes
                graph_num_nodes = end_node - start_node        
                
                node_loss = torch.norm(x_recon[start_node:end_node] - data.x[start_node:end_node], p='fro')**2
                graph_node_loss = node_loss/graph_num_nodes
                node_recon_error = graph_node_loss * 2

                adj_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i])
                edge_recon_error = adj_loss / 50
                
                # edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
                # edge_index = edges.t()
                
                # recon_z_node_loss = torch.norm(z[start_node:end_node] - z_tilde[start_node:end_node], p='fro')**2
                # graph_z_node_loss = recon_z_node_loss/graph_num_nodes

                # recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
                # graph_recon_loss = (graph_z_node_loss / 2) + (recon_z_graph_loss / 2)

                z_dist = torch.pdist(z[start_node:end_node])
                z_tilde_dist = torch.pdist(z_tilde[start_node:end_node])
                w_distance = torch.tensor(wasserstein_distance(z_dist.detach().cpu().numpy(), z_tilde_dist.detach().cpu().numpy()), device='cpu') 
                
                # print(f'node_recon_error: {node_recon_error}')
                # print(f'edge_recon_error: {edge_recon_error}')
                # # print(f'graph_recon_loss: {graph_recon_loss}')
                # print(f'w_distance: {w_distance}')
                
                recon_error += node_recon_error + edge_recon_error + w_distance
                recon_errors.append(recon_error.item())

                # test loss
                l1_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i]) / 20
                
                # recon_z_node_loss_ = torch.norm(z[start_node:end_node] - z_tilde[start_node:end_node], p='fro')**2
                # graph_z_node_loss_ = recon_z_node_loss_/graph_num_nodes
                
                # recon_z_graph_loss_ = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
                # l3_loss = (graph_z_node_loss_/2) + (recon_z_graph_loss_/2)
                
                if data[i].y.item() == 0:
                    loss += l1_loss + node_recon_error + w_distance
                else:
                    loss_anomaly += l1_loss + node_recon_error + w_distance

                max_ = torch.norm(z_g_train - z_g_test[i], p='fro')**2
                mean_ = torch.norm(z_g_mean_train - z_g_mean_test[i], p='fro')**2
                recon_error_ = max_ + mean_
                recon_errors_.append(recon_error_.item())
                
                start_node = end_node
                
            print(recon_errors_)
            # triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) / 10
            # l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) * 3
            # loss += triplet_loss + l2_loss
            total_loss += loss.item()
            
            if loss_anomaly != 0:
                total_loss_anomaly += loss_anomaly.item()
            else:
                print('loss_anomaly = 0')
            
            all_scores.extend(recon_errors_)
            all_labels.extend(data.y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auroc = auc(fpr, tpr)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    precision, recall, _ = precision_recall_curve(all_labels, all_scores)
    auprc = auc(recall, precision)

    if auroc > max_AUC:
        max_AUC=auroc

    # 추가된 평가 지표
    pred_labels = (all_scores > optimal_threshold).astype(int)
    precision = precision_score(all_labels, pred_labels)
    recall = recall_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)
    
    return auroc, auprc, precision, recall, f1, max_AUC, total_loss / len(val_loader), total_loss_anomaly / len(val_loader)


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='DHFR')
parser.add_argument("--assets-root", type=str, default="./assets")
parser.add_argument("--data-root", type=str, default='./dataset/data')

parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=10)
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
patience: list = args.patience
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

dataset_AN: bool = args.dataset_AN

set_seed(random_seed)

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
graph_dataset_ = TUDataset(root=data_root, name=dataset_name)

prefix = os.path.join(data_root, dataset_name, 'raw', dataset_name)
filename_node_attrs=prefix + '_node_attributes.txt'
node_attrs=[]

try:
    with open(filename_node_attrs) as f:
        for line in f:
            line = line.strip("\s\n")
            attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
            node_attrs.append(np.array(attrs))
except IOError:
    print('No node attributes')
    
node_attrs = np.array(node_attrs)

graph_dataset = []
node_idx = 0
for i in range(len(graph_dataset_)):
    old_data = graph_dataset_[i]
    num_nodes = old_data.num_nodes
    new_x = torch.tensor(node_attrs[node_idx:node_idx+num_nodes], dtype=torch.float)
    new_data = Data(x=new_x, edge_index=old_data.edge_index, y=old_data.y)
    graph_dataset.append(new_data)
    node_idx += num_nodes

print(graph_dataset[0].x)  # 새 데이터셋의 첫 번째 그래프 x 확인

labels = np.array([data.y.item() for data in graph_dataset])

print(f'Number of graphs: {len(graph_dataset)}')
print(f'Number of features: {graph_dataset[0].x.shape[1]}')
print(f'Number of edge features: {graph_dataset_.num_edge_features}')
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
num_features = graph_dataset[0].x.size()[1]
max_nodes = max([graph_dataset[i].num_nodes for i in range(len(graph_dataset))])  # 데이터셋에서 최대 노드 수 계산

model = GRAPH_AUTOENCODER2(num_features, hidden_dims, max_nodes, dropout_rate=0.1).to(device)
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
    
    model = GRAPH_AUTOENCODER2(num_features, hidden_dims, max_nodes, dropout_rate=0.1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2 regularization
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)
    early_stopping = EarlyStopping(patience=777, verbose=True)
    
    for epoch in range(epochs):
        train_loss, z_g_train, z_g_mean_train = train(model, train_loader, optimizer)
        auroc, auprc, precision, recall, f1, max_AUC, test_loss, test_loss_anomaly = evaluate_model(model, val_loader, z_g_train, z_g_mean_train)
        scheduler.step(auprc)  # AUC 기반으로 학습률 조정
        early_stopping(auroc, model)

        print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation loss = {test_loss:.4f}, Validation loss_anomaly = {test_loss_anomaly:.4f}, Validation AUC = {auroc:.4f}, Validation MAX_AUC = {max_AUC:.4f}, Validation AURPC = {auprc:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}')
        wandb.log({"epoch": epoch, 
                   "train loss": train_loss, 
                   "test loss": test_loss, 
                    "test loss_anomaly": test_loss_anomaly, 
                   "test AUC": auroc, 
                   "test max_AUC": max_AUC,
                   "test AURPC": auprc,
                   "precision": precision,
                   "recall": recall,
                   "f1": f1
                   })
        
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print("\n")
        

wandb.finish()
