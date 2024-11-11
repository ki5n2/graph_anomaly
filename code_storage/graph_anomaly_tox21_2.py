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

from scipy.stats import wasserstein_distance
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from module.model import ResidualBlock, GRAPH_AUTOENCODER
from module.loss import info_nce_loss, Triplet_loss, loss_cal
from module.utils import set_seed, set_device, EarlyStopping
        
        
#%%
'''TARIN'''
def train(model, train_loader, optimizer):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        adj, z, z_g, x_recon, adj_recon_list, z_tilde, z_tilde_g, aug_z_g = model(data)
        
        loss = 0
        start_node = 0
        
        for i in range(data.num_graphs): 
            num_nodes = (data.batch == i).sum().item() 
            end_node = start_node + num_nodes
            # graph_num_nodes = end_node - start_node        
            
            adj_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i])
            l1_loss = adj_loss / 2
            
            loss += l1_loss
            
            start_node = end_node
        
        node_loss = torch.norm(x_recon - data.x, p='fro')**2
        node_loss = (node_loss/x_recon.size(0)) / 20
        
        recon_z_node_loss = torch.norm(z - z_tilde, p='fro')**2
        graph_z_node_loss = recon_z_node_loss / (z.size(1) * 2)
        
        z_g_dist = torch.pdist(z_g)
        z_tilde_g_dist = torch.pdist(z_tilde_g)
        w_distance = torch.tensor(wasserstein_distance(z_g_dist.detach().cpu().numpy(), z_tilde_g_dist.detach().cpu().numpy()), device='cuda')
        w_distance = w_distance * 50
        
        info_nce_loss_value = info_nce_loss(z_g, aug_z_g)
        
        # triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) / 10
        # l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) * 3
        loss += node_loss + graph_z_node_loss + w_distance + info_nce_loss_value
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(train_loader)


#%%
'''EVALUATION'''
def evaluate_model(model, val_loader):
    model.eval()
    total_loss = 0
    max_AUC=0
    
    all_labels = []
    all_scores = []
    
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)  
            adj, z, z_g, x_recon, adj_recon_list, z_tilde, z_tilde_g, aug_z_g = model(data)
            
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
                node_recon_error = graph_node_loss / 20
                
                adj_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i])
                edge_recon_error = adj_loss
                    
                # edges = (adj_recon_list[i] > threshold).nonzero(as_tuple=False)
                # edge_index = edges.t()
                
                recon_z_node_loss = torch.norm(z[start_node:end_node] - z_tilde[start_node:end_node], p='fro')**2
                graph_z_node_loss = recon_z_node_loss/graph_num_nodes

                recon_z_graph_loss = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
                graph_recon_loss = (graph_z_node_loss) + (recon_z_graph_loss)
                print(node_recon_error)
                print(edge_recon_error)
                print(graph_recon_loss)
                recon_error += node_recon_error + edge_recon_error + graph_recon_loss
                recon_errors.append(recon_error.item())

                # test loss
                l1_loss = F.binary_cross_entropy(adj_recon_list[i], adj[i])
                
                recon_z_node_loss_ = torch.norm(z[start_node:end_node] - z_tilde[start_node:end_node], p='fro')**2
                graph_z_node_loss_ = recon_z_node_loss_/graph_num_nodes
                
                recon_z_graph_loss_ = torch.norm(z_g[i] - z_tilde_g[i], p='fro')**2
                l3_loss = (graph_z_node_loss_ / 10) + (recon_z_graph_loss_ / 10)
                            
                loss += l1_loss + l3_loss

                start_node = end_node
            
            node_loss = torch.norm(x_recon - data.x, p='fro')**2
            node_loss = (node_loss/x_recon.size(0)) / 20
            # triplet_loss = torch.sum(Triplet_loss(target_z, pos_sub_z_g, neg_sub_z_g)) / 10
            # l2_loss = torch.sum(loss_cal(z_prime_g_mlp, z_g_mlp)) * 3
            loss += node_loss
            total_loss += loss.item()
            
            all_scores.extend(recon_errors)
            all_labels.extend(data.y.cpu().numpy())

    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    auroc = auc(fpr, tpr)
    
    if auroc > max_AUC:
        max_AUC=auroc

    # 추가된 평가 지표
    pred_labels = (all_scores > optimal_threshold).astype(int)
    precision = precision_score(all_labels, pred_labels)
    recall = recall_score(all_labels, pred_labels)
    f1 = f1_score(all_labels, pred_labels)
    
    return auroc, precision, recall, f1, max_AUC, total_loss / len(val_loader)


#%%
'''ARGPARSER'''
parser = argparse.ArgumentParser()

parser.add_argument("--dataset-name", type=str, default='Tox21_p53_training')
parser.add_argument("--assets-root", type=str, default="./assets")
parser.add_argument("--data-root", type=str, default='./dataset/data')

parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--n-cross-val", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=2000)
parser.add_argument("--random-seed", type=int, default=42)
parser.add_argument("--test-batch-size", type=int, default=128)
parser.add_argument("--n-test-anomaly", type=int, default=400)
parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256, 128])

parser.add_argument("--factor", type=float, default=0.5)
parser.add_argument("--step-size", type=int, default=20)
parser.add_argument("--test-size", type=float, default=0.25)
parser.add_argument("--weight-decay", type=float, default=1e-5)
parser.add_argument("--learning-rate", type=float, default=0.00001)

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
graph_dataset = TUDataset(root=data_root, name=dataset_name).shuffle()

print(f'Number of graphs: {len(graph_dataset)}')
print(f'Number of features: {graph_dataset.num_features}')
print(f'Number of edge features: {graph_dataset.num_edge_features}')

dataset_normal = [data for data in graph_dataset if data.y.item() == 0]
dataset_anomaly = [data for data in graph_dataset if data.y.item() == 1]

print(f"Number of normal samples: {len(dataset_normal)}")
print(f"Number of anomaly samples: {len(dataset_anomaly)}")

train_normal_data, test_normal_data = train_test_split(dataset_normal, test_size=test_size, random_state=random_seed)
evaluation_data = test_normal_data + dataset_anomaly[:n_test_anomaly]

train_loader = DataLoader(train_normal_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(evaluation_data, batch_size=test_batch_size, shuffle=True)

print(f"Number of samples in the evaluation dataset: {len(evaluation_data)}")
print(f"Number of test normal data: {len(test_normal_data)}")
print(f"Number of test anomaly samples: {len(dataset_anomaly[:n_test_anomaly])}")
print(f"Ratio of test anomaly: {len(dataset_anomaly[:n_test_anomaly]) / len(evaluation_data)}")


#%%
'''MODEL AND OPTIMIZER DEFINE'''
num_features = graph_dataset.num_features
max_nodes = max([graph_dataset[i].num_nodes for i in range(len(graph_dataset))])  # 데이터셋에서 최대 노드 수 계산

model = GRAPH_AUTOENCODER(num_features, hidden_dims, max_nodes, dropout_rate=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # L2 regularization
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
early_stopping = EarlyStopping(patience=777, verbose=True)


#%%
'''TRAIN PROCESS'''
torch.autograd.set_detect_anomaly(True)  

   
for epoch in range(epochs):
    train_loss = train(model, train_loader, optimizer)
    auroc, precision, recall, f1, max_AUC, test_loss = evaluate_model(model, val_loader)
    scheduler.step(auroc)  # AUC 기반으로 학습률 조정
    early_stopping(auroc, model)

    print(f'Epoch {epoch+1}: Training Loss = {train_loss:.4f}, Validation loss = {test_loss:.4f}, Validation AUC = {auroc:.4f}, Validation MAX_AUC = {max_AUC:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}')
    wandb.log({"epoch": epoch, 
               "train loss": train_loss, 
               "test loss": test_loss, 
               "test AUC": auroc, 
               "test max_AUC": max_AUC,
               "precision": precision,
               "recall": recall,
               "f1": f1
               })
        
    if early_stopping.early_stop:
        print("Early stopping")
        break

    print("\n")
        

wandb.finish()
