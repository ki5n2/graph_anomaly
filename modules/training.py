import torch
import torch.nn.functional as F
from .visualization import plot_error_distribution
import matplotlib.pyplot as plt
import os
from .utils import process_batch_graphs, persistence_stats_loss

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
        
        # BERT 인코딩 및 마스크 토큰 예측
        node_embeddings, masked_outputs = model(
            x, edge_index, batch, num_graphs,
            mask_indices=mask_indices,
            is_pretrain=True
        )
        
        # 마스크된 노드의 특징 재구성 손실 계산
        mask_loss = torch.norm(masked_outputs - x[mask_indices], p='fro')**2 / mask_indices.sum()
        
        mask_loss.backward()
        bert_optimizer.step()
        total_loss += mask_loss.item()
        num_sample += num_graphs
        
        print(f'mask_node_feature:{mask_loss.item()}')
    
    return total_loss / len(train_loader), num_sample

def train_bert_edge_reconstruction(model, train_loader, bert_optimizer, device):
    model.train()
    total_loss = 0
    num_sample = 0
    
    for data in train_loader:
        bert_optimizer.zero_grad()
        data = data.to(device)
        x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
        
        # Edge reconstruction 수행
        node_embeddings, adj_recon_list = model(
            x, edge_index, batch, num_graphs,
            is_pretrain=True,
            edge_training=True
        )
        
        start_idx = 0
        edge_loss = 0
        
        for i in range(num_graphs):
            # 현재 그래프의 노드 수 계산
            mask = (batch == i)
            num_nodes = mask.sum().item()
            end_idx = start_idx + num_nodes
            
            # 현재 그래프의 edge_index 추출 및 조정
            graph_edges = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < end_idx)]
            graph_edges = graph_edges - start_idx
            
            # 실제 adjacency matrix 생성 (max_nodes 크기로)
            true_adj = torch.zeros((model.encoder.edge_decoder.max_nodes, 
                                  model.encoder.edge_decoder.max_nodes), 
                                 device=device)
            true_adj[graph_edges[0], graph_edges[1]] = 1
            true_adj = true_adj + true_adj.t()
            true_adj = (true_adj > 0).float()
            
            # 손실 계산 (실제 노드가 있는 부분만)
            adj_recon = adj_recon_list[i]  # 이미 max_nodes 크기로 패딩된 상태
            
            # 실제 노드가 있는 부분의 마스크 생성
            node_mask = torch.zeros_like(adj_recon, dtype=torch.bool)
            node_mask[:num_nodes, :num_nodes] = True
            
            # MSE 손실 계산 (실제 노드 영역만)
            mse_loss = torch.sum((adj_recon[node_mask] - true_adj[node_mask]) ** 2) / (node_mask.sum())
            edge_loss += mse_loss
            
            start_idx = end_idx
        
        edge_loss = edge_loss / num_graphs
        
        edge_loss.backward()
        bert_optimizer.step()
        
        total_loss += edge_loss.item()
        num_sample += num_graphs
        
        print(f'edge_reconstruction_mse_loss: {edge_loss.item()}')
    
    return total_loss / len(train_loader), num_sample

def train(model, train_loader, recon_optimizer, device, epoch, dataset_name, alpha=1.0, gamma=10.0):
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
        x_recon, stats_pred = model(
            x, edge_index, batch, num_graphs, is_pretrain=False
        )
        
        loss = 0
        node_loss = 0
        start_node = 0
        for i in range(num_graphs):
            num_nodes = (batch == i).sum().item()
            end_node = start_node + num_nodes

            if dataset_name == 'AIDS':
                node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2
            else:
                node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
            
            node_loss += node_loss_
            start_node = end_node
            
        stats_loss = persistence_stats_loss(stats_pred, true_stats)
        stats_loss = gamma * stats_loss

        loss = node_loss + stats_loss
        
        print(f'node_loss: {node_loss}')
        print(f'stats_loss: {stats_loss}')
        
        num_sample += num_graphs
        loss.backward()
        recon_optimizer.step()
        total_loss += loss.item()

        # 모델 업데이트 완료 후 산점도 데이터 수집
        model.eval()  # 평가 모드로 전환
        with torch.no_grad():
            # Forward pass
            x_recon, stats_pred = model(
                x, edge_index, batch, num_graphs, is_pretrain=False
            )
            
            # 산점도 데이터 수집
            if epoch % 5 == 0:
                start_node = 0
                for i in range(num_graphs):
                    num_nodes = (batch == i).sum().item()
                    end_node = start_node + num_nodes

                    if dataset_name == 'AIDS':
                        node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2
                    else:
                        node_loss_ = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
                    
                    node_loss_scaled = node_loss_.item() * alpha
                    stats_loss_scaled = persistence_stats_loss(
                        stats_pred[i], 
                        true_stats[i]
                    ).item() * gamma
                    
                    reconstruction_errors.append({
                        'reconstruction': node_loss_scaled,
                        'topology': stats_loss_scaled,
                        'type': 'train_normal'
                    })
                    
                    start_node = end_node

        if epoch % 5 == 0:
            plot_training_error_distribution(reconstruction_errors, epoch, dataset_name)

        model.train()  # 다시 훈련 모드로 전환

    return total_loss / len(train_loader), num_sample, reconstruction_errors

def plot_training_error_distribution(reconstruction_errors, epoch, dataset_name):
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    recon_errors = [point['reconstruction'] for point in reconstruction_errors]
    topo_errors = [point['topology'] for point in reconstruction_errors]
    
    # Normal scale plot
    ax1.scatter(recon_errors, topo_errors, c='blue', alpha=0.6)
    ax1.set_xlabel('Node Reconstruction Error')
    ax1.set_ylabel('Topology Reconstruction Error')
    ax1.set_title(f'Training Error Distribution (Epoch {epoch})')
    ax1.grid(True)

    # Log scale plot
    ax2.scatter(recon_errors, topo_errors, c='blue', alpha=0.6)
    ax2.set_xlabel('Node Reconstruction Error')
    ax2.set_ylabel('Topology Reconstruction Error')
    ax2.set_title(f'Training Error Distribution - Log Scale (Epoch {epoch})')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.grid(True)

    plt.tight_layout()
    save_path = f'error_distribution_plots/train_error_distribution_epoch_{epoch}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
