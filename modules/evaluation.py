import torch
import numpy as np
from sklearn.metrics import auc, roc_curve, precision_score, recall_score, f1_score, precision_recall_curve
from .visualization import DensityBasedScoring, plot_error_distribution
from .utils import process_batch_graphs, persistence_stats_loss
import os
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, reconstruction_errors, epoch, dataset_name, device, alpha=1.0, gamma=10.0):
    model.eval()
    total_loss_ = 0
    total_loss_anomaly_ = 0
    all_labels = []
    all_scores = []
    reconstruction_errors_test = []
    
    with torch.no_grad():
        for data in test_loader:
            data = process_batch_graphs(data)
            data = data.to(device)
            x, edge_index, batch, num_graphs, true_stats = data.x, data.edge_index, data.batch, data.num_graphs, data.true_stats
            
            # Forward pass
            x_recon, stats_pred = model(
                x, edge_index, batch, num_graphs, is_pretrain=False
            )
            
            start_node = 0
            for i in range(num_graphs):
                num_nodes = (batch == i).sum().item()
                end_node = start_node + num_nodes
                
                # 노드 재구성 오류 계산
                if dataset_name == 'AIDS':
                    node_loss = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2
                else:
                    node_loss = torch.norm(x[start_node:end_node] - x_recon[start_node:end_node], p='fro')**2 / num_nodes
                    
                node_loss = node_loss.item() * alpha
                    
                # Persistent homology 재구성 오류 계산
                stats_loss = persistence_stats_loss(
                    stats_pred[i], 
                    true_stats[i]
                ).item() * gamma

                # 전체 오류 계산
                total_error = node_loss + stats_loss
                
                # 결과 저장
                reconstruction_errors_test.append({
                    'reconstruction': node_loss,
                    'topology': stats_loss,
                    'type': 'test_normal' if data.y[i].item() == 0 else 'test_anomaly'
                })
                
                if data.y[i].item() == 0:
                    total_loss_ += total_error
                else:
                    total_loss_anomaly_ += total_error
                    
                start_node = end_node
            
            all_labels.extend(data.y.cpu().numpy())
    
    # 시각화를 위한 데이터 변환
    visualization_data = {
        'normal': [
            {'reconstruction': error['reconstruction'], 
             'topology': error['topology']}
            for error in reconstruction_errors_test if error['type'] == 'test_normal'
        ],
        'anomaly': [
            {'reconstruction': error['reconstruction'], 
             'topology': error['topology']}
            for error in reconstruction_errors_test if error['type'] == 'test_anomaly'
        ]
    }

    # 데이터 분리 및 특징 벡터 구성
    train_data = np.array([[error['reconstruction'], error['topology']] 
                          for error in reconstruction_errors if error['type'] == 'train_normal'])
    test_normal = np.array([[error['reconstruction'], error['topology']] 
                           for error in reconstruction_errors_test if error['type'] == 'test_normal'])
    test_anomaly = np.array([[error['reconstruction'], error['topology']] 
                            for error in reconstruction_errors_test if error['type'] == 'test_anomaly'])
    
    # 밀도 기반 스코어링 적용
    density_scorer = DensityBasedScoring()
    density_scorer.fit(train_data)
    
    # 이상 스코어 계산
    normal_scores = density_scorer.score_samples(test_normal)
    anomaly_scores = density_scorer.score_samples(test_anomaly)
    
    # 전체 스코어 및 라벨 구성
    all_scores = np.concatenate([normal_scores, anomaly_scores])
    all_labels = np.array([0] * len(normal_scores) + [1] * len(anomaly_scores))
    
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
    
    # 결과 시각화
    plot_density_analysis(density_scorer, train_data, test_normal, test_anomaly, 
                         normal_scores, anomaly_scores, epoch, dataset_name)
    
    total_loss_mean = total_loss_ / sum(all_labels == 0)
    total_loss_anomaly_mean = total_loss_anomaly_ / sum(all_labels == 1)
    
    return auroc, auprc, precision, recall, f1, total_loss_mean, total_loss_anomaly_mean, all_scores, all_labels, reconstruction_errors_test, visualization_data

def plot_density_analysis(density_scorer, train_data, test_normal, test_anomaly, 
                         normal_scores, anomaly_scores, epoch, dataset_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 원본 산점도와 밀도 등고선
    contour = density_scorer.plot_density_contours(train_data, test_normal, test_anomaly, ax1)
    fig.colorbar(contour, ax=ax1)
    ax1.set_xlabel('Node Reconstruction Error', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Topology Reconstruction Error', fontsize=12, fontweight='bold')
    ax1.set_title('Density-based Anomaly Detection')
    ax1.legend()
    
    # 이상 스코어 분포
    ax2.hist(normal_scores, bins=50, alpha=0.5, density=True, label='Normal', color='green')
    ax2.hist(anomaly_scores, bins=50, alpha=0.5, density=True, label='Anomaly', color='red')
    ax2.set_xlabel('Anomaly Score (-log density)')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Anomaly Scores')
    ax2.legend()
    
    plt.tight_layout()
    save_path = f'density_analysis/{dataset_name}_epoch_{epoch}.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
