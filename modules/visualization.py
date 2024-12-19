import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import os
import json
from .utils import loocv_bandwidth_selection

class DensityBasedScoring:
    def __init__(self, bandwidth=None):
        self.bandwidth = bandwidth
        self.kde = None
        self.scaler = StandardScaler()
        
    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        if self.bandwidth is None:
            self.bandwidth, _ = loocv_bandwidth_selection(X_scaled)
        self.kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        self.kde.fit(X_scaled)
    
    def score_samples(self, X):
        X_scaled = self.scaler.transform(X)
        log_density = self.kde.score_samples(X_scaled)
        log_density = np.nan_to_num(log_density, neginf=-10000)
        anomaly_scores = -log_density
        anomaly_scores = np.clip(anomaly_scores, 0, 10000)
        return anomaly_scores
    
    def plot_density_contours(self, X_train, X_test_normal, X_test_anomaly, ax, num_points=100):
        # 그리드 범위 설정
        x_min = min(X_train[:, 0].min(), X_test_normal[:, 0].min(), X_test_anomaly[:, 0].min()) - 0.1
        x_max = max(X_train[:, 0].max(), X_test_normal[:, 0].max(), X_test_anomaly[:, 0].max()) + 0.1
        y_min = min(X_train[:, 1].min(), X_test_normal[:, 1].min(), X_test_anomaly[:, 1].min()) - 0.1
        y_max = max(X_train[:, 1].max(), X_test_normal[:, 1].max(), X_test_anomaly[:, 1].max()) + 0.1
        
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, num_points),
                            np.linspace(y_min, y_max, num_points))
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.score_samples(grid_points)
        Z = Z.reshape(xx.shape)
        
        levels = np.linspace(Z.min(), Z.max(), 20)
        contour = ax.contourf(xx, yy, Z, levels=levels, cmap='RdYlBu_r', alpha=0.7)
        
        scatter_params = {'alpha': 0.7, 'edgecolor': 'white', 's': 80, 'linewidth': 1.5}
        
        ax.scatter(X_train[:, 0], X_train[:, 1],
                  c='dodgerblue', label='Train (Normal)',
                  marker='o', **scatter_params)
        ax.scatter(X_test_normal[:, 0], X_test_normal[:, 1],
                  c='mediumseagreen', label='Test (Normal)',
                  marker='o', **scatter_params)
        ax.scatter(X_test_anomaly[:, 0], X_test_anomaly[:, 1],
                  c='crimson', label='Test (Anomaly)',
                  marker='o', **scatter_params)
        
        ax.set_xlabel('Reconstruction Error', fontsize=12, fontweight='bold')
        ax.set_ylabel('Topology Error', fontsize=12, fontweight='bold')
        
        ax.grid(True, linestyle='--', alpha=0.3, color='gray')
        ax.set_axisbelow(True)
        
        ax.legend(fontsize=10, frameon=True, facecolor='white', 
                 edgecolor='gray', loc='upper right')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=10)
        
        return contour

def plot_error_distribution(train_errors, test_errors, epoch, trial, dataset_name, current_time):
    # 데이터 분리
    train_normal_recon = [e['reconstruction'] for e in train_errors if e['type'] == 'train_normal']
    train_normal_cluster = [e['topology'] for e in train_errors if e['type'] == 'train_normal']
    
    test_normal_recon = [e['reconstruction'] for e in test_errors if e['type'] == 'test_normal']
    test_normal_cluster = [e['topology'] for e in test_errors if e['type'] == 'test_normal']
    
    test_anomaly_recon = [e['reconstruction'] for e in test_errors if e['type'] == 'test_anomaly']
    test_anomaly_cluster = [e['topology'] for e in test_errors if e['type'] == 'test_anomaly']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Normal scale plot
    ax1.scatter(train_normal_recon, train_normal_cluster, c='blue', label='Train (Normal)', alpha=0.6)
    ax1.scatter(test_normal_recon, test_normal_cluster, c='green', label='Test (Normal)', alpha=0.6)
    ax1.scatter(test_anomaly_recon, test_anomaly_cluster, c='red', label='Test (Anomaly)', alpha=0.6)
    
    ax1.set_xlabel('Reconstruction Error')
    ax1.set_ylabel('Topology Error')
    ax1.set_title(f'Error Distribution (Epoch {epoch})')
    ax1.legend()
    ax1.grid(True)

    # Log scale plot
    ax2.scatter(train_normal_recon, train_normal_cluster, c='blue', label='Train (Normal)', alpha=0.6)
    ax2.scatter(test_normal_recon, test_normal_cluster, c='green', label='Test (Normal)', alpha=0.6)
    ax2.scatter(test_anomaly_recon, test_anomaly_cluster, c='red', label='Test (Anomaly)', alpha=0.6)
    
    ax2.set_xlabel('Reconstruction Error')
    ax2.set_ylabel('Topology Error')
    ax2.set_title(f'Error Distribution - Log Scale (Epoch {epoch})')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    
    # Save plot and data
    for path_type in ['plot', 'json']:
        save_dir = f'error_distribution_{path_type}/{dataset_name}_time_{current_time}'
        os.makedirs(save_dir, exist_ok=True)
        
    plot_path = f'error_distribution_plot/{dataset_name}_time_{current_time}/epoch_{epoch}_fold_{trial}.png'
    plt.savefig(plot_path)
    plt.close()
    
    # Save data as JSON
    error_data = {
        'train_normal': [{'reconstruction': r, 'topology': c} 
                        for r, c in zip(train_normal_recon, train_normal_cluster)],
        'test_normal': [{'reconstruction': r, 'topology': c} 
                       for r, c in zip(test_normal_recon, test_normal_cluster)],
        'test_anomaly': [{'reconstruction': r, 'topology': c} 
                        for r, c in zip(test_anomaly_recon, test_anomaly_cluster)]
    }
    
    json_path = f'error_distribution_json/{dataset_name}_time_{current_time}/epoch_{epoch}_fold_{trial}.json'
    with open(json_path, 'w') as f:
        json.dump(error_data, f)