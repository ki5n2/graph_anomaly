import os
import gc
import json
import time
import wandb
import torch
import argparse
import numpy as np
from modules.models import GRAPH_AUTOENCODER
from modules.training import train, train_bert_embedding, train_bert_edge_reconstruction
from modules.evaluation import evaluate_model
from modules.utils import (
    set_seed, set_device, get_ad_split_TU, get_data_loaders_TU,
    EarlyStopping
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default='COX2')
    parser.add_argument("--data-root", type=str, default='./dataset')
    parser.add_argument("--assets-root", type=str, default="./assets")
    parser.add_argument("--n-head-BERT", type=int, default=2)
    parser.add_argument("--n-layer-BERT", type=int, default=2)
    parser.add_argument("--n-head", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--BERT-epochs", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--n-cluster", type=int, default=1)
    parser.add_argument("--step-size", type=int, default=20)
    parser.add_argument("--n-cross-val", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=300)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--hidden-dims", nargs='+', type=int, default=[256])
    parser.add_argument("--factor", type=float, default=0.5)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--dropout-rate", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--node-theta", type=float, default=0.03)
    parser.add_argument("--adj-theta", type=float, default=0.01)
    
    return parser.parse_args()

def run(dataset_name, random_seed, dataset_AN, trial, device, epoch_results=None):
    if epoch_results is None:
        epoch_results = {}
    epoch_interval = 10
    
    set_seed(random_seed)    
    all_results = []
    splits = get_ad_split_TU(dataset_name, args.n_cross_val)
    split = splits[trial]
    
    loaders, meta = get_data_loaders_TU(dataset_name, args.batch_size, 
                                       args.test_batch_size, split, dataset_AN)
    num_features = meta['num_feat']
    max_nodes = meta['max_nodes']
    
    # BERT 모델 경로
    bert_save_path = f'BERT_model/pretrained_bert_{dataset_name}_fold{trial}.pth'
    
    model = GRAPH_AUTOENCODER(
        num_features=num_features, 
        hidden_dims=args.hidden_dims, 
        max_nodes=max_nodes,
        nhead_BERT=args.n_head_BERT,
        nhead=args.n_head,
        num_layers_BERT=args.n_layer_BERT,
        num_layers=args.n_layer,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    train_loader = loaders['train']
    test_loader = loaders['test']
    
    # BERT 사전학습
    if os.path.exists(bert_save_path):
        print("Loading pretrained BERT...")
        model.encoder.load_state_dict(torch.load(bert_save_path))
    else:
        print("Training BERT from scratch...")
        pretrain_params = list(model.encoder.parameters())
        bert_optimizer = torch.optim.Adam(pretrain_params, lr=args.learning_rate)
        
        print("Stage 1-1: Mask token reconstruction...")
        for epoch in range(1, args.BERT_epochs + 1):
            train_loss, num_sample = train_bert_embedding(
                model, train_loader, bert_optimizer, device
            )
            if epoch % args.log_interval == 0:
                print(f'BERT Mask Training Epoch {epoch}: Loss = {train_loss:.4f}')
        
        print("Stage 1-2: Edge reconstruction...")
        for epoch in range(1, args.BERT_epochs + 1):
            train_loss, num_sample = train_bert_edge_reconstruction(
                model, train_loader, bert_optimizer, device
            )
            if epoch % args.log_interval == 0:
                print(f'BERT Edge Training Epoch {epoch}: Loss = {train_loss:.4f}')
        
        print("Saving pretrained BERT...")
        torch.save(model.encoder.state_dict(), bert_save_path)
    
    # 메인 훈련
    print("Stage 2: Main training...")
    recon_optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    current_time = time.strftime("%Y_%m_%d_%H_%M")
    best_auroc = 0
    
    for epoch in range(1, args.epochs + 1):
        train_loss, num_sample, train_errors = train(
            model, train_loader, recon_optimizer, device, epoch, dataset_name
        )
        
        if epoch % args.log_interval == 0:
            metrics = evaluate_model(
                model, test_loader, train_errors, epoch, 
                dataset_name, device
            )
            auroc, auprc, precision, recall, f1, test_loss, test_loss_anomaly = metrics[:7]
            
            print(f'Epoch {epoch}:')
            print(f'Training Loss = {train_loss:.4f}')
            print(f'Test Loss = {test_loss:.4f}, Anomaly Loss = {test_loss_anomaly:.4f}')
            print(f'AUROC = {auroc:.4f}, AUPRC = {auprc:.4f}')
            print(f'Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}')
            
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "test_loss": test_loss,
                "test_loss_anomaly": test_loss_anomaly,
                "auroc": auroc,
                "auprc": auprc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })
            
            if auroc > best_auroc:
                best_auroc = auroc
                torch.save(model.state_dict(), f'models/best_model_{dataset_name}_{current_time}.pth')
            
            if epoch % epoch_interval == 0:
                if epoch not in epoch_results:
                    epoch_results[epoch] = {
                        'aurocs': [], 'auprcs': [], 
                        'precisions': [], 'recalls': [], 'f1s': []
                    }
                
                epoch_results[epoch]['aurocs'].append(auroc)
                epoch_results[epoch]['auprcs'].append(auprc)
                epoch_results[epoch]['precisions'].append(precision)
                epoch_results[epoch]['recalls'].append(recall)
                epoch_results[epoch]['f1s'].append(f1)
    
    return best_auroc, epoch_results

def main():
    args = parse_args()
    
    # wandb 초기화
    wandb.init(
        project="graph_anomaly_detection",
        entity="your_entity",
        config=vars(args)
    )
    
    # 디바이스 설정
    device = set_device()
    print(f"Using device: {device}")
    
    # 데이터셋 설정
    dataset_AN = args.dataset_name in ['AIDS', 'NCI1', 'DHFR']
    
    # 결과 저장용 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # 크로스 밸리데이션 실행
    ad_aucs = []
    fold_times = []
    epoch_results = {}
    start_time = time.time()
    
    for trial in range(args.n_cross_val):
        fold_start = time.time()
        print(f"\nStarting fold {trial + 1}/{args.n_cross_val}")
        
        ad_auc, epoch_results = run(
            args.dataset_name, args.random_seed, 
            dataset_AN, trial, device, 
            epoch_results=epoch_results
        )
        
        ad_aucs.append(ad_auc)
        fold_duration = time.time() - fold_start
        fold_times.append(fold_duration)
        print(f"Fold {trial + 1} finished in {fold_duration:.2f} seconds")
    
    # 최종 결과 계산 및 저장
    process_final_results(ad_aucs, epoch_results, args.dataset_name, start_time)
    
    wandb.finish()

def process_final_results(ad_aucs, epoch_results, dataset_name, start_time):
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
    
    best_epoch = max(epoch_means.keys(), key=lambda x: epoch_means[x]['auroc'])
    
    print("\n=== Final Results ===")
    for epoch in sorted(epoch_means.keys()):
        print(f"Epoch {epoch}: AUROC = {epoch_means[epoch]['auroc']:.4f} ± {epoch_stds[epoch]['auroc']:.4f}")
    
    print(f"\nBest performance at epoch {best_epoch}:")
    print(f"AUROC = {epoch_means[best_epoch]['auroc']:.4f} ± {epoch_stds[best_epoch]['auroc']:.4f}")
    print(f"AUPRC = {epoch_means[best_epoch]['auprc']:.4f} ± {epoch_stds[best_epoch]['auprc']:.4f}")
    print(f"F1 = {epoch_means[best_epoch]['f1']:.4f} ± {epoch_stds[best_epoch]['f1']:.4f}")
    
    total_time = time.time() - start_time
    results = f'AUC: {np.mean(ad_aucs)*100:.2f}±{np.std(ad_aucs)*100:.2f}'
    print(f'[FINAL RESULTS] {results}')
    print(f"Total execution time: {total_time:.2f} seconds")
    
    current_time = time.strftime("%Y_%m_%d_%H_%M")
    results_path = f'results/results_{dataset_name}_{current_time}.json'
    
    with open(results_path, 'w') as f:
        json.dump({
            'epoch_means': epoch_means,
            'epoch_stds': epoch_stds,
            'best_epoch': int(best_epoch),
            'final_auroc_mean': float(np.mean(ad_aucs)),
            'final_auroc_std': float(np.std(ad_aucs)),
            'total_time': total_time
        }, f, indent=4)

if __name__ == "__main__":
    main()
