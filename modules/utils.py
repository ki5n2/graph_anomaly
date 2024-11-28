import re
import os
import gc
import torch
import random
import numpy as np
import gudhi as gd
import os.path as osp

from torch_geometric.data import Data
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Constant
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from scipy.spatial.distance import pdist, squareform
from torch_geometric.utils import to_dense_adj, to_undirected, to_networkx, to_scipy_sparse_matrix, degree, from_networkx

import networkx as nx

# 메모리 설정
torch.cuda.empty_cache()
gc.collect()



def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def set_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device


def node_iter(G):
    return G.nodes


def node_dict(G):
    return G.nodes


def prepare_dataset(graph_dataset, labels, train_idx, val_idx, dataset_AN):
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
    
    return train_dataset, val_dataset


def add_gaussian_perturbation(z, epsilon=0.1):
    """
    Add Gaussian perturbations to the encoder output z to create z'
    :param z: torch.Tensor, the output of the encoder
    :param epsilon: float, the standard deviation of the Gaussian noise to add
    :return: torch.Tensor, the perturbed output z'
    """
    # Gaussian noise generation
    noise = torch.randn_like(z) * epsilon
    z_prime = z + noise
        
    return z_prime


def randint_exclude(data, start_node, end_node):
    exclude = set(range(start_node, end_node))
    choices = list(set(range(data.x.size(0))) - exclude)
        
    return torch.tensor(choices[torch.randint(0, len(choices), (1,)).item()])


def extract_subgraph(data, node_idx, start_node, end_node, max_attempts=10, device='cuda'):
    attempts = 0
    subgraph = data.clone()
    while attempts < max_attempts:
        edge_index = to_undirected(subgraph.edge_index, num_nodes=subgraph.x.size(0))
        mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
        node_indices = torch.cat((edge_index[0][mask], edge_index[1][mask])).unique()

        if node_indices.numel() > 0:  # Check if there are connected nodes
            subgraph.x = data.x[node_indices]
            edge_mask = torch.tensor([start in node_indices and end in node_indices for start, end in data.edge_index.t()], dtype=torch.bool)
            subgraph.edge_index = data.edge_index[:, edge_mask]
            subgraph.edge_attr = data.edge_attr[edge_mask] if 'edge_attr' in data else None
                    
            node_to_graph = data.batch[node_indices]  # Find new graph IDs
            unique_graphs = node_to_graph.unique(sorted=True)
            batch_device = torch.arange(unique_graphs[0], unique_graphs[0] + 1).to(device)
            subgraph.batch = batch_device.repeat_interleave(
                len(subgraph.x)
                )
                
                # subgraph.batch = batch_device.repeat_interleave(
                # torch.bincount(node_to_graph).sort(descending=True)[0][0]
                # )
                    
            if 'y' in data:
                subgraph.y = data.y[unique_graphs]
                
            if 'ptr' in subgraph:
                del subgraph.ptr
                
            return subgraph

        # No connected nodes, select a new node_idx and try again
        if start_node <= node_idx < end_node:
            node_idx = torch.randint(start_node, end_node, size=(1,)).item()
        else:
            node_idx = randint_exclude(data, start_node, end_node).item()
        attempts += 1

    return None  # Return None after max_attempts
    

def batch_nodes_subgraphs_(data):
    batched_target_nodes = []
    batched_initial_nodes = []
    batched_target_node_features = []
    pos_subgraphs = []
    neg_subgraphs = []
        
    start_node = 0        
    for i in range(len(data)): 
        num_nodes = (data.batch == i).sum().item() 
        end_node = start_node + num_nodes
            
        target_node = torch.randint(start_node, end_node, size=(1,)).item()
        pos_subgraph = extract_subgraph(data, node_idx=target_node, start_node=start_node, end_node=end_node)
        target_node_feature = data.x[target_node]
                        
        initial_node = randint_exclude(data, start_node=start_node, end_node=end_node).item()
        neg_subgraph = extract_subgraph(data, node_idx=initial_node, start_node=start_node, end_node=end_node)
            
        batched_target_nodes.append(target_node)
        batched_initial_nodes.append(initial_node)
        pos_subgraphs.append(pos_subgraph)
        neg_subgraphs.append(neg_subgraph)
        batched_target_node_features.append(target_node_feature)
            
        start_node = end_node
            
    batched_pos_subgraphs = Batch.from_data_list(pos_subgraphs)
    batched_neg_subgraphs = Batch.from_data_list(neg_subgraphs)
    batched_target_node_features = torch.stack(batched_target_node_features)
    # return batched_target_nodes, batched_initial_nodes, batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features
    return batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features


def batch_nodes_subgraphs(data):
    batched_target_nodes = []
    batched_initial_nodes = []
    batched_target_node_features = []
    pos_subgraphs = []
    neg_subgraphs = []
    start_node = 0

    for i in range(len(data)):
        num_nodes = (data.batch == i).sum().item()
        end_node = start_node + num_nodes

        max_attempts = 10
        for _ in range(max_attempts):
            target_node = torch.randint(start_node, end_node, size=(1,)).item()
            pos_subgraph = extract_subgraph(data, node_idx=target_node, start_node=start_node, end_node=end_node)

            initial_node = randint_exclude(data, start_node=start_node, end_node=end_node).item()
            neg_subgraph = extract_subgraph(data, node_idx=initial_node, start_node=start_node, end_node=end_node)

            if pos_subgraph is not None and neg_subgraph is not None:
                target_node_feature = data.x[target_node]
                batched_target_nodes.append(target_node)
                batched_initial_nodes.append(initial_node)
                pos_subgraphs.append(pos_subgraph)
                neg_subgraphs.append(neg_subgraph)
                batched_target_node_features.append(target_node_feature)
                break
        else:
            print(f"Warning: Failed to find valid subgraphs for graph {i}")

        start_node = end_node

    if not pos_subgraphs or not neg_subgraphs:
        print("Error: No valid subgraphs found")
        return None, None, None

    batched_pos_subgraphs = Batch.from_data_list(pos_subgraphs)
    batched_neg_subgraphs = Batch.from_data_list(neg_subgraphs)
    batched_target_node_features = torch.stack(batched_target_node_features)

    return batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features

    
def adj_original__(edge_index, batch):
    adj_matrices = []
    for batch_idx in torch.unique(batch):
        # 현재 그래프에 속하는 노드들의 마스크
        mask = (batch == batch_idx)
        # 현재 그래프의 에지 인덱스 추출
        sub_edge_index_ = edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]
        # 노드 인덱스를 0부터 시작하도록 재매핑
        _, sub_edge_index = torch.unique(sub_edge_index_, return_inverse=True)
        sub_edge_index = sub_edge_index.reshape(2, -1)
        # 인접 행렬 생성
        adj_matrix = to_dense_adj(sub_edge_index, max_num_nodes=sum(mask).item())[0]
        adj_matrices.append(adj_matrix)
        
    return adj_matrices
    

def adj_original(edge_index, batch, max_nodes, device='cuda'):
    adj_matrices = []
    # 가장 큰 그래프의 노드 수 찾기
    
    for batch_idx in torch.unique(batch):
        # 현재 그래프에 속하는 노드들의 마스크
        mask = (batch == batch_idx)
        # 현재 그래프의 에지 인덱스 추출
        sub_edge_index = edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]
        
        # 노드 인덱스를 0부터 시작하도록 재매핑
        _, sub_edge_index = torch.unique(sub_edge_index, return_inverse=True)
        sub_edge_index = sub_edge_index.reshape(2, -1)
        
        # 현재 그래프의 노드 수
        num_nodes = sum(mask).item()
        
        # 인접 행렬 생성 (현재 그래프 크기로)
        adj_matrix = to_dense_adj(sub_edge_index, max_num_nodes=num_nodes)[0]
        
        # 최대 크기에 맞춰 패딩
        padded_adj_matrix = torch.zeros(max_nodes, max_nodes)
        padded_adj_matrix[:num_nodes, :num_nodes] = adj_matrix
        padded_adj_matrix = padded_adj_matrix.to(device)
        
        adj_matrices.append(padded_adj_matrix)
    
    return adj_matrices


def adj_original_(edge_index, batch):
    adj_matrices = []
    for batch_idx in torch.unique(batch):
        # 현재 그래프에 속하는 노드들의 마스크
        mask = (batch == batch_idx)
        # 현재 그래프의 에지 인덱스 추출
        sub_edge_index = edge_index[:, mask[edge_index[0]] & mask[edge_index[1]]]
        # 노드 인덱스를 0부터 시작하도록 재매핑
        _, sub_edge_index = torch.unique(sub_edge_index, return_inverse=True)
        sub_edge_index = sub_edge_index.reshape(2, -1)
        # 인접 행렬 생성
        adj_matrix = to_dense_adj(sub_edge_index, max_num_nodes=sum(mask).item())[0]
        adj_matrices.append(adj_matrix)
        
    return adj_matrices


def adj_recon(z, z_prime, batch):
    adj_recon_list = []
    adj_recon_prime_list = []
        
    # Iterate over each graph in the batch
    for batch_idx in torch.unique(batch):
        mask = (batch == batch_idx)
            
        # Select the latent vectors corresponding to the current graph
        z_graph = z[mask]
        z_prime_graph = z_prime[mask]
                
        # Reconstruct adjacency matrices for the current graph
        adj_recon_graph = torch.sigmoid(torch.mm(z_graph, z_graph.t()))
        adj_recon_prime_graph = torch.sigmoid(torch.mm(z_prime_graph, z_prime_graph.t()))

        # Append the reconstructed matrices to the lists
        adj_recon_list.append(adj_recon_graph)
        adj_recon_prime_list.append(adj_recon_prime_graph)
                
    return adj_recon_list, adj_recon_prime_list


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


'''SIGNET'''
def get_ad_split_TU(dataset_name, n_cross_val):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset = TUDataset(path, name=dataset_name)
    data_list = []
    label_list = []

    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())

    kfd = StratifiedKFold(n_splits=n_cross_val, random_state=1, shuffle=True)

    splits = []
    for k, (train_index, test_index) in enumerate(kfd.split(data_list, label_list)):
        splits.append((train_index, test_index))

    return splits


def get_data_loaders_TU(dataset_name, batch_size, test_batch_size, split, dataset_AN):
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')
    dataset_ = TUDataset(path, name=dataset_name)
        
    prefix = os.path.join(path, dataset_name, 'raw', dataset_name)
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

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')
    # max_node_label = max(node_labels)

    dataset = []
    node_idx = 0
    for i in range(len(dataset_)):
        old_data = dataset_[i]
        num_nodes = old_data.num_nodes
        new_x = torch.tensor(node_attrs[node_idx:node_idx+num_nodes], dtype=torch.float)
        node_label_graph = torch.tensor(node_labels[node_idx:node_idx+num_nodes], dtype=torch.float)   

        # if new_x.shape[0] == node_label_graph.shape[0]:
        #     print(True)
        # else:
        #     print(False)
            
        if dataset_name != 'NCI1':
            new_data = Data(x=new_x, edge_index=old_data.edge_index, y=old_data.y, node_label = node_label_graph)
        else:
            new_data = Data(x=old_data.x, edge_index=old_data.edge_index, y=old_data.y, node_label = node_label_graph)
        
        dataset.append(new_data)
        node_idx += num_nodes

    dataset_num_features = dataset[0].x.shape[1]
    # print(dataset[0].x)  # 새 데이터셋의 첫 번째 그래프 x 확인
    
    data_list = []
    label_list = []

    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())

    (train_index, test_index) = split
    data_train_ = [data_list[i] for i in train_index]
    data_test = [data_list[i] for i in test_index]
    
    print(f'len train_ {len(data_train_)}')
    
    data_train = []
    if dataset_AN:
        for data in data_train_:
            if data.y != 0:
                data_train.append(data) 
    else:
        for data in data_train_:
            if data.y == 0:
                data_train.append(data) 

    if dataset_AN:
        idx = 0
        for data in data_train:
            data.y = 0
            data['idx'] = idx
            idx += 1
    print(f'len train {len(data_train)}')
    
    if dataset_AN:
        for data in data_test:
            data.y = 1 if data.y == 0 else 0
    print(f'len test {len(data_test)}')
    
    max_nodes = max([dataset[i].num_nodes for i in range(len(dataset))])
    dataloader = DataLoader(data_train, batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, test_batch_size, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'num_test':len(data_test), 'num_edge_feat':0, 'max_nodes':max_nodes, 'max_node_label':num_unique_node_labels}
    loader_dict = {'train': dataloader, 'test': dataloader_test}

    return loader_dict, meta


def read_graph_file(dataset_name, path):
    prefix = os.path.join(path, dataset_name, 'raw', dataset_name)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    # 수정해줘야 함, 데이터 셋에 따라 라벨 맵핑 순서가 바뀜
    label_map_to_int = {0: 0, 1: 1}
    # label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        G = nx.from_edgelist(adj_list[i])
        G.graph['label'] = graph_labels[i - 1]
        for u in node_iter(G):
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                node_dict(G)[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                node_dict(G)[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        mapping = {}
        it = 0
        for n in node_iter(G):
            mapping[n] = it
            it += 1

        # graphs.append(nx.relabel_nodes(G, mapping))
        graphs.append(from_networkx(nx.relabel_nodes(G, mapping)))
                
    return graphs

# def get_ad_dataset_Tox21(dataset_name, batch_size, test_batch_size, need_str_enc=False):
#     set_seed(1)
#     path = osp.join(osp.dirname(osp.dirname(osp.realpath(__file__))), 'dataset/data')
    
#     data_train_ = read_graph_file(dataset_name + '_training', path)
#     data_test = read_graph_file(dataset_name + '_testing', path)
    
#     dataset_num_features = data_train_[0].label.shape[1]
    
#     data_train = [] # 0이 정상, 1이 이상
#     for data in data_train_:
#         if getattr(data, 'graph_label', None) == 0:  # 'graph_label'이 1인 데이터 확인
#             data.y = data.graph_label  # 'graph_label'을 'y'로 변경
#             data.x = data.label  # 'label'을 'x'로 변경
#             data.x = data.x.float()  
#             del data.graph_label  # 기존 'graph_label' 삭제
#             del data.label  # 기존 'label' 삭제
#             data_train.append(data)

#     for data in data_test:
#         data.y = data.graph_label  # 'graph_label'을 'y'로 변경
#         data.x = data.label  # 'label'을 'x'로 변경
#         data.x = data.x.float()  
#         del data.graph_label  # 기존 'graph_label' 삭제
#         del data.label  # 기존 'label' 삭제
    
#     # a= []
#     # for i in range(len(data_test)):
#     #     if data_test[i].y == 0:
#     #         a.append(data_test[i].y)
#     # len(a)
    
#     data_ = data_train + data_test
#     max_nodes = max([data_[i].num_nodes for i in range(len(data_))])
    
#     dataloader = DataLoader(data_train, batch_size, shuffle=True)
#     dataloader_test = DataLoader(data_test, test_batch_size, shuffle=True)
#     meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'max_nodes': max_nodes}
#     loader_dict = {'train': dataloader, 'test': dataloader_test}
    
#     return loader_dict, meta



def get_ad_dataset_Tox21_(dataset_name, batch_size, test_batch_size, need_str_enc=False):
    set_seed(1)
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')

    data_train_ = read_graph_file(dataset_name + '_training', path)
    
    dataset_num_features = data_train_[0].label.shape[1]
    
    data_train = [] # 0이 정상, 1이 이상
    data_test = []
    for data in data_train_:
        if getattr(data, 'graph_label', None) == 0:  # 'graph_label'이 1인 데이터 확인
            data.y = data.graph_label  # 'graph_label'을 'y'로 변경
            data.x = data.label  # 'label'을 'x'로 변경
            data.x = data.x.float()  
            del data.graph_label  # 기존 'graph_label' 삭제
            del data.label  # 기존 'label' 삭제
            data_train.append(data)
        elif getattr(data, 'graph_label', None) == 1:
            data.y = data.graph_label  # 'graph_label'을 'y'로 변경
            data.x = data.label  # 'label'을 'x'로 변경
            data.x = data.x.float()  
            del data.graph_label  # 기존 'graph_label' 삭제
            del data.label  # 기존 'label' 삭제
            data_test.append(data)
    
    print(len(data_train))            
    print(len(data_test))
    
    if dataset_name == 'Tox21_p53':
        random_indices_test = random.sample(range(len(data_test)), 28)
    elif dataset_name == 'Tox21_HSE':
        random_indices_test = random.sample(range(len(data_test)), 10)
    elif dataset_name == 'Tox21_MMP':
        random_indices_test = random.sample(range(len(data_test)), 38)

    # 선택된 인덱스에 해당하는 데이터만 남기기
    data_test = [data_test[i] for i in random_indices_test]

    # 결과 확인
    print(f"data_test에 남아 있는 데이터 개수: {len(data_test)}")
    
    if dataset_name == 'Tox21_p53':
        random_indices = random.sample(range(len(data_train)), 241)
    elif dataset_name == 'Tox21_HSE':
        random_indices = random.sample(range(len(data_train)), 257)
    elif dataset_name == 'Tox21_MMP':
        random_indices = random.sample(range(len(data_train)), 200)
        
    # 선택된 인덱스에 해당하는 데이터 가져오기
    random_sampled_data = [data_train[i] for i in random_indices]

    # data_test에 선택된 100개 데이터 추가
    data_test.extend(random_sampled_data)

    data_train = [data for i, data in enumerate(data_train) if i not in random_indices]

    # 결과 확인
    print(f"data_test에 추가된 데이터 개수: {len(random_sampled_data)}")
    print(f"data_train에서 삭제 후 데이터 개수: {len(data_train)}")

    # for data in data_test:
    #     data.y = data.graph_label  # 'graph_label'을 'y'로 변경
    #     data.x = data.label  # 'label'을 'x'로 변경
    #     data.x = data.x.float()  
    #     del data.graph_label  # 기존 'graph_label' 삭제
    #     del data.label  # 기존 'label' 삭제
    
    # a= []
    # for i in range(len(data_test)):
    #     if data_test[i].y == 0:
    #         a.append(data_test[i].y)
    # len(a)
    data_ = data_train + data_test
    max_nodes = max([data_[i].num_nodes for i in range(len(data_))])
    
    dataloader = DataLoader(data_train, batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'max_nodes': max_nodes}
    loader_dict = {'train': dataloader, 'test': dataloader_test}
    
    return loader_dict, meta


def read_graph_file_(dataset_name, path):
    prefix = os.path.join(path, dataset_name, 'raw', dataset_name)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    graph_indic = {}
    with open(filename_graph_indic) as f:
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    # 수정해줘야 함, 데이터 셋에 따라 다름
    label_map_to_int = {0: 0, 1: 1}
    # label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        G = nx.from_edgelist(adj_list[i])
        G.graph['label'] = graph_labels[i - 1]
        for u in node_iter(G):
            if len(node_labels) > 0:
                node_label_one_hot = [0] * num_unique_node_labels
                node_label = node_labels[u - 1]
                node_label_one_hot[node_label] = 1
                node_dict(G)[u]['label'] = node_label_one_hot
            if len(node_attrs) > 0:
                node_dict(G)[u]['feat'] = node_attrs[u - 1]
        if len(node_attrs) > 0:
            G.graph['feat_dim'] = node_attrs[0].shape[0]

        mapping = {}
        it = 0
        for n in node_iter(G):
            mapping[n] = it
            it += 1

        # graphs.append(nx.relabel_nodes(G, mapping))
        graphs.append(from_networkx(nx.relabel_nodes(G, mapping)))
                
    return graphs


def get_ad_dataset_Tox21(dataset_name, batch_size, test_batch_size, need_str_enc=False):
    set_seed(1)
    path = osp.join(osp.dirname(osp.realpath(__file__)), 'dataset')

    data_train_ = read_graph_file_(dataset_name + '_training', path)
    
    dataset_num_features = data_train_[0].label.shape[1]
    
    data_train = [] # 0이 정상, 1이 이상
    data_test = []
    for data in data_train_:
        if getattr(data, 'graph_label', None) == 0:  # 'graph_label'이 1인 데이터 확인
            data.y = data.graph_label  # 'graph_label'을 'y'로 변경
            data.x = data.label  # 'label'을 'x'로 변경
            data.x = data.x.float()  
            del data.graph_label  # 기존 'graph_label' 삭제
            del data.label  # 기존 'label' 삭제
            data_train.append(data)
        elif getattr(data, 'graph_label', None) == 1:
            data.y = data.graph_label  # 'graph_label'을 'y'로 변경
            data.x = data.label  # 'label'을 'x'로 변경
            data.x = data.x.float()  
            del data.graph_label  # 기존 'graph_label' 삭제
            del data.label  # 기존 'label' 삭제
            data_test.append(data)
    
    print(len(data_train))            
    print(len(data_test))
    
    if dataset_name == 'Tox21_p53':
        random_indices_test = random.sample(range(len(data_test)), 28)
    elif dataset_name == 'Tox21_HSE':
        random_indices_test = random.sample(range(len(data_test)), 10)
    elif dataset_name == 'Tox21_MMP':
        random_indices_test = random.sample(range(len(data_test)), 38)

    # 선택된 인덱스에 해당하는 데이터만 남기기
    data_test = [data_test[i] for i in random_indices_test]

    # 결과 확인
    print(f"data_test에 남아 있는 데이터 개수: {len(data_test)}")
    
    if dataset_name == 'Tox21_p53':
        random_indices = random.sample(range(len(data_train)), 241)
    elif dataset_name == 'Tox21_HSE':
        random_indices = random.sample(range(len(data_train)), 257)
    elif dataset_name == 'Tox21_MMP':
        random_indices = random.sample(range(len(data_train)), 200)
        
    # 선택된 인덱스에 해당하는 데이터 가져오기
    random_sampled_data = [data_train[i] for i in random_indices]

    # data_test에 선택된 100개 데이터 추가
    data_test.extend(random_sampled_data)

    data_train = [data for i, data in enumerate(data_train) if i not in random_indices]

    # 결과 확인
    print(f"data_test에 추가된 데이터 개수: {len(random_sampled_data)}")
    print(f"data_train에서 삭제 후 데이터 개수: {len(data_train)}")

    print(data_test[10])
    print(data_test[50])
    print(data_test[100])
    print(data_test[150])
    print(data_test[200])
    # for data in data_test:
    #     data.y = data.graph_label  # 'graph_label'을 'y'로 변경
    #     data.x = data.label  # 'label'을 'x'로 변경
    #     data.x = data.x.float()  
    #     del data.graph_label  # 기존 'graph_label' 삭제
    #     del data.label  # 기존 'label' 삭제
    
    # a= []
    # for i in range(len(data_test)):
    #     if data_test[i].y == 0:
    #         a.append(data_test[i].y)
    # len(a)
    data_ = data_train + data_test
    max_nodes = max([data_[i].num_nodes for i in range(len(data_))])
    
    dataloader = DataLoader(data_train, batch_size, shuffle=True, num_workers=1) # num_workers=4 -> 4개 병렬작업, num_workers=1 -> 단일 작업(시간은 오래 걸리나 안정적일수도)
    dataloader_test = DataLoader(data_test, test_batch_size, shuffle=True)
    # dataloader_test = DataLoader(data_test, batch_size, shuffle=True)
    # meta = {'num_feat':dataset_num_features, 'num_feat_':dataset_num_features_ ,'num_train':len(data_train), 'max_nodes': max_nodes}
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'max_nodes': max_nodes}
    loader_dict = {'train': dataloader, 'test': dataloader_test}
    
    return loader_dict, meta


# def compute_persistence_and_betti(graph_distance_matrix, max_dimension=2):
#     try:
#         # Rips Complex 생성
#         rips_complex = gd.RipsComplex(distance_matrix=graph_distance_matrix, max_edge_length=2.0)
#         simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)
        
#         # Persistent Homology 계산
#         simplex_tree.compute_persistence()
        
#         # persistence diagram 가져오기
#         persistence_diagram = simplex_tree.persistence()
        
#         if persistence_diagram:
#             min_val = min(min(birth, death if death != float('inf') else birth) 
#                          for _, (birth, death) in persistence_diagram)
#             max_val = max(max(birth, death if death != float('inf') else birth) 
#                          for _, (birth, death) in persistence_diagram)
#         else:
#             min_val, max_val = 0.0, 2.0
        
#         # Betti Numbers 계산
#         betti_numbers = simplex_tree.persistent_betti_numbers(min_val, max_val)
        
#         return persistence_diagram, betti_numbers
#     except Exception as e:
#         print(f"Error in persistence computation: {str(e)}")
#         return [], [0, 0, 0]


# def compute_persistence_and_betti(graph_distance_matrix, dataset_name, max_dimension=2):
#     try:
#         # 입력 검증
#         if not isinstance(graph_distance_matrix, np.ndarray):
#             return [], [0, 0, 0]
        
#         if graph_distance_matrix.size == 0:
#             return [], [0, 0, 0]
            
#         # 메모리 제한을 위한 크기 체크
#         if graph_distance_matrix.shape[0] > 100:
#             # 큰 행렬은 샘플링
#             indices = np.random.choice(graph_distance_matrix.shape[0], 100, replace=False)
#             graph_distance_matrix = graph_distance_matrix[indices][:, indices]
        
#         # GUDHI 계산
#         if dataset_name == 'AIDS':
#             max_dimension = 1
#         else:
#             max_dimension = 2
#         rips = gd.RipsComplex(distance_matrix=graph_distance_matrix)
#         st = rips.create_simplex_tree(max_dimension=max_dimension)  # 차원 제한
#         st.compute_persistence()
        
#         # 결과 추출
#         persistence = st.persistence()
#         betti = st.persistent_betti_numbers(0, 2.0)
        
#         # 메모리 정리
#         del rips
#         del st
#         gc.collect()
        
#         return persistence, betti
        
#     except Exception as e:
#         print(f"Error in safe_compute_persistence_and_betti: {str(e)}")
#         return [], [0, 0, 0]


# def merge_persistence_results(persistence_results):
#     # 모든 persistence diagram 병합
#     merged = []
#     for diagram in persistence_results:
#         merged.extend(diagram)
    
#     # 중복 제거 및 정렬
#     merged = list(set(merged))
#     merged.sort(key=lambda x: (x[1][0], x[1][1]))
    
#     return merged

# def compute_merged_betti(persistence_results):
#     # 각 차원별 Betti 수 계산
#     betti = [0, 0, 0]
#     for diagram in persistence_results:
#         for dim, (birth, death) in diagram:
#             if dim < len(betti) and death != float('inf'):
#                 betti[dim] += 1
    
#     return betti


# def process_batch_graphs(data):
#     graphs = split_batch_graphs(data)
#     true_stats_list = []
    
#     print(f"\nProcessing {len(graphs)} graphs...")
    
#     for i, (x, edge_index) in enumerate(graphs):
#         try:
#             # 거리 행렬 계산
#             distance_matrix = squareform(pdist(x.cpu().numpy(), metric='euclidean'))
            
#             # Persistent Homology 계산
#             persistence_diagram, betti_numbers = compute_persistence_and_betti(distance_matrix)
            
#             # 통계 추출
#             stats = {
#                 "mean_survival": np.mean([death - birth for _, (birth, death) in persistence_diagram 
#                                         if death != float('inf')]) if persistence_diagram else 0.0,
#                 "max_survival": np.max([death - birth for _, (birth, death) in persistence_diagram 
#                                       if death != float('inf')]) if persistence_diagram else 0.0,
#                 "variance_survival": np.var([death - birth for _, (birth, death) in persistence_diagram 
#                                            if death != float('inf')]) if persistence_diagram else 0.0,
#                 "mean_birth": np.mean([birth for _, (birth, death) in persistence_diagram]) if persistence_diagram else 0.0,
#                 "mean_death": np.mean([death for _, (birth, death) in persistence_diagram 
#                                      if death != float('inf')]) if persistence_diagram else 0.0,
#                 "betti_0": betti_numbers[0] if len(betti_numbers) > 0 else 0,
#                 "betti_1": betti_numbers[1] if len(betti_numbers) > 1 else 0,
#                 "betti_2": betti_numbers[2] if len(betti_numbers) > 2 else 0
#             }
            
#             true_stats_list.append(stats)
            
#             if i % 50 == 0:  # 50개 그래프마다 진행상황 출력
#                 print(f"Processed {i}/{len(graphs)} graphs")
                
#         except Exception as e:
#             print(f"Error processing graph {i}: {str(e)}")
#             true_stats_list.append({
#                 "mean_survival": 0.0, "max_survival": 0.0, "variance_survival": 0.0,
#                 "mean_birth": 0.0, "mean_death": 0.0,
#                 "betti_0": 0, "betti_1": 0, "betti_2": 0
#             })
    
#     # 모든 통계를 tensor로 변환
#     true_stats_tensor = torch.tensor([
#         [stats['mean_survival'], stats['max_survival'], stats['variance_survival'],
#          stats['mean_birth'], stats['mean_death'],
#          stats['betti_0'], stats['betti_1'], stats['betti_2']]
#         for stats in true_stats_list
#     ], dtype=torch.float32)
    
#     # 데이터에 통계 추가
#     data.true_stats = true_stats_tensor
    
#     print("\nProcessing completed!")
#     print(f"Final statistics shape: {data.true_stats.shape}")
    
#     # 처음 몇 개 그래프의 통계 출력
#     print("\nFirst few graphs statistics:")
#     for i in range(min(3, len(true_stats_list))):
#         print(f"\nGraph {i}:")
#         print(f"Betti numbers: β₀={true_stats_list[i]['betti_0']}, "
#               f"β₁={true_stats_list[i]['betti_1']}, β₂={true_stats_list[i]['betti_2']}")
#         print(f"Mean survival: {true_stats_list[i]['mean_survival']:.4f}")
#         print(f"Max survival: {true_stats_list[i]['max_survival']:.4f}")
    
#     return data


# def process_batch_graphs(data, dataset_name):
#     """최소한의 기능으로 구현한 배치 처리"""
#     try:
#         graphs = split_batch_graphs(data)
#         true_stats_list = []
        
#         for i, (x, edge_index) in enumerate(graphs):
#             try:
#                 # CPU로 이동하고 numpy로 변환
#                 x_np = x.cpu().detach().numpy()
                
#                 # 작은 그래프만 처리
#                 if x_np.shape[0] <= 100:
#                     distance_matrix = squareform(pdist(x_np))
#                     persistence_diagram, betti_numbers = compute_persistence_and_betti(distance_matrix, dataset_name)
#                 else:
#                     # 큰 그래프는 기본값 사용
#                     persistence_diagram, betti_numbers = [], [0, 0, 0]
                
#                 # 통계 계산
#                 stats = {
#                     "mean_survival": 0.0,
#                     "max_survival": 0.0,
#                     "variance_survival": 0.0,
#                     "mean_birth": 0.0,
#                     "mean_death": 0.0,
#                     "betti_0": betti_numbers[0] if len(betti_numbers) > 0 else 0,
#                     "betti_1": betti_numbers[1] if len(betti_numbers) > 1 else 0,
#                     "betti_2": betti_numbers[2] if len(betti_numbers) > 2 else 0
#                 }
                
#                 if persistence_diagram:
#                     survivals = [death - birth for _, (birth, death) in persistence_diagram 
#                                if death != float('inf')]
#                     if survivals:
#                         stats["mean_survival"] = float(np.mean(survivals))
#                         stats["max_survival"] = float(np.max(survivals))
#                         stats["variance_survival"] = float(np.var(survivals))
                
#                 true_stats_list.append(stats)
                
#             except Exception as e:
#                 print(f"Error processing graph {i}: {str(e)}")
#                 true_stats_list.append(get_default_stats())
            
#             # 각 그래프 처리 후 메모리 정리
#             gc.collect()
        
#         # 결과를 텐서로 변환
#         true_stats_tensor = torch.tensor([
#             [stats['mean_survival'], stats['max_survival'], stats['variance_survival'],
#              stats['mean_birth'], stats['mean_death'],
#              stats['betti_0'], stats['betti_1'], stats['betti_2']]
#             for stats in true_stats_list
#         ], dtype=torch.float32)
        
#         data.true_stats = true_stats_tensor
#         return data
        
#     except Exception as e:
#         print(f"Error in process_batch_graphs: {str(e)}")
#         # 오류 발생시 기본값으로 채운 텐서 반환
#         data.true_stats = torch.zeros((len(graphs), 8), dtype=torch.float32)
#         return data


# def calculate_persistence_stats(persistence_diagram, betti_numbers):
#     """통계 계산을 위한 헬퍼 함수"""
#     return {
#         "mean_survival": np.mean([death - birth for _, (birth, death) in persistence_diagram 
#                                 if death != float('inf')]) if persistence_diagram else 0.0,
#         "max_survival": np.max([death - birth for _, (birth, death) in persistence_diagram 
#                               if death != float('inf')]) if persistence_diagram else 0.0,
#         "variance_survival": np.var([death - birth for _, (birth, death) in persistence_diagram 
#                                    if death != float('inf')]) if persistence_diagram else 0.0,
#         "mean_birth": np.mean([birth for _, (birth, death) in persistence_diagram]) if persistence_diagram else 0.0,
#         "mean_death": np.mean([death for _, (birth, death) in persistence_diagram 
#                              if death != float('inf')]) if persistence_diagram else 0.0,
#         "betti_0": betti_numbers[0] if len(betti_numbers) > 0 else 0,
#         "betti_1": betti_numbers[1] if len(betti_numbers) > 1 else 0,
#         "betti_2": betti_numbers[2] if len(betti_numbers) > 2 else 0
#     }


# def get_default_stats():
#     """기본 통계값 반환"""
#     return {
#         "mean_survival": 0.0, "max_survival": 0.0, "variance_survival": 0.0,
#         "mean_birth": 0.0, "mean_death": 0.0,
#         "betti_0": 0, "betti_1": 0, "betti_2": 0
#     }


# def print_statistics_summary(true_stats_list):
#     """통계 요약 출력"""
#     print("\nProcessing completed!")
#     print(f"Final statistics shape: {len(true_stats_list)}")
    
#     print("\nFirst few graphs statistics:")
#     for i in range(min(3, len(true_stats_list))):
#         print(f"\nGraph {i}:")
#         print(f"Betti numbers: β₀={true_stats_list[i]['betti_0']}, "
#               f"β₁={true_stats_list[i]['betti_1']}, β₂={true_stats_list[i]['betti_2']}")
#         print(f"Mean survival: {true_stats_list[i]['mean_survival']:.4f}")
#         print(f"Max survival: {true_stats_list[i]['max_survival']:.4f}")


# def merge_persistence_diagrams(persistence_diagrams):
#     """여러 persistence diagram을 하나로 병합"""
#     merged = []
#     for diagram in persistence_diagrams:
#         if diagram:  # 빈 다이어그램이 아닌 경우에만 처리
#             merged.extend(diagram)
    
#     if not merged:  # 모든 다이어그램이 비어있는 경우
#         return []
    
#     # 중복 제거 및 정렬
#     # 튜플의 리스트를 집합으로 변환할 수 없으므로, 비교 가능한 형태로 변환
#     unique_pairs = set((dim, birth, death) for dim, (birth, death) in merged)
    
#     # 다시 원래 형식으로 변환하고 정렬
#     merged = [(dim, (birth, death)) for dim, birth, death in unique_pairs]
#     merged.sort(key=lambda x: (x[0], x[1][0], x[1][1]))  # 차원, birth, death 순으로 정렬
    
#     return merged

# def merge_betti_numbers(betti_numbers_list):
#     """여러 Betti number 리스트를 하나로 병합"""
#     if not betti_numbers_list:
#         return [0, 0, 0]
    
#     # 각 차원별로 최대값 선택
#     max_betti = []
#     for dim in range(3):  # 0, 1, 2 차원에 대해
#         max_val = max(betti[dim] if len(betti) > dim else 0 
#                      for betti in betti_numbers_list)
#         max_betti.append(max_val)
    
#     return max_betti


# def scott_rule_bandwidth(X):
#     """
#     Scott의 규칙을 사용하여 KDE의 최적 bandwidth를 계산합니다.
    
#     Parameters:
#     -----------
#     X : array-like of shape (n_samples, n_features)
#         입력 데이터
    
#     Returns:
#     --------
#     bandwidth : float
#         Scott의 규칙으로 계산된 optimal bandwidth
#     """
#     n = len(X)
#     d = X.shape[1]  # 특징 차원
    
#     # 각 차원별 표준편차 계산
#     sigma = np.std(X, axis=0)
    
#     # Scott의 규칙: h = n^(-1/(d+4)) * sigma
#     bandwidth = np.power(n, -1./(d+4)) * sigma
    
#     # 다변량의 경우 기하평균 사용
#     if d > 1:
#         bandwidth = np.prod(bandwidth) ** (1./d)
        
#     return bandwidth

# def loocv_bandwidth_selection(X, bandwidths=None, cv=None):
#     """
#     Leave-one-out 교차 검증을 사용하여 최적의 bandwidth를 선택합니다.
    
#     Parameters:
#     -----------
#     X : array-like of shape (n_samples, n_features)
#         입력 데이터
#     bandwidths : array-like, optional
#         테스트할 bandwidth 값들. None이면 자동으로 범위 생성
#     cv : int, cross-validation generator or iterable, optional
#         교차 검증 분할기. None이면 LeaveOneOut 사용
    
#     Returns:
#     --------
#     optimal_bandwidth : float
#         LOOCV로 선택된 optimal bandwidth
#     cv_scores : dict
#         각 bandwidth에 대한 교차 검증 점수
#     """
#     if bandwidths is None:
#         # Scott의 규칙으로 초기 추정치를 구하고 그 주변 값들을 테스트
#         scott_bw = scott_rule_bandwidth(X)
#         bandwidths = np.logspace(np.log10(scott_bw/5), np.log10(scott_bw*5), 20)
    
#     if cv is None:
#         cv = LeaveOneOut()
    
#     n_samples = X.shape[0]
#     cv_scores = {bw: 0.0 for bw in bandwidths}
    
#     for train_idx, test_idx in cv.split(X):
#         X_train = X[train_idx]
#         X_test = X[test_idx]
        
#         for bw in bandwidths:
#             # 현재 bandwidth로 KDE 학습
#             kde = KernelDensity(bandwidth=bw, kernel='gaussian')
#             kde.fit(X_train)
            
#             # 테스트 샘플의 log-likelihood 계산
#             log_likelihood = kde.score(X_test)
#             cv_scores[bw] += log_likelihood
    
#     # 평균 log-likelihood가 가장 높은 bandwidth 선택
#     optimal_bandwidth = max(cv_scores.items(), key=lambda x: x[1])[0]
    
#     return optimal_bandwidth, cv_scores


def split_batch_graphs(data):
    graphs = []
    # ptr을 사용하여 각 그래프의 경계를 찾음
    for i in range(len(data.ptr) - 1):
        start, end = data.ptr[i].item(), data.ptr[i + 1].item()
        # 해당 그래프의 노드 특성
        x = data.x[start:end]
        # 해당 그래프의 엣지 인덱스 추출 및 조정
        mask = (data.edge_index[0] >= start) & (data.edge_index[1] < end)
        edge_index = data.edge_index[:, mask]
        edge_index = edge_index - start  # 노드 인덱스 조정
        graphs.append((x, edge_index))
    return graphs


def compute_persistence(graph_distance_matrix):
    """Persistence diagram만 계산하는 최적화된 함수"""
    try:
        if not isinstance(graph_distance_matrix, np.ndarray) or graph_distance_matrix.size == 0:
            return []
            
        # 더 효율적인 샘플링
        if graph_distance_matrix.shape[0] > 100:
            # 균일한 간격으로 샘플링하여 더 대표성 있는 부분집합 선택
            step = graph_distance_matrix.shape[0] // 100
            indices = np.arange(0, graph_distance_matrix.shape[0], step)[:100]
            graph_distance_matrix = graph_distance_matrix[indices][:, indices]
        
        # 최소한의 차원과 계산만 수행
        rips = gd.RipsComplex(distance_matrix=graph_distance_matrix)
        st = rips.create_simplex_tree(max_dimension=1)  # 1차원까지만 계산
        st.persistence()  # 결과 저장 없이 바로 반환
        persistence = [(dim, (birth, death)) for dim, (birth, death) in st.persistence() 
                      if death != float('inf')]  # 무한대 값 필터링
        
        del rips, st
        return persistence
        
    except Exception as e:
        print(f"Error in persistence computation: {str(e)}")
        return []
    

def process_batch_graphs(data):
    try:
        graphs = split_batch_graphs(data)
        stats_list = []
        
        for i, (x, _) in enumerate(graphs):
            try:
                x_np = x.cpu().detach().numpy()
                distance_matrix = squareform(pdist(x_np))
                persistence = compute_persistence(distance_matrix)
                
                if not persistence:
                    stats_list.append(np.zeros(5, dtype=np.float32))
                    continue
                
                # 한 번의 순회로 모든 통계 계산
                births, deaths = zip(*[birth_death for _, birth_death in persistence])
                survivals = np.array([d - b for b, d in zip(births, deaths)])
                
                stats = np.array([
                    np.mean(survivals),
                    np.max(survivals),
                    np.var(survivals) if len(survivals) > 1 else 0,
                    np.mean(births),
                    np.mean(deaths)
                ], dtype=np.float32)
                
                stats_list.append(stats)
                
            except Exception as e:
                print(f"Error processing graph {i}: {str(e)}")
                stats_list.append(np.zeros(5, dtype=np.float32))
            
            # 더 효율적인 메모리 관리
            if i % 50 == 0:
                gc.collect()
        
        # 한 번에 텐서로 변환
        data.true_stats = torch.tensor(stats_list, dtype=torch.float32)
        return data
        
    except Exception as e:
        print(f"Error in process_batch_graphs: {str(e)}")
        data.true_stats = torch.zeros((len(graphs), 5), dtype=torch.float32)
        return data
    
    
def scott_rule_bandwidth(X):
    n = len(X)
    d = X.shape[1]  # 특징 차원
    sigma = np.std(X, axis=0)  # 각 차원별 표준편차
    bandwidth = np.power(n, -1./(d+4)) * sigma
    if d > 1:
        bandwidth = np.prod(bandwidth) ** (1./d)  # 다변량의 경우 기하평균 사용
    return bandwidth

def loocv_bandwidth_selection(X, bandwidths=None, cv=None, range_factor=5):
    if bandwidths is None:
        scott_bw = scott_rule_bandwidth(X)
        bandwidths = np.logspace(np.log10(scott_bw/range_factor), 
                                 np.log10(scott_bw*range_factor), 20)
    
    if cv is None:
        cv = LeaveOneOut()
    
    n_samples = X.shape[0]
    cv_scores = {bw: 0.0 for bw in bandwidths}
    
    for train_idx, test_idx in cv.split(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
        
        for bw in bandwidths:
            kde = KernelDensity(bandwidth=bw, kernel='gaussian')
            kde.fit(X_train)
            log_likelihood = kde.score_samples(X_test)[0]  # 단일 샘플 점수
            cv_scores[bw] += log_likelihood / n_samples  # 평균 점수로 누적
    
    optimal_bandwidth = max(cv_scores.items(), key=lambda x: x[1])[0]
    return optimal_bandwidth, cv_scores

