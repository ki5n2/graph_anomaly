import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool

from module.utils import add_gaussian_perturbation, randint_exclude, extract_subgraph, batch_nodes_subgraphs, adj_original, adj_recon


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)
        # self.conv2 = GCNConv(out_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()
        
        self.reset_parameters()

    def reset_parameters(self):
        # GCNConv 층에 대한 특별한 초기화
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.conv1.lin.weight, gain=gain)
        # nn.init.xavier_uniform_(self.conv2.lin.weight, gain=gain)
        nn.init.zeros_(self.conv1.bias)
        # nn.init.zeros_(self.conv2.bias)

        # BatchNorm 층 초기화
        nn.init.constant_(self.bn1.weight, 1)
        nn.init.constant_(self.bn1.bias, 0)
        # nn.init.constant_(self.bn2.weight, 1)
        # nn.init.constant_(self.bn2.bias, 0)

        # Shortcut 층 초기화 (Linear인 경우)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.xavier_uniform_(self.shortcut.weight, gain=1.0)
            nn.init.zeros_(self.shortcut.bias)

    def forward(self, x, edge_index):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x, edge_index)))
        x = self.dropout(x)
        # x = self.bn2(self.conv2(x, edge_index))
        # x = self.dropout(x)
        return F.relu(x + residual)
    
    
class Feature_Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dims, num_features):
        super(Feature_Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList()
        current_dim = embed_dim
        for hidden_dim in reversed(hidden_dims[:-1]):
            self.decoder_layers.append(nn.Linear(current_dim, hidden_dim))
            self.decoder_layers.append(nn.ReLU())
            self.decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            current_dim = hidden_dim
        self.decoder_layers.append(nn.Linear(current_dim, num_features))

    def forward(self, z):
        for layer in self.decoder_layers:
            z = layer(z)
        return z


class Adj_Decoder(nn.Module):
    def __init__(self, embed_dim):
        super(Adj_Decoder, self).__init__()
        self.W = nn.Parameter(torch.Tensor(embed_dim, embed_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, z, batch):
        # z: 노드 임베딩 (N x embed_dim)
        # batch: 각 노드가 속한 그래프를 나타내는 텐서 (N,)
        
        adj_recon_list = []
        
        # 가중치 행렬을 적용한 임베딩 계산
        weighted_z = torch.matmul(z, self.W)
        
        for batch_idx in torch.unique(batch):
            mask = (batch == batch_idx)
            z_graph = z[mask]
            weighted_z_graph = weighted_z[mask]
            
            # 개선된 인접 행렬 재구성
            adj_recon_graph = torch.sigmoid(torch.matmul(z_graph, weighted_z_graph.t()))
            adj_recon_list.append(adj_recon_graph)
        
        return adj_recon_list
    

class BilinearEdgeDecoder(nn.Module):
    def __init__(self, input_dim, max_nodes, threshold=0.5):
        super(BilinearEdgeDecoder, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.threshold = threshold
        self.max_nodes = max_nodes
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, z):
        actual_nodes = z.size(0)
        adj = torch.sigmoid(torch.mm(torch.mm(z, self.weight), z.t()))
        adj = adj * (1 - torch.eye(actual_nodes, device=z.device))
       
        # adj_binary = (adj > self.threshold).float()    
        padded_adj = torch.zeros(self.max_nodes, self.max_nodes, device=z.device)
        padded_adj[:actual_nodes, :actual_nodes] = adj
        
        return padded_adj
    
    
class GRAPH_AUTOENCODER_(torch.nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, dropout_rate=0.2):
        super(GRAPH_AUTOENCODER_, self).__init__()
        self.encoder_blocks = nn.ModuleList()        
        self.encoder_node_blocks = nn.ModuleList()        
        self.encoder_sub_blocks = nn.ModuleList()
        
        self.edge_decoder = BilinearEdgeDecoder(hidden_dims[-1], max_nodes, threshold=0.5)
        self.feature_decoder = Feature_Decoder(hidden_dims[-1], hidden_dims, num_features)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_blocks.append(ResidualBlock(current_dim, hidden_dim, dropout_rate))
            current_dim = hidden_dim
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_node_blocks.append(nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            current_dim = hidden_dim  
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_sub_blocks.append(ResidualBlock(current_dim, hidden_dim, dropout_rate))
            current_dim = hidden_dim
        

        # 가중치 초기화
        self.apply(self._init_weights)
    
                
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        aug_data = self.conservative_augment_molecular_graph(data)
        aug_x, aug_edge_index, aug_batch = aug_data.x, aug_data.edge_index, aug_data.batch
        
        # latent vector
        aug_z = self.encode(aug_x, aug_edge_index)
        aug_z = self.dropout(aug_z)
        aug_z_g = global_max_pool(aug_z, aug_batch)  # Aggregate features for classification
        
        # adjacency matrix
        adj = adj_original(edge_index, batch, self.max_nodes)
        
        # latent vector
        z = self.encode(x, edge_index)
        z = self.dropout(z)
        
        # perturbation
        z_prime = add_gaussian_perturbation(z)
        
        # adjacency matrix reconstruction
        adj_recon_list = []
        for i in range(data.num_graphs):
            mask = (batch == i)
            z_graph = z[mask]
            adj_recon = self.edge_decoder(z_graph)
            adj_recon_list.append(adj_recon)

        new_edge_index = self.get_edge_index_from_adj_list(adj_recon_list, batch)

        # node reconstruction
        x_recon = self.feature_decoder(z)
        
        # Graph classification
        z_g = global_max_pool(z, batch) # Aggregate features for classification
        z_prime_g = global_max_pool(z_prime, batch) # (batch_size, embedded size)
        
        z_g_mlp = self.projection_head(z_g)
        z_prime_g_mlp = self.projection_head(z_prime_g) # (batch_size, embedded size)
        
        z_tilde = self.encode(x_recon, new_edge_index)
        z_tilde_g = global_max_pool(z_tilde, batch)
        
        # subgraph
        batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features = batch_nodes_subgraphs(data)
        
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
        
        return adj, z, z_g, x_recon, adj_recon_list, z_tilde, z_tilde_g, pos_sub_z_g, neg_sub_z_g, z_g_mlp, z_prime_g_mlp, target_z, aug_z_g
    
    
    def get_edge_index_from_adj_list(self, adj_recon_list, batch, threshold=0.5):
        edge_index_list = []
        start_idx = 0
        for i, adj in enumerate(adj_recon_list):
            num_nodes = (batch == i).sum().item()
            adj_binary = (adj > threshold).float()  # 임계값 적용
            edge_index = adj_binary.nonzero().t()
            edge_index += start_idx  # 전체 그래프에서의 인덱스로 조정
            edge_index_list.append(edge_index)
            start_idx += num_nodes
        return torch.cat(edge_index_list, dim=1)
    
    def encode(self, x, edge_index):
        for block in self.encoder_blocks:
            x = block(x, edge_index)
            x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)

    def encode_node(self, x):
        for encoder in self.encoder_node_blocks[:-1]:
            x = self.act(encoder(x))
            bn_module = nn.BatchNorm1d(x.size()[1]).to('cuda')
            x = bn_module(x)
        x = self.encoder_node_blocks[-1](x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def encode_subgraph(self, x, edge_index):
        for block in self.encoder_sub_blocks:
            x = block(x, edge_index)
            x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)
    
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
        
        # 1. 노드 특성에 미세한 가우시안 노이즈 추가
        if graph.x is not None:
            noise = torch.randn_like(graph.x) * node_attr_noise_std
            augmented_graph.x = graph.x + noise
        
        # 2. 매우 낮은 확률로 일부 엣지 마스킹 (완전히 제거하지 않음)
        if random.random() < edge_mask_prob:
            edge_index = augmented_graph.edge_index
            num_edges = edge_index.size(1)
            mask = torch.rand(num_edges) > 0.1  # 10%의 엣지만 마스킹
            masked_edge_index = edge_index[:, mask]
            
            # 마스킹된 엣지의 정보를 별도로 저장
            augmented_graph.masked_edges = edge_index[:, ~mask]
            augmented_graph.edge_index = masked_edge_index
        
        return augmented_graph

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    #         if module.bias is not None:
    #             nn.init.constant_(module.bias, 0)
    #     elif isinstance(module, nn.BatchNorm1d):
    #         nn.init.constant_(module.weight, 1)
    #         nn.init.constant_(module.bias, 0)  
         
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


class GRAPH_AUTOENCODER(torch.nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, dropout_rate=0.1):
        super(GRAPH_AUTOENCODER, self).__init__()
        self.encoder_blocks = nn.ModuleList()        
        self.encoder_node_blocks = nn.ModuleList()        
        self.encoder_sub_blocks = nn.ModuleList()
        
        self.edge_decoder = BilinearEdgeDecoder(hidden_dims[-1], max_nodes, threshold=0.5)
        self.feature_decoder = Feature_Decoder(hidden_dims[-1], hidden_dims, num_features)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_blocks.append(ResidualBlock(current_dim, hidden_dim, dropout_rate))
            current_dim = hidden_dim
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_node_blocks.append(nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            current_dim = hidden_dim  
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_sub_blocks.append(ResidualBlock(current_dim, hidden_dim, dropout_rate))
            current_dim = hidden_dim

        # 가중치 초기화
        self.apply(self._init_weights)
    
                
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        aug_data = self.conservative_augment_molecular_graph(data)
        aug_x, aug_edge_index, aug_batch = aug_data.x, aug_data.edge_index, aug_data.batch
        
        # latent vector
        aug_z = self.encode(aug_x, aug_edge_index)
        aug_z = self.dropout(aug_z)
        aug_z_g = global_max_pool(aug_z, aug_batch)  # Aggregate features for classification
        
        # adjacency matrix
        adj = adj_original(edge_index, batch, self.max_nodes)
            
        # latent vector
        z = self.encode(x, edge_index)
        z = self.dropout(z)
        
        # perturbation
        z_prime = add_gaussian_perturbation(z)
        
        # adjacency matrix reconstruction
        adj_recon_list = []
        for i in range(data.num_graphs):
            mask = (batch == i)
            z_graph = z[mask]
            adj_recon = self.edge_decoder(z_graph)
            adj_recon_list.append(adj_recon)

        new_edge_index = self.get_edge_index_from_adj_list(adj_recon_list, batch)
        
        # node reconstruction
        x_recon = self.feature_decoder(z)

        # Graph classification
        z_g = global_max_pool(z, batch)  # Aggregate features for classification
        z_prime_g = global_max_pool(z_prime, batch) # (batch_size, embedded size)
        
        z_g_mlp = self.projection_head(z_g)
        z_prime_g_mlp = self.projection_head(z_prime_g) # (batch_size, embedded size)
        
        z_tilde = self.encode(x_recon, new_edge_index)
        z_tilde_g = global_max_pool(z_tilde, batch)
        
        # # subgraph
        # batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features = batch_nodes_subgraphs(data)
        
        # pos_x, pos_edge_index, pos_batch = batched_pos_subgraphs.x, batched_pos_subgraphs.edge_index, batched_pos_subgraphs.batch
        # pos_sub_z, pos_new_edge_index = self.process_subgraphs(batched_pos_subgraphs)
        # pos_sub_z = torch.cat(pos_sub_z) # (number of nodes, embedded size)
        
        # unique_pos_batch, new_pos_batch = torch.unique(pos_batch, return_inverse=True)
        # pos_sub_z_g = global_mean_pool(pos_sub_z, new_pos_batch)
        
        # neg_x, neg_edge_index, neg_batch = batched_neg_subgraphs.x, batched_neg_subgraphs.edge_index, batched_neg_subgraphs.batch
        # neg_sub_z, neg_new_edge_index = self.process_subgraphs(batched_neg_subgraphs)
        # neg_sub_z = torch.cat(neg_sub_z)
        
        # unique_neg_batch, new_neg_batch = torch.unique(neg_batch, return_inverse=True)
        # neg_sub_z_g = global_mean_pool(neg_sub_z, new_neg_batch)
        
        # target_z = self.encode_node(batched_target_node_features) # (batch_size, feature_size)
        
        return adj, z, z_g, x_recon, adj_recon_list, z_tilde, z_tilde_g, aug_z_g
    
    
    def get_edge_index_from_adj_list(self, adj_recon_list, batch, threshold=0.5):
        edge_index_list = []
        start_idx = 0
        for i, adj in enumerate(adj_recon_list):
            num_nodes = (batch == i).sum().item()
            adj_binary = (adj > threshold).float()  # 임계값 적용
            edge_index = adj_binary.nonzero().t()
            edge_index += start_idx  # 전체 그래프에서의 인덱스로 조정
            edge_index_list.append(edge_index)
            start_idx += num_nodes
        return torch.cat(edge_index_list, dim=1)

    def encode(self, x, edge_index):
        for block in self.encoder_blocks:
            x = block(x, edge_index)
            x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)

    def encode_node(self, x):
        for encoder in self.encoder_node_blocks[:-1]:
            x = self.act(encoder(x))
            bn_module = nn.BatchNorm1d(x.size()[1]).to('cuda')
            x = bn_module(x)
        x = self.encoder_node_blocks[-1](x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def encode_subgraph(self, x, edge_index):
        for block in self.encoder_sub_blocks:
            x = block(x, edge_index)
            x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)
    
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
        
        # 1. 노드 특성에 미세한 가우시안 노이즈 추가
        if graph.x is not None:
            noise = torch.randn_like(graph.x) * node_attr_noise_std
            augmented_graph.x = graph.x + noise
        
        # 2. 매우 낮은 확률로 일부 엣지 마스킹 (완전히 제거하지 않음)
        if random.random() < edge_mask_prob:
            edge_index = augmented_graph.edge_index
            num_edges = edge_index.size(1)
            mask = torch.rand(num_edges) > 0.1  # 10%의 엣지만 마스킹
            masked_edge_index = edge_index[:, mask]
            
            # 마스킹된 엣지의 정보를 별도로 저장
            augmented_graph.masked_edges = edge_index[:, ~mask]
            augmented_graph.edge_index = masked_edge_index
        
        return augmented_graph

    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    #         if module.bias is not None:
    #             nn.init.constant_(module.bias, 0)
    #     elif isinstance(module, nn.BatchNorm1d):
    #         nn.init.constant_(module.weight, 1)
    #         nn.init.constant_(module.bias, 0)  
         
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


class GRAPH_AUTOENCODER2(torch.nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, dropout_rate=0.1):
        super(GRAPH_AUTOENCODER2, self).__init__()
        self.encoder_blocks = nn.ModuleList()        
        self.encoder_node_blocks = nn.ModuleList()        
        self.encoder_sub_blocks = nn.ModuleList()
        
        self.edge_decoder = BilinearEdgeDecoder(hidden_dims[-1], max_nodes, threshold=0.5)
        self.feature_decoder = Feature_Decoder(hidden_dims[-1], hidden_dims, num_features)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes

        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_blocks.append(ResidualBlock(current_dim, hidden_dim, dropout_rate))
            current_dim = hidden_dim
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_node_blocks.append(nn.Sequential(
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
            current_dim = hidden_dim  
        
        current_dim = num_features
        for hidden_dim in hidden_dims:
            self.encoder_sub_blocks.append(ResidualBlock(current_dim, hidden_dim, dropout_rate))
            current_dim = hidden_dim

        # 가중치 초기화
        self.apply(self._init_weights)
    
                
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        aug_data = self.conservative_augment_molecular_graph(data)
        aug_x, aug_edge_index, aug_batch = aug_data.x, aug_data.edge_index, aug_data.batch
            
        # latent vector
        aug_z = self.encode(aug_x, aug_edge_index)
        aug_z = self.dropout(aug_z)
        aug_z_g = global_max_pool(aug_z, aug_batch)  # Aggregate features for classification
        
        # adjacency matrix
        adj = adj_original(edge_index, batch, self.max_nodes)
            
        # latent vector
        z = self.encode(x, edge_index)
        z = self.dropout(z)
        
        # perturbation
        z_prime = add_gaussian_perturbation(z)
        
        # adjacency matrix reconstruction
        adj_recon_list = []
        for i in range(data.num_graphs):
            mask = (batch == i)
            z_graph = z[mask]
            adj_recon = self.edge_decoder(z_graph)
            adj_recon_list.append(adj_recon)

        new_edge_index = self.get_edge_index_from_adj_list(adj_recon_list, batch)
        
        # node reconstruction
        x_recon = self.feature_decoder(z)

        # Graph classification
        z_g = global_max_pool(z, batch) # Aggregate features for classification
        z_prime_g = global_max_pool(z_prime, batch) # (batch_size, embedded size)
        
        z_g_mlp = self.projection_head(z_g)
        z_prime_g_mlp = self.projection_head(z_prime_g) # (batch_size, embedded size)
        
        z_tilde = self.encode(x_recon, new_edge_index)
        z_tilde_g = global_max_pool(z_tilde, batch)
        
        # # subgraph
        # batched_pos_subgraphs, batched_neg_subgraphs, batched_target_node_features = batch_nodes_subgraphs(data)
        
        # pos_x, pos_edge_index, pos_batch = batched_pos_subgraphs.x, batched_pos_subgraphs.edge_index, batched_pos_subgraphs.batch
        # pos_sub_z, pos_new_edge_index = self.process_subgraphs(batched_pos_subgraphs)
        # pos_sub_z = torch.cat(pos_sub_z) # (number of nodes, embedded size)
        
        # unique_pos_batch, new_pos_batch = torch.unique(pos_batch, return_inverse=True)
        # pos_sub_z_g = global_mean_pool(pos_sub_z, new_pos_batch)
        
        # neg_x, neg_edge_index, neg_batch = batched_neg_subgraphs.x, batched_neg_subgraphs.edge_index, batched_neg_subgraphs.batch
        # neg_sub_z, neg_new_edge_index = self.process_subgraphs(batched_neg_subgraphs)
        # neg_sub_z = torch.cat(neg_sub_z)
        
        # unique_neg_batch, new_neg_batch = torch.unique(neg_batch, return_inverse=True)
        # neg_sub_z_g = global_mean_pool(neg_sub_z, new_neg_batch)
        
        # target_z = self.encode_node(batched_target_node_features) # (batch_size, feature_size)
        
        return adj, z, z_g, batch, x_recon, adj_recon_list, z_tilde, z_tilde_g, aug_z_g
    
    
    def get_edge_index_from_adj_list(self, adj_recon_list, batch, threshold=0.5):
        edge_index_list = []
        start_idx = 0
        for i, adj in enumerate(adj_recon_list):
            num_nodes = (batch == i).sum().item()
            adj_binary = (adj > threshold).float()  # 임계값 적용
            edge_index = adj_binary.nonzero().t()
            edge_index += start_idx  # 전체 그래프에서의 인덱스로 조정
            edge_index_list.append(edge_index)
            start_idx += num_nodes
        return torch.cat(edge_index_list, dim=1)

    def encode(self, x, edge_index):
        for block in self.encoder_blocks:
            x = block(x, edge_index)
            x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)

    def encode_node(self, x):
        for encoder in self.encoder_node_blocks[:-1]:
            x = self.act(encoder(x))
            bn_module = nn.BatchNorm1d(x.size()[1]).to('cuda')
            x = bn_module(x)
        x = self.encoder_node_blocks[-1](x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def encode_subgraph(self, x, edge_index):
        for block in self.encoder_sub_blocks:
            x = block(x, edge_index)
            x = self.dropout(x)
        return F.normalize(x, p=2, dim=1)
    
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
        
        # 1. 노드 특성에 미세한 가우시안 노이즈 추가
        if graph.x is not None:
            noise = torch.randn_like(graph.x) * node_attr_noise_std
            augmented_graph.x = graph.x + noise
        
        # 2. 매우 낮은 확률로 일부 엣지 마스킹 (완전히 제거하지 않음)
        if random.random() < edge_mask_prob:
            edge_index = augmented_graph.edge_index
            num_edges = edge_index.size(1)
            mask = torch.rand(num_edges) > 0.1  # 10%의 엣지만 마스킹
            masked_edge_index = edge_index[:, mask]
            
            # 마스킹된 엣지의 정보를 별도로 저장
            augmented_graph.masked_edges = edge_index[:, ~mask]
            augmented_graph.edge_index = masked_edge_index
        
        return augmented_graph

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)            
            
            