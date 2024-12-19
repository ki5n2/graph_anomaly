import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.utils as utils
from torch_geometric.nn import GCNConv
from torch_geometric.utils import get_laplacian
import networkx as nx
from scipy.linalg import eigh

class BatchUtils:
    @staticmethod
    def process_batch(x, edge_index, batch, num_graphs=None):
        batch_size = num_graphs if num_graphs is not None else batch.max().item() + 1
        z_list = [x[batch == i] for i in range(batch_size)]
        edge_index_list = []
        start_idx = 0
        
        for i in range(batch_size):
            num_nodes = z_list[i].size(0)
            graph_edges = edge_index[:, (edge_index[0] >= start_idx) & (edge_index[0] < start_idx + num_nodes)]
            graph_edges = graph_edges - start_idx
            edge_index_list.append(graph_edges)
            start_idx += num_nodes
            
        max_nodes_in_batch = max(z_graph.size(0) for z_graph in z_list)
        
        return z_list, edge_index_list, max_nodes_in_batch

    @staticmethod
    def add_cls_token(z_list, cls_token, max_nodes_in_batch, device):
        z_with_cls_list = []
        mask_list = []
        
        for z_graph in z_list:
            num_nodes = z_graph.size(0)
            cls_token = cls_token.to(device)
            z_graph = z_graph.unsqueeze(1)
            
            pad_size = max_nodes_in_batch - num_nodes
            z_graph_padded = F.pad(z_graph, (0, 0, 0, 0, 0, pad_size), 'constant', 0)
            
            z_with_cls = torch.cat([cls_token, z_graph_padded.transpose(0, 1)], dim=1)
            z_with_cls_list.append(z_with_cls)
            
            graph_mask = torch.cat([torch.tensor([False]), torch.tensor([False]*num_nodes + [True]*pad_size)])
            mask_list.append(graph_mask)
            
        z_with_cls_batch = torch.cat(z_with_cls_list, dim=0)
        mask = torch.stack(mask_list).to(device)
        
        return z_with_cls_batch, mask
    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.1, negative_slope=0.1):
        super(ResidualBlock, self).__init__()
        self.conv = GCNConv(in_channels, out_channels, improved=True, add_self_loops=True, normalize=True)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.shortcut = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.activation = nn.LeakyReLU(negative_slope=negative_slope)
        self.negative_slope = negative_slope
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu', self.negative_slope)
        nn.init.xavier_uniform_(self.conv.lin.weight, gain=gain)
        nn.init.zeros_(self.conv.bias)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        if isinstance(self.shortcut, nn.Linear):
            nn.init.xavier_uniform_(self.shortcut.weight, gain=1.0)
            nn.init.zeros_(self.shortcut.bias)

    def forward(self, x, edge_index):
        residual = self.shortcut(x)
        edge_index, _ = utils.add_self_loops(edge_index, num_nodes=x.size(0))
        deg = utils.degree(edge_index[0], x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        x = self.conv(x, edge_index, norm)
        x = self.activation(self.bn(x))
        x = self.dropout(x)
        return self.activation(x + residual)
    

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
    def __init__(self, embed_dim, num_features, dropout_rate=0.1):
        super(FeatureDecoder, self).__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim//2)
        self.fc2 = nn.Linear(embed_dim//2, num_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        z = self.leaky_relu(self.fc1(z))
        z = self.dropout(z)
        z = self.fc2(z)
        return z
    

class BilinearEdgeDecoder(nn.Module):
    def __init__(self, max_nodes):
        super(BilinearEdgeDecoder, self).__init__()
        self.max_nodes = max_nodes
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, z):
        actual_nodes = z.size(0)
        z_norm = F.normalize(z, p=2, dim=1)
        cos_sim = torch.mm(z_norm, z_norm.t())
        adj = self.sigmoid(cos_sim)
        adj = adj * (1 - torch.eye(actual_nodes, device=z.device))
        padded_adj = torch.zeros(self.max_nodes, self.max_nodes, device=z.device)
        padded_adj[:actual_nodes, :actual_nodes] = adj
        return padded_adj
    

class BertEncoder(nn.Module):
    def __init__(self, num_features, hidden_dims, d_model, nhead, num_layers, max_nodes, dropout_rate=0.1):
        super().__init__()
        self.gcn_encoder = Encoder(num_features, hidden_dims, dropout_rate)
        self.positional_encoding = GraphBertPositionalEncoding(hidden_dims[-1], max_nodes)
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_dims[-1], nhead, hidden_dims[-1] * 4, dropout_rate, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mask_token = nn.Parameter(torch.randn(1, hidden_dims[-1]))
        self.predicter = nn.Linear(hidden_dims[-1], num_features)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dims[-1]))
        self.edge_decoder = BilinearEdgeDecoder(max_nodes)
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = max_nodes
        self.d_model = d_model
        self.apply(self._init_weights)

    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, training=False, edge_training=False):
        h = self.gcn_encoder(x, edge_index)
        z_list, edge_index_list, max_nodes_in_batch = BatchUtils.process_batch(h, edge_index, batch)
        pos_encoded_list = []
        for i, (z_graph, edge_idx) in enumerate(zip(z_list, edge_index_list)):
            pos_encoding = self.positional_encoding(edge_idx, z_graph.size(0))
            z_graph_with_pos = z_graph + pos_encoding
            pos_encoded_list.append(z_graph_with_pos)
        
        z_with_cls_batch, padding_mask = BatchUtils.add_cls_token(
            pos_encoded_list, self.cls_token, max_nodes_in_batch, x.device
        )
        
        if training and mask_indices is not None:
            mask_positions = torch.zeros_like(padding_mask)
            start_idx = 0
            for i in range(len(z_list)):
                num_nodes = z_list[i].size(0)
                graph_mask_indices = mask_indices[start_idx:start_idx + num_nodes]
                mask_positions[i, 1:num_nodes+1] = graph_mask_indices
                node_indices = mask_positions[i].nonzero().squeeze(-1)
                z_with_cls_batch[i, node_indices] = self.mask_token
                padding_mask[i, num_nodes+1:] = True
                start_idx += num_nodes
        
        transformed = self.transformer(z_with_cls_batch, src_key_padding_mask=padding_mask)
        
        if edge_training == False:
            node_embeddings, masked_outputs = self._process_outputs(
                transformed, batch, mask_positions if training and mask_indices is not None else None
            )
        else:
            node_embeddings, _ = self._process_outputs(transformed, batch, mask_positions=None)
        
        if training and edge_training:
            adj_recon_list = []
            idx = 0
            for i in range(num_graphs):
                num_nodes = z_list[i].size(0)
                z_graph = node_embeddings[idx:idx + num_nodes]
                adj_recon = self.edge_decoder(z_graph)
                adj_recon_list.append(adj_recon)
                idx += num_nodes
            return node_embeddings, adj_recon_list
            
        if training:
            return node_embeddings, masked_outputs
            
        return node_embeddings

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            nn.init.normal_(module, mean=0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.zeros_(module.bias)

    def _process_outputs(self, transformed, batch, mask_positions=None):
        node_embeddings = []
        masked_outputs = []
        batch_size = transformed.size(0)
        start_idx = 0
        
        for i in range(batch_size):
            mask = (batch == i)
            num_nodes = mask.sum().item()
            graph_encoded = transformed[i, 1:num_nodes+1]
            node_embeddings.append(graph_encoded)
            
            all_predictions = self.predicter(graph_encoded)
            
            if mask_positions is not None:
                current_mask_positions = mask_positions[i, 1:num_nodes+1]
                if current_mask_positions.any():
                    masked_predictions = all_predictions[current_mask_positions]
                    masked_outputs.append(masked_predictions)
            
            start_idx += num_nodes
        
        node_embeddings = torch.cat(node_embeddings, dim=0)
        
        if mask_positions is not None and masked_outputs:
            return node_embeddings, torch.cat(masked_outputs, dim=0)
        return node_embeddings, None
    

class GraphBertPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_nodes):
        super().__init__()
        self.d_model = d_model
        self.max_nodes = max_nodes
        self.wsp_encoder = nn.Linear(max_nodes, d_model // 2)
        self.le_encoder = nn.Linear(max_nodes, d_model // 2)
    
    def get_wsp_encoding(self, edge_index, num_nodes):
        edge_index_np = edge_index.cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(zip(edge_index_np[0], edge_index_np[1]))
        
        spl_matrix = torch.zeros((num_nodes, self.max_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    try:
                        path_length = nx.shortest_path_length(G, source=i, target=j)
                    except nx.NetworkXNoPath:
                        path_length = self.max_nodes
                    if j < self.max_nodes:
                        spl_matrix[i, j] = path_length
                        
        return spl_matrix.to(edge_index.device)
    
    def get_laplacian_encoding(self, edge_index, num_nodes):
        edge_index, edge_weight = get_laplacian(edge_index, normalization='sym', num_nodes=num_nodes)
        L = torch.sparse_coo_tensor(edge_index, edge_weight, (num_nodes, num_nodes)).to_dense()
        
        L_np = L.cpu().numpy()
        _, eigenvecs = eigh(L_np)
        le_matrix = torch.from_numpy(eigenvecs).float().to(edge_index.device)
        
        padded_le = torch.zeros((num_nodes, self.max_nodes), device=edge_index.device)
        padded_le[:, :num_nodes] = le_matrix
        
        return padded_le
    
    def forward(self, edge_index, num_nodes):
        wsp_matrix = self.get_wsp_encoding(edge_index, num_nodes)
        wsp_encoding = self.wsp_encoder(wsp_matrix)
        le_matrix = self.get_laplacian_encoding(edge_index, num_nodes)
        le_encoding = self.le_encoder(le_matrix)
        
        return torch.cat([wsp_encoding, le_encoding], dim=-1)
    

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, max_nodes, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation='relu', batch_first = True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.d_model = d_model

    def forward(self, src, src_key_padding_mask):
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        return output
    

class GRAPH_AUTOENCODER(nn.Module):
    def __init__(self, num_features, hidden_dims, max_nodes, nhead_BERT, nhead, 
                 num_layers_BERT, num_layers, dropout_rate=0.1):
        super().__init__()
        self.encoder = BertEncoder(
            num_features=num_features,
            hidden_dims=hidden_dims,
            d_model=hidden_dims[-1],
            nhead=nhead_BERT,
            num_layers=num_layers_BERT,
            max_nodes=max_nodes,
            dropout_rate=dropout_rate
        )
        self.transformer_d = TransformerEncoder(
            d_model=hidden_dims[-1],
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=hidden_dims[-1] * 4,
            max_nodes=max_nodes,
            dropout=dropout_rate
        )
        self.feature_decoder = FeatureDecoder(hidden_dims[-1], num_features)
        self.u_mlp = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], hidden_dims[-1])
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dims[-1]))
        self.stats_predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], 5)
        )
        self.edge_recon = BilinearEdgeDecoder(max_nodes)

    def forward(self, x, edge_index, batch, num_graphs, mask_indices=None, is_pretrain=False, edge_training=False):
        if is_pretrain:
            if edge_training:
                node_embeddings, adj_recon_list = self.encoder(
                    x, edge_index, batch, num_graphs,
                    training=True,
                    edge_training=True
                )
                return node_embeddings, adj_recon_list
            else:
                node_embeddings, masked_outputs = self.encoder(
                    x, edge_index, batch, num_graphs,
                    mask_indices=mask_indices,
                    training=True,
                    edge_training=False
                )
                return node_embeddings, masked_outputs
        
        else:
            node_embeddings = self.encoder(
                x, edge_index, batch, num_graphs,
                training=False,
                edge_training=False
            )
            
            z_list, edge_index_list, max_nodes_in_batch = BatchUtils.process_batch(
                node_embeddings, edge_index, batch, num_graphs
            )
            
            z_with_cls_batch, mask = BatchUtils.add_cls_token(
                z_list, self.cls_token, max_nodes_in_batch, x.device
            )
            
            encoded = self.transformer_d(z_with_cls_batch, mask)
            
            cls_output = encoded[:, 0, :]  # [batch_size, hidden_dim]
            node_outputs = [encoded[i, 1:z_list[i].size(0)+1, :] for i in range(num_graphs)]
            u = torch.cat(node_outputs, dim=0)
            
            stats_pred = self.stats_predictor(cls_output)
            u_prime = self.u_mlp(u)
            x_recon = self.feature_decoder(u_prime)
            
            return x_recon, stats_pred
