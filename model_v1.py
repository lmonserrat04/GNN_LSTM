import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DGPool(nn.Module):
    def __init__(self, input_dim, pool_ratio):
        print("   ↳ Inicializando DGPool...")
        super(DGPool, self).__init__()
        self.pool_ratio = pool_ratio
        self.trainable_vector_pooling = nn.Parameter(torch.randn(input_dim, 1))

    def forward(self, lw_matrix_hidden_state_last):
        x = lw_matrix_hidden_state_last  # [N, F]
        num_nodes = x.size(0)
        k = max(1, int(num_nodes * self.pool_ratio))

        norm2 = torch.norm(self.trainable_vector_pooling)
        scores = x @ (self.trainable_vector_pooling / (norm2 + 1e-8))  # [N,1]

        scores = (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-8)
        sig_scores = torch.sigmoid(scores)  # [N,1]

        x_scaled = x * sig_scores

        _, indices = torch.topk(sig_scores.squeeze(), k=k)
        new_x = x_scaled[indices]

        sig_scores_sorted, _ = torch.sort(sig_scores.squeeze(), descending=True)
        topk_scores = sig_scores_sorted[:k]
        rest_scores = sig_scores_sorted[k:]

        eps = 1e-8
        pool_loss = -(
            torch.log(topk_scores + eps).sum() +
            torch.log(1.0 - rest_scores + eps).sum()
        ) / num_nodes

        return new_x, pool_loss


class GNN_LSTM(nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, pool_ratio=0.15, num_nodes=200):
        print("   ↳ Inicializando GNN_LSTM...")
        super().__init__()

        self.pool_ratio = pool_ratio
        self.hidden_channels = hidden_channels
        self.node_feat_dim = num_node_features

        # GCNConv para la entrada (G_t)
        self.input_gnn = GCNConv(num_node_features, hidden_channels)
        self.forget_gnn = GCNConv(num_node_features, hidden_channels)
        self.output_gnn = GCNConv(num_node_features, hidden_channels)
        self.modulation_gnn = GCNConv(num_node_features, hidden_channels)

        # GCN para el hidden state (H_{t-p})
        self.input_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels)
        self.forget_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels)
        self.output_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels)
        self.modulation_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels)

        # Normalización
        self.layer_norm = nn.LayerNorm(hidden_channels * 2)

        # Dynamic Graph Pooling (sin post_pool_fc — usa mean pooling)
        self.dg_pool = DGPool(hidden_channels, pool_ratio)

        # LSTM para datos raw fMRI
        self.lstm_raw_fmri = nn.LSTM(
            input_size=num_nodes,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=False
        )

        # MLP Clasificación
        self.mlp_layer_1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.mlp_layer_2 = nn.Linear(hidden_channels, 1)
        self.mlp_dropout = nn.Dropout(p=0.3)

    def gconv(self, gcn_layer, x, edge_index):
        return F.relu(gcn_layer(x, edge_index))

    def forward(self, lw_matrixes_sequence, edge_index, hidden_state, cell_state, time_series):
        if torch.isnan(hidden_state).any() or torch.isnan(cell_state).any():
            print("⚠️ NaN detectado en hidden o cell")

        for x in lw_matrixes_sequence:
            input_gate = torch.sigmoid(
                self.gconv(self.input_gnn, x, edge_index) +
                self.gconv(self.input_gnn_hidden_state, hidden_state, edge_index)
            )
            forget_gate = torch.sigmoid(
                self.gconv(self.forget_gnn, x, edge_index) +
                self.gconv(self.forget_gnn_hidden_state, hidden_state, edge_index)
            )
            output_gate = torch.sigmoid(
                self.gconv(self.output_gnn, x, edge_index) +
                self.gconv(self.output_gnn_hidden_state, hidden_state, edge_index)
            )
            modulation = torch.relu(
                self.gconv(self.modulation_gnn, x, edge_index) +
                self.gconv(self.modulation_gnn_hidden_state, hidden_state, edge_index)
            )

            cell_state = torch.tanh(input_gate * modulation + forget_gate * cell_state)
            hidden_state = output_gate * torch.tanh(cell_state)

        # DG-Pooling → mean sobre nodos seleccionados (sin post_pool_fc)
        pooled_graph, pool_loss = self.dg_pool(hidden_state)  # [k, hidden_channels]
        high_level_embeddings = pooled_graph.mean(dim=0)       # [hidden_channels]

        # LSTM raw fMRI
        low_level_embeddings = self.lstm_raw_time_series(time_series)  # [hidden_channels]

        # Fusión
        fusion = torch.cat([high_level_embeddings, low_level_embeddings], dim=0)  # [hidden_channels * 2]
        fusion = self.layer_norm(fusion.unsqueeze(0)).squeeze(0)

        # Clasificación
        pred = self.mlp_classiffier(fusion)

        return pred, pool_loss

    def lstm_raw_time_series(self, time_series_data):
        _, (h_last, _) = self.lstm_raw_fmri(time_series_data)
        h_last = h_last[-1].squeeze(0)
        return h_last

    def mlp_classiffier(self, concat_embedding):
        concat_embedding = F.relu(self.mlp_layer_1(concat_embedding))
        concat_embedding = self.mlp_dropout(concat_embedding)
        concat_embedding = self.mlp_layer_2(concat_embedding)
        return concat_embedding

    def compute_loss(self, prediction_batch, label_batch, pool_losses_batch, lambda_pool=0.1):
        loss_ce = F.binary_cross_entropy_with_logits(prediction_batch, label_batch)
        loss_pool = torch.mean(pool_losses_batch)
        return loss_ce + lambda_pool * loss_pool
