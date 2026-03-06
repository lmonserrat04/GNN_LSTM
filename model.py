import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from utils import get_edge_indexes_sparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, num_node_features, hidden_channels=128, pool_ratio=0.5, num_nodes=200):
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

        for gnn in [self.input_gnn, self.forget_gnn, self.output_gnn, self.modulation_gnn,
            self.input_gnn_hidden_state, self.forget_gnn_hidden_state,
            self.output_gnn_hidden_state, self.modulation_gnn_hidden_state]:
            nn.init.xavier_uniform_(gnn.lin.weight, gain=2.0)

        self.layer_norm = nn.LayerNorm(hidden_channels * 2)

        self.mod_norm = nn.LayerNorm(hidden_channels)

        # Dynamic Graph Pooling
        self.k = max(1, int(num_nodes * pool_ratio))
        self.dg_pool = DGPool(hidden_channels, pool_ratio)

        # LSTM para procesar datos raw
        self.lstm_raw_fmri = nn.LSTM(
            input_size=num_nodes,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=False
        )

        # MLP Clasificacion final — 3 capas: hidden*2 → hidden → hidden//2 → 1
        self.mlp_layer_1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.mlp_layer_2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.mlp_layer_3 = nn.Linear(hidden_channels // 2, 1)
        self.mlp_dropout = nn.Dropout(p=0.5)
        self.mlp_ln = nn.LayerNorm(hidden_channels)

    def gconv(self, gcn_layer, x, edge_index, edge_weight=None):
        """GCNConv con relu — para input, forget, output gates."""
        return F.relu(gcn_layer(x, edge_index, edge_weight=edge_weight))

    def gconv_linear(self, gcn_layer, x, edge_index, edge_weight=None):
        """GCNConv sin activación — para modulation, activación se aplica fuera."""
        return gcn_layer(x, edge_index, edge_weight=edge_weight)

    def forward(self, lw_matrixes_sequence, hidden_state, cell_state, time_series):

        if torch.isnan(hidden_state).any() or torch.isnan(cell_state).any():
            print("⚠️ NaN detectado en hidden o cell")

        for i, x in enumerate(lw_matrixes_sequence):
            
            x = x.double()
            edge_index = get_edge_indexes_sparse(x, threshold=0.5, device=device)
            edge_weight = torch.abs(x[edge_index[0], edge_index[1]])
            
                        

            # ==== GATES ====
            input_gate = torch.sigmoid(
                self.gconv(self.input_gnn, x, edge_index, edge_weight) +
                self.gconv(self.input_gnn_hidden_state, hidden_state, edge_index, edge_weight)
            )
            forget_gate = torch.sigmoid(
                self.gconv(self.forget_gnn, x, edge_index, edge_weight) +
                self.gconv(self.forget_gnn_hidden_state, hidden_state, edge_index, edge_weight)
            )
            output_gate = torch.sigmoid(
                self.gconv(self.output_gnn, x, edge_index, edge_weight) +
                self.gconv(self.output_gnn_hidden_state, hidden_state, edge_index, edge_weight)
            )
            mod_raw = (
                self.gconv_linear(self.modulation_gnn, x, edge_index, edge_weight) +
                self.gconv_linear(self.modulation_gnn_hidden_state, hidden_state, edge_index, edge_weight)
            )
            modulation = torch.relu(self.mod_norm(mod_raw))

            # 
            # ==== CELL STATE ====
            cell_state = torch.tanh(input_gate * modulation + forget_gate * cell_state)

            # ==== NEW HIDDEN STATE ====
            hidden_state = output_gate * torch.tanh(cell_state)
        
        #print(f"hidden FINAL mean={hidden_state.mean():.4f} std={hidden_state.std():.4f}")

        # ==== DG-Pooling ====
        pooled_graph, pool_loss = self.dg_pool(hidden_state)
        high_level_embeddings = pooled_graph.mean(dim=0)  # [k, F] → [F]

        # ==== LSTM raw fMRI ====
        low_level_embeddings = self.lstm_raw_time_series(time_series)


        # ==== Fusión ====
        fusion = torch.cat([high_level_embeddings, low_level_embeddings], dim=0)
        #fusion = self.layer_norm(fusion.unsqueeze(0)).squeeze(0)

        # ==== Clasificación ====
        pred = self.mlp_classiffier(fusion)

        return pred, pool_loss

    def lstm_raw_time_series(self, time_series_data):
        _, (h_last, _) = self.lstm_raw_fmri(time_series_data)
        h_last = h_last[-1].squeeze(0)
        return h_last

    def mlp_classiffier(self, concat_embedding):
        x = F.relu(self.mlp_layer_1(concat_embedding))
        x = self.mlp_ln(x)
        x = self.mlp_dropout(x)
        x = F.relu(self.mlp_layer_2(x))
        x = self.mlp_dropout(x)
        x = self.mlp_layer_3(x)
        return x

    def compute_loss(self, prediction_batch, label_batch, pool_losses_batch, lambda_pool = 0.01):
        loss_ce = F.binary_cross_entropy_with_logits(prediction_batch, label_batch)
        loss_pool = torch.mean(pool_losses_batch)
        return loss_ce + lambda_pool * loss_pool
