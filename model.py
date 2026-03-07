from config import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from jump_connection import jump_connection_parallel

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

        # Xavier initialization
        for gnn in [self.input_gnn, self.forget_gnn, self.output_gnn, self.modulation_gnn,
            self.input_gnn_hidden_state, self.forget_gnn_hidden_state,
            self.output_gnn_hidden_state, self.modulation_gnn_hidden_state]:
            nn.init.xavier_uniform_(gnn.lin.weight, gain=2.0)

        # ELIMINADO: self.combinate_skip = nn.Linear(hidden_channels*2, hidden_channels)
        # NUEVO: mapping layers según ecuación 14 del paper
        # p=1 → 1 layer para H_T
        # p=2 → 2 layers para H_T y H_(T-1)
        self.mapping_layers = nn.ModuleDict({
            'p1_i0': nn.Linear(hidden_channels, hidden_channels, bias=False),
            'p2_i0': nn.Linear(hidden_channels, hidden_channels, bias=False),
            'p2_i1': nn.Linear(hidden_channels, hidden_channels, bias=False),
        }).double()

        # Layer norm modulation gate
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
        
        # NUEVO: jump_connection_parallel devuelve lista de listas de tensors
        # [[h_T], [h_T, h_(T-1)]] para p=1 y p=2
        hidden_states_last_by_p = jump_connection_parallel(
            self, [1, 2], lw_matrixes_sequence, hidden_state, cell_state
        )

        # ELIMINADO: hidden_state_cat = torch.cat(hidden_states_last_by_p, dim=-1)
        # ELIMINADO: hidden_state = self.combinate_skip(hidden_state_cat)

        # NUEVO: ecuación 14 del paper — suma ponderada con mapping layers
        output = torch.zeros(
            hidden_state.shape[0], self.hidden_channels, dtype=torch.float64
        ).to(hidden_state.device)

        for p_idx, p in enumerate([1, 2]):
            last_states = hidden_states_last_by_p[p_idx]  # lista de p tensors
            for i, h in enumerate(last_states):
                output = output + self.mapping_layers[f'p{p_idx+1}_i{i}'](h)

        hidden_state = output

        # ==== DG-Pooling ====
        pooled_graph, pool_loss = self.dg_pool(hidden_state)
        high_level_embeddings = pooled_graph.mean(dim=0)  # [k, F] → [F]

        # ==== LSTM raw fMRI ====
        low_level_embeddings = self.lstm_raw_time_series(time_series)

        # ==== Fusión ====
        fusion = torch.cat([high_level_embeddings, low_level_embeddings], dim=0)

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

    def compute_loss(self, prediction_batch, label_batch, pool_losses_batch, lambda_pool=0.01):
        loss_ce = F.binary_cross_entropy_with_logits(prediction_batch, label_batch)
        loss_pool = torch.mean(pool_losses_batch)
        return loss_ce + lambda_pool * loss_pool