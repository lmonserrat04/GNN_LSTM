from config import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from jump_connection import jump_connection_parallel
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DGPool(nn.Module):
    def __init__(self, input_dim, pool_ratio, num_nodes):
        print("   ↳ Inicializando DGPool...")
        super(DGPool, self).__init__()
        self.pool_ratio = pool_ratio
        # BUG CORREGIDO: antes no se guardaba num_nodes, necesario para reshape del batch
        self.num_nodes = num_nodes
        self.trainable_vector_pooling = nn.Parameter(torch.randn(input_dim, 1))

    def forward(self, x_batch):
        # BUG CORREGIDO: antes trataba [N*batch, F] como un único grafo gigante
        # Ahora procesa cada individuo por separado → devuelve [batch, F] + pool_loss scalar
        batch_size = x_batch.shape[0] // self.num_nodes
        x = x_batch.view(batch_size, self.num_nodes, -1)  # [batch, N, F]

        k = max(1, int(self.num_nodes * self.pool_ratio))
        norm2 = torch.norm(self.trainable_vector_pooling)
        w = self.trainable_vector_pooling / (norm2 + 1e-8)  # [F, 1]

        pooled_list = []
        pool_loss_list = []

        for b in range(batch_size):
            x_b = x[b]  # [N, F]
            scores = x_b @ w  # [N, 1]
            scores = (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-8)
            sig_scores = torch.sigmoid(scores)
            x_scaled = x_b * sig_scores

            _, indices = torch.topk(sig_scores.squeeze(), k=k)
            new_x = x_scaled[indices]  # [k, F]
            pooled_list.append(new_x.mean(dim=0))  # [F]

            sig_sorted, _ = torch.sort(sig_scores.squeeze(), descending=True)
            topk = sig_sorted[:k]
            rest = sig_sorted[k:]
            eps = 1e-8
            loss_b = -(torch.log(topk + eps).sum() + torch.log(1.0 - rest + eps).sum()) / self.num_nodes
            pool_loss_list.append(loss_b)

        pooled = torch.stack(pooled_list)               # [batch, F]
        pool_loss = torch.stack(pool_loss_list).mean()  # scalar
        return pooled, pool_loss


class GNN_LSTM(nn.Module):
    def __init__(self, num_node_features,num_nodes, hidden_channels=128, pool_ratio=0.5):
        print("   ↳ Inicializando GNN_LSTM...")
        super().__init__()

        self.pool_ratio = pool_ratio
        self.hidden_channels = hidden_channels
        self.node_feat_dim = num_node_features
        self.num_nodes = num_nodes

        # GCNConv para la entrada (G_t)
        self.input_gnn = GCNConv(num_node_features, hidden_channels)
        self.forget_gnn = GCNConv(num_node_features, hidden_channels)
        self.output_gnn = GCNConv(num_node_features, hidden_channels)
        self.modulation_gnn = GCNConv(num_node_features, hidden_channels, bias=False)

        # GCN para el hidden state (H_{t-p})
        self.input_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels)
        self.forget_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels)
        self.output_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels)
        self.modulation_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels, bias=False)

        # Dropout gnn
        self.gnn_dropout = nn.Dropout(p=0.3)

        self.gnn_norm = nn.LayerNorm(hidden_channels)

        
        # Xavier initialization
        for gnn in [self.input_gnn, self.forget_gnn, self.output_gnn, self.modulation_gnn,
                    self.input_gnn_hidden_state, self.forget_gnn_hidden_state,
                    self.output_gnn_hidden_state, self.modulation_gnn_hidden_state]:
            nn.init.xavier_uniform_(gnn.lin.weight, gain=2.0)

        for gnn in [self.forget_gnn, self.forget_gnn_hidden_state]:
            nn.init.constant_(gnn.bias, 1.0)

        # Mapping layers ecuación 14 del paper
        self.mapping_layers = nn.ModuleDict({
            'p1_i0': nn.Linear(hidden_channels, hidden_channels, bias=False),
            'p2_i0': nn.Linear(hidden_channels, hidden_channels, bias=False),
            'p2_i1': nn.Linear(hidden_channels, hidden_channels, bias=False),
        }).double()


        # BUG CORREGIDO: antes se instanciaba sin num_nodes
        self.dg_pool = DGPool(hidden_channels, pool_ratio, num_nodes)

        # LSTM para procesar datos raw
        self.lstm_raw_fmri = nn.LSTM(
            input_size=50,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=False
        )

        self.input_norm_lstm = nn.LayerNorm(50)
        self.embed_lstm_norm = nn.LayerNorm(hidden_channels)

        # MLP Clasificacion final
        self.mlp_layer_1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.mlp_layer_2 = nn.Linear(hidden_channels, hidden_channels // 2)
        self.mlp_layer_3 = nn.Linear(hidden_channels // 2, 1)
        self.mlp_dropout = nn.Dropout(p=0.5)
        self.mlp_ln = nn.LayerNorm(hidden_channels)

    def gconv(self, gcn_layer, batch):
        # BUG CORREGIDO: antes hacía gcn_layer(x) con objeto Batch directamente
        # GCNConv necesita (x, edge_index, edge_weight)
        return F.relu(gcn_layer(batch.x, batch.edge_index, edge_weight=batch.edge_attr))

    def gconv_linear(self, gcn_layer, batch):
        # BUG CORREGIDO: mismo problema que gconv
        return gcn_layer(batch.x, batch.edge_index, edge_weight=batch.edge_attr)

    def forward(self, lw_matrixes_sequence_batch,node_features_data_sequence_batch, hidden_state_batch, cell_state_batch, time_series_batch):

        hidden_states_last_by_p = jump_connection_parallel(
            model = self, 
            p_s = [1,2], 
            lw_matrixes_sequence_batch=lw_matrixes_sequence_batch, 
            node_features_sequence_batch=node_features_data_sequence_batch,
            hidden_state_input_batch=hidden_state_batch, 
            cell_state_batch=cell_state_batch,
            threshold= 0.5
        )

        # Ecuación 14 — suma ponderada con mapping layers
        output = torch.zeros(
            hidden_state_batch.shape[0], self.hidden_channels, dtype=torch.float64
        ).to(hidden_state_batch.device)

        for p_idx, p in enumerate([1, 2]):
            last_states = hidden_states_last_by_p[p_idx]
            for i, h in enumerate(last_states):
                output = output + self.mapping_layers[f'p{p_idx+1}_i{i}'](h)

        # ==== DG-Pooling batched ====
        # BUG CORREGIDO: antes mean(dim=0) colapsaba todo el batch en un solo vector
        # Ahora DGPool devuelve [batch, F] y pool_loss scalar
        high_level_embeddings, pool_loss = self.dg_pool(output)  # [batch, F]
        high_level_embeddings = self.gnn_norm(high_level_embeddings)

        # ==== LSTM raw fMRI ====
        # BUG CORREGIDO: antes recibía tensor individual, ahora recibe lista
        low_level_embeddings = self.lstm_raw_time_series(time_series_batch) # [batch, F]
        low_level_embeddings = self.embed_lstm_norm(low_level_embeddings)

        
        # ==== Fusión por individuo ====
        # BUG CORREGIDO: antes cat(..., dim=0) para un solo individuo
        # Ahora dim=1 porque ambos son [batch, F]
        fusion = torch.cat([high_level_embeddings, low_level_embeddings], dim=1)  # [batch, F*2]

        # ==== Clasificación ====
        preds = self.mlp_classiffier(fusion)  # [batch, 1]
        preds = preds.view(-1)                # [batch]

        return preds, pool_loss

    def lstm_raw_time_series(self, time_series_batch):

        time_series_batch = [self.input_norm_lstm(ts) for ts in time_series_batch]
        # Obtener longitudes reales de cada serie
        lengths = torch.tensor([ts.shape[0] for ts in time_series_batch], dtype=torch.long)

        # Padear al máximo del batch → [T_max, batch, N]
        padded = pad_sequence(time_series_batch, batch_first=False, padding_value=0.0)

        # Empaquetar para que el LSTM ignore el padding
        packed = pack_padded_sequence(padded, lengths.cpu(), batch_first=False, enforce_sorted=False)

        _, (h_last, _) = self.lstm_raw_fmri(packed)
        return h_last[-1]  # [batch, hidden]

    def mlp_classiffier(self, concat_embedding):
        # concat_embedding: [batch, hidden*2] — nn.Linear maneja batch automáticamente
        x = F.relu(self.mlp_layer_1(concat_embedding))
        x = self.mlp_ln(x)
        x = self.mlp_dropout(x)
        x = F.relu(self.mlp_layer_2(x))
        x = self.mlp_dropout(x)
        x = self.mlp_layer_3(x)
        return x  # [batch, 1]

    def compute_loss(self, prediction_batch, label_batch, pool_loss, lambda_pool=0.05):
        # BUG CORREGIDO: antes recibía pool_losses_batch tensor y hacía mean
        # Ahora pool_loss ya es scalar directo de DGPool batched
        loss_ce = F.binary_cross_entropy_with_logits(prediction_batch, label_batch)
        return loss_ce + lambda_pool * pool_loss
