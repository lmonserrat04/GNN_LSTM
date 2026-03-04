import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class DGPool(nn.Module):
    def __init__(self, input_dim, pool_ratio):
        print("   ↳ Inicializando DGPool...")
        super(DGPool, self).__init__()
        self.pool_ratio = pool_ratio
        self.trainable_vector_pooling = nn.Parameter(torch.randn(input_dim,1))

    def forward(self, lw_matrix_hidden_state_last):
        """
        Args:
            lw_matrix_hidden_state_last (torch.Tensor): Matriz de features nodales del último
                paso temporal del procesamiento GNN-LSTM. Forma: [N, F] donde N es el número 
                de nodos y F es la dimensión de features (hidden_channels).
    
        Returns:
            tuple: Una tupla de 3 elementos conteniendo:
                - new_x (torch.Tensor): Features nodales agrupadas de los top-k nodos. Forma: [k, F]
                - pool_loss (torch.Tensor): Pérdida de regularización que fomenta diversidad de scores. Tensor escalar.
                - scores (torch.Tensor): Scores sigmoid crudos de todos los nodos antes de la selección.
                    Forma: [N, 1]. Usado para análisis/visualización.

        """

        x = lw_matrix_hidden_state_last # [N, F]
        num_nodes = x.size(0)
        k = max(1, int(num_nodes * self.pool_ratio))

        # Scores por nodo
        norm2 = torch.norm(self.trainable_vector_pooling)
        scores = x @ (self.trainable_vector_pooling / (norm2 + 1e-8))  # [N,1]

        # Normalización (opcional depende del paper)
        scores = (scores - scores.mean()) / (scores.std(unbiased=False) + 1e-8)

        # Sigmoid para suavizar
        sig_scores = torch.sigmoid(scores)  # [N,1]

        # Escalar features
        x_scaled = x * sig_scores

        # Tomar top-k
        _, indices = torch.topk(sig_scores.squeeze(), k=k)
        new_x = x_scaled[indices]

        # # Crear nuevo grafo completamente conectado (como en el paper)
        # new_edge_index = self._fully_connect(indices, device=x.device)


        # Pooling loss
        # Ordenar scores descendente
        sig_scores_sorted, _ = torch.sort(sig_scores.squeeze(), descending=True)

        # Separar top-k y resto
        topk_scores = sig_scores_sorted[:k]
        rest_scores = sig_scores_sorted[k:]

        # Evitar log(0)
        eps = 1e-8

        # Pooling loss según ecuación (20)
        pool_loss = -(
            torch.log(topk_scores + eps).sum() +
            torch.log(1.0 - rest_scores + eps).sum()
        ) / num_nodes

        return new_x, pool_loss


class GNN_LSTM(nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, pool_ratio=0.15, num_nodes=200):
        print("   ↳ Inicializando GNN_LSTM...")
        super().__init__()

        self.pool_ratio = pool_ratio
        self.hidden_channels = hidden_channels
        self.node_feat_dim = num_node_features

        #GCNConv para la entrada (G_t)
        self.input_gnn = GCNConv(num_node_features, hidden_channels)
        self.forget_gnn = GCNConv(num_node_features, hidden_channels)
        self.output_gnn = GCNConv(num_node_features, hidden_channels)
        self.modulation_gnn = GCNConv(num_node_features, hidden_channels)

        # GCN para el hidden state (H_{t-p})
        self.input_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels)
        self.forget_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels)
        self.output_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels)
        self.modulation_gnn_hidden_state = GCNConv(hidden_channels, hidden_channels)

        ## Añadir capa de normalización para estabilidad
        self.layer_norm = nn.LayerNorm(hidden_channels * 2)

        # Dynamic Graph Pooling
        self.k = max(1, int(num_nodes * pool_ratio))
        self.dg_pool = DGPool(hidden_channels, pool_ratio)

        # Proyección post-pool: [k * hidden_channels] → [hidden_channels]
        # Preserva la identidad de cada nodo antes de fusionar (en lugar de mean)
        self.post_pool_fc = nn.Linear(self.k * hidden_channels, hidden_channels)

        #LSTM para procesar datos raw
        self.lstm_raw_fmri = nn.LSTM(
            input_size=num_nodes,                   # número de ROIs
            hidden_size=hidden_channels,      # tamaño del embedding temporal
            num_layers=1,                     # una sola capa
            batch_first=False
        )

        #MLP Clasificacion final
        self.mlp_layer_1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.mlp_layer_2 = nn.Linear(hidden_channels, 1)
        self.mlp_dropout = nn.Dropout(p=0.3)

    def gconv(self, gcn_layer, x, edge_index):
        """Aplica GCNConv + relu, implementando Gconv según Ecuación 13 del paper."""
        return F.relu(gcn_layer(x, edge_index))

    def forward(self, lw_matrixes_sequence, edge_index, hidden_state, cell_state, time_series):
        """
        Args:
        lw_matrixes_sequence (list): Lista de tensores representando la secuencia temporal
            de matrices de conectividad funcional. Cada elemento tiene forma [N, F] donde
            N = num_nodes y F = num_node_features.

        edge_index (torch.Tensor): Índices de aristas del grafo completamente conectado.
            Forma: [2, E] donde E es el número de aristas. Se reutiliza para todos los timesteps.

        hidden_state (torch.Tensor): Estado oculto inicial del GNN-LSTM. Forma: [N, hidden_channels].
            Típicamente inicializado con zeros al inicio de cada sujeto.
        cell_state (torch.Tensor): Estado de celda inicial del GNN-LSTM. Forma: [N, hidden_channels].
            Típicamente inicializado con zeros al inicio de cada sujeto.
        time_series (torch.Tensor): Serie temporal completa de fMRI raw del sujeto.
            Forma: [T, N] donde T = timepoints (~140-200) y N = num_nodes (200 ROIs).

             
        Returns:
            tuple: Una tupla de 3 elementos conteniendo:
                - pred (torch.Tensor): Logit de predicción binaria (antes de sigmoid). 
                    
                - pool_loss (torch.Tensor): Pérdida de regularización del pooling dinámico.
        """

        # Normalización del hidden y cell
        if torch.isnan(hidden_state).any() or torch.isnan(cell_state).any():
            print("⚠️ NaN detectado en hidden o cell")

        # Por cada matriz lw de la ventana en el tiempo t de un individuo
        for x in lw_matrixes_sequence:
            
            # ==== GATES ====
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

            # ==== CELL STATE ====
            cell_state = torch.tanh(input_gate * modulation + forget_gate * cell_state)

            # ==== NEW HIDDEN STATE ====
            hidden_state = output_gate * torch.tanh(cell_state)

        # ==== DG-Pooling ====
        pooled_graph, pool_loss = self.dg_pool(hidden_state)  # [N, hidden_channels] → [k, hidden_channels]
        high_level_embeddings = self.post_pool_fc(pooled_graph.flatten())  # [k * hidden_channels] → [hidden_channels]

        # ==== LSTM raw fMRI ====
        low_level_embeddings = self.lstm_raw_time_series(time_series)  # [T, N] → [hidden_channels]

        # ==== Fusión ====
        fusion = torch.cat([high_level_embeddings, low_level_embeddings], dim=0)  # [hidden_channels] + [hidden_channels] → [hidden_channels * 2]
        fusion = self.layer_norm(fusion.unsqueeze(0)).squeeze(0)  # [hidden_channels * 2] → [1, hidden_channels * 2] → [hidden_channels * 2]

        # ==== Clasificación ====
        pred = self.mlp_classiffier(fusion)  # [hidden_channels * 2] → [1]

        return pred, pool_loss


    def lstm_raw_time_series(self, time_series_data):

        _, (h_last,_) = self.lstm_raw_fmri(time_series_data)
        h_last = h_last[-1].squeeze(0)  # [64]
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