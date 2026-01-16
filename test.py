from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import os
import tempfile
import shutil
import random



print("=== INICIO DEL PROGRAMA TEST ===")
print("1. Importando librerías...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando device:", device)


print("2. Configurando semilla aleatoria...")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

print("3. Cargando configuración y datos...")
pipeline = 'cpac'
rois = 'rois_cc200'
phenotypic = 'all_cases'

BASE_DIR = Path.cwd()
data_path = BASE_DIR / "ABIDE_pcp"/ "cpac" / "filt_noglobal"
csv_path = data_path / "data.csv" 
df = pd.read_csv(csv_path)
origin_path = Path(data_path)


# List of all available neuroimaging sites in the dataset
all_sites = df['SITE_ID'].unique()
print("Sitios disponibles:", all_sites)

# Sites include in the analysis
sites = all_sites

# Designated site for external testing
test_site = 'YALE'
num_nodes = 200
num_node_features = 200


print("4. Definiendo función load_rois_data...")
def load_rois_data(sites):
    """
    Loads time series and diagnostic labels from neuroimaging data files for the specified sites.
    """
    print("   ↳ Ejecutando load_rois_data...")
    rois_time_series = {}  # Dictionary to store time series data for each site
    rois_labels = {}  # Dictionary to store labels for each site

    total_of_subjects = 0

    for site in sites:
        print(f"   ↳ Procesando sitio: {site}")
        site_time_series = []  # List to store time series for each subject at the site
        site_labels = []  # List to store labels for each subject at the site

        sub_df = df[df['SITE_ID'] == site]

        parent_folder = origin_path / site
        asd_child_folder = parent_folder / "ASD"
        tc_child_folder = parent_folder / "TC"

        # Define path for the time series data file
        for file_id in sub_df['FILE_ID']:
            name = file_id + '_rois_cc200.1D'
            filtered = sub_df.loc[sub_df['FILE_ID'] == file_id, :]

            if filtered.loc[:,'DX_GROUP'].iloc[0] == 1:
                data_path = asd_child_folder / name
                if not data_path.exists():
                    print(f"   ⚠️ File Not Found Error: Data file not found at path {data_path}")
                    continue

                data = np.loadtxt(data_path)

                # Check for NaN values and add time series to the site list
                if np.isnan(data).any():
                    print(f"   ⚠️ Value Error: NaN value found for subject {file_id}")
                else:
                    site_time_series.append(data)
                    site_labels.append(1)  # Assign 1 for ASD, 0 for control

            else:
                data_path = tc_child_folder / name
                if not data_path.exists():
                    print(f"   ⚠️ File Not Found Error: Data file not found at path {data_path}")
                    continue

                data = np.loadtxt(data_path)

                # Check for NaN values and add time series to the site list
                if np.isnan(data).any():
                    print(f"   ⚠️ Value Error: NaN value found for subject {file_id}")
                else:
                    site_time_series.append(data)
                    site_labels.append(0)  # Assign 1 for ASD, 0 for control

         # Store loaded data for the current site in the dictionaries
        rois_time_series[site] = site_time_series
        rois_labels[site] = np.array(site_labels)
        loaded_subjects_from_site = len(site_time_series)
        total_of_subjects+= loaded_subjects_from_site
        print(f"   ✅ Loaded {loaded_subjects_from_site} subjects from site {site}.")

    print(f"✅ Total loaded: {total_of_subjects} subjects")
    print("   ↳ Fin de load_rois_data")
    return rois_time_series, rois_labels

print("5. Cargando datos de ROIs...")
rois_time_series, rois_labels = load_rois_data(sites)

print("✅ Datos de ROIs cargados correctamente")

print("6. Cargando datos de grafos...")
lw_matrixes_data = torch.load((data_path / "lw_matrixes.pt"))

print("✅ Datos de grafos cargados correctamente")
def create_starting_hidden_state_graph(num_nodes: int, hidden_channels: int):
    return torch.zeros((num_nodes, hidden_channels), dtype=torch.float64)

def create_starting_cell_state(num_nodes:int, hidden_channels):
    return torch.zeros((num_nodes, hidden_channels), dtype=torch.float64)

from sklearn.metrics import confusion_matrix, roc_auc_score

def calculate_metrics(y_true, y_pred, y_pred_prob):
    """
    Calculate key evaluation metrics for binary classification tasks.
    """
    print("   ↳ Calculando métricas...")
    # Compute confusion matrix and unpack values
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Calculate metrics with safeguards against division by zero
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    auc = roc_auc_score(y_true, y_pred_prob)

    return accuracy, sensitivity, precision, specificity, auc, cm

def print_metrics(split, dataset_type, accuracy, sensitivity, precision, specificity, auc, cm):
    """
    Display evaluation metrics for a specific data split and dataset type.
    """
    print(f"   ↳ Mostrando métricas para {dataset_type}...")
    print(f"\n{dataset_type.capitalize()} Metrics for Split {split + 1}:")
    print(f"  Accuracy: {accuracy * 100:.2f}%")
    print(f"  Sensitivity (Recall): {sensitivity * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Specificity: {specificity * 100:.2f}%")
    print(f"  AUC-ROC Score: {auc * 100:.2f}%")
    print(f"  Confusion Matrix:\n{cm}")

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
        pool_loss = ((sig_scores * (1 - sig_scores)).mean()).to(device)

        return new_x, pool_loss
    

from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GNN_LSTM(nn.Module):
    def __init__(self, num_node_features, hidden_channels = 64, pool_ratio = 0.15):
        print("   ↳ Inicializando GNN_LSTM...")
        super().__init__()

        self.pool_ratio = pool_ratio
        self.hidden_channels = hidden_channels
        self.node_feat_dim = num_node_features

        #GraphConv para la entrada (G_t)
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
        self.dg_pool = DGPool(hidden_channels, pool_ratio)

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
        self.mlp_dropout = nn.Dropout(p = 0.3)


        

    def forward(self, lw_matrixes_sequence,edge_index , hidden_state, cell_state, time_series):
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
        for m in lw_matrixes_sequence:
            x, edge_index = m , edge_index
            # Normalizamos features del timestep
            x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

            # ==== GATES ====
            input_gate = torch.sigmoid(
                self.input_gnn(x, edge_index) +
                self.input_gnn_hidden_state(hidden_state, edge_index)
            )
            forget_gate = torch.sigmoid(
                self.forget_gnn(x, edge_index) +
                self.forget_gnn_hidden_state(hidden_state, edge_index)
            )
            output_gate = torch.sigmoid(
                self.output_gnn(x, edge_index) +
                self.output_gnn_hidden_state(hidden_state, edge_index)
            )
            modulation = torch.relu(
                self.modulation_gnn(x, edge_index) +
                self.modulation_gnn_hidden_state(hidden_state, edge_index)
            )

            # ==== CELL STATE ====
            cell_state = torch.tanh(input_gate * modulation + forget_gate * cell_state)

            # ==== NEW HIDDEN STATE ====
            hidden_state = output_gate * torch.tanh(cell_state)

        # ==== DG-Pooling ====
        pooled_graph, pool_loss = self.dg_pool(hidden_state)  # [N, hidden_channels] → [k, hidden_channels]
        high_level_embeddings = torch.mean(pooled_graph, dim=0)  # [k, hidden_channels] → [hidden_channels]

        # ==== LSTM raw fMRI ====
        low_level_embeddings = self.lstm_raw_time_series(time_series)  # [T, N] → [hidden_channels]

        # ==== Fusión ====
        fusion = torch.cat([high_level_embeddings, low_level_embeddings], dim=0)  # [hidden_channels] + [hidden_channels] → [hidden_channels * 2]
        fusion = self.layer_norm(fusion.unsqueeze(0)).squeeze(0)  # [hidden_channels * 2] → [1, hidden_channels * 2] → [hidden_channels * 2]

        # ==== Clasificación ====
        pred = self.mlp_classiffier(fusion)  # [hidden_channels * 2] → [1]

        return pred, pool_loss


    def lstm_raw_time_series(self,time_series_data):

        _, (h_last,_) = self.lstm_raw_fmri(time_series_data)
        h_last = h_last[-1].squeeze(0)  # [64]
        return h_last

    def mlp_classiffier(self,concat_embedding):

        concat_embedding = F.relu(self.mlp_layer_1(concat_embedding))
        concat_embedding = self.mlp_dropout(concat_embedding)
        concat_embedding = self.mlp_layer_2(concat_embedding)
        return concat_embedding

    def compute_loss(self, prediction_batch, label_batch, pool_losses_batch, lambda_pool=0.5):

        loss_ce = F.binary_cross_entropy_with_logits(prediction_batch, label_batch)
        loss_pool = torch.mean(pool_losses_batch)
        return loss_ce + lambda_pool * loss_pool




def get_edge_indexes_fully_connected():
    idx = torch.arange(num_nodes, device=device, dtype = torch.long)
    edge_index = torch.cartesian_prod(idx, idx).t()
    return edge_index[:, edge_index[0] != edge_index[1]]







# --- Evaluación ---
print("7. Inicializando modelo y optimizador...")
gnn_lstm = GNN_LSTM(num_node_features).to(device).double()
gnn_lstm.load_state_dict(torch.load('best_model.pth', map_location=device))



X = []
for site in sites:
    for subject_ts in rois_time_series[site]:
        X.append(subject_ts)
X_lw_matrixes = lw_matrixes_data

y = np.concatenate([rois_labels[site] for site in sites])

indices = np.arange(len(X))

_, idx_test = train_test_split(
    indices,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#print(idx_test)

X_tensors = [torch.tensor(ts, dtype=torch.float64) for ts in X]
y_tensor = torch.tensor(y, dtype=torch.float64)

gnn_lstm.eval()


test_hidden_state = create_starting_hidden_state_graph(num_nodes, gnn_lstm.hidden_channels).to(device)
test_cell_state = create_starting_cell_state(num_nodes, gnn_lstm.hidden_channels).to(device)


edge_index = get_edge_indexes_fully_connected()

test_y_true, test_y_pred, test_y_pred_prob = [], [], []

batch_size = 8
batch_count = 0

thresholds = [0.3,0.5,0.7]

print("7. Inicializando modelo y optimizador...")
gnn_lstm = GNN_LSTM(num_node_features).double()
gnn_lstm.load_state_dict(torch.load('best_model.pth', map_location=device))
gnn_lstm = gnn_lstm.to(device)

X = []
for site in sites:
    for subject_ts in rois_time_series[site]:
        X.append(subject_ts)
X_lw_matrixes = lw_matrixes_data

y = np.concatenate([rois_labels[site] for site in sites])

indices = np.arange(len(X))

_, idx_test = train_test_split(
    indices,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_tensors = [torch.tensor(ts, dtype=torch.float64) for ts in X]
y_tensor = torch.tensor(y, dtype=torch.float64)

gnn_lstm.eval()

edge_index = get_edge_indexes_fully_connected()

batch_size = 8
thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:
    print(f"\n{'='*50}")
    print(f"Evaluando con threshold = {threshold}")
    print(f"{'='*50}")
    
    test_y_true, test_y_pred, test_y_pred_prob = [], [], []
    batch_count = 0

    with torch.no_grad():
        for i in range(0, len(idx_test), batch_size):
            batch_count += 1
            print(f"\n   📦 Procesando batch {batch_count}/{(len(idx_test) + batch_size - 1) // batch_size}...")
            
            idxs_for_batch = idx_test[i:i+batch_size]
            time_series_batch = [X_tensors[idx].to(device) for idx in idxs_for_batch]

            for j, idx in enumerate(idxs_for_batch):
                # Mover matrices lw a device
                lw_matrixes_sequence = [m.to(device) for m in X_lw_matrixes[idx]]
                
                # Crear estados frescos para cada sujeto
                test_hidden_state = create_starting_hidden_state_graph(
                    num_nodes, gnn_lstm.hidden_channels
                ).to(device)
                test_cell_state = create_starting_cell_state(
                    num_nodes, gnn_lstm.hidden_channels
                ).to(device)

                pred_logits, pool_loss = gnn_lstm(
                    lw_matrixes_sequence=lw_matrixes_sequence,
                    edge_index=edge_index,
                    hidden_state=test_hidden_state,
                    cell_state=test_cell_state,
                    time_series=time_series_batch[j]
                )

                pred_logits = pred_logits.view(-1)
                pred_prob = torch.sigmoid(pred_logits)
                y_pred = 1 if pred_prob >= threshold else 0

                test_y_true.append(y_tensor[idx].item())
                test_y_pred.append(y_pred)
                test_y_pred_prob.append(pred_prob.cpu().item())

            print(f"   ✅ Batch {batch_count} completado")

    # --- Calcular métricas ---
    print(f"\n{'='*50}")
    print(f"Resultados finales con threshold = {threshold}")
    print(f"{'='*50}")
    accuracy, sensitivity, precision, specificity, auc, cm = calculate_metrics(
        test_y_true, test_y_pred, test_y_pred_prob
    )
    print_metrics(0, "test", accuracy, sensitivity, precision, specificity, auc, cm)