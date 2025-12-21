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

ruta = r"C:\Users\JonKo\Documents\Leandro ia\ABIDE_pcp\cpac\filt_noglobal\data.csv"
df = pd.read_csv(ruta)
path_string = r"C:\Users\JonKo\Documents\Leandro ia\ABIDE_pcp\cpac\filt_noglobal"
origin_path = Path(path_string)

# ruta = "C:/Users/marle/temp_abide/ABIDE_pcp/cpac/filt_noglobal/data.csv"
# df = pd.read_csv(ruta)
# path_string = "C:/Users/marle/temp_abide/ABIDE_pcp/cpac/filt_noglobal"
# origin_path = Path(path_string)


# List of all available neuroimaging sites in the dataset
all_sites = df['SITE_ID'].unique()
print("Sitios disponibles:", all_sites)

# Sites include in the analysis
sites = all_sites

# Designated site for external testing
test_site = 'YALE'
num_nodes = 200

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
graph_data = torch.load(r"C:\Users\JonKo\Documents\Leandro ia\Model\graph_data.pt")

print("✅ Datos de grafos cargados correctamente")
def create_starting_hidden_state_graph(num_nodes: int, hidden_channels: int):


    return  torch.zeros((num_nodes,hidden_channels))

def create_starting_cell_state(num_nodes:int, hidden_channels):

    return torch.zeros((num_nodes, hidden_channels))

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
    def __init__(self, input_dim, pool_ratio=0.5):
        print("   ↳ Inicializando DGPool...")
        super(DGPool, self).__init__()
        self.pool_ratio = pool_ratio
        self.trainable_vector_pooling = nn.Parameter(torch.randn(input_dim,1))

    def forward(self, graph_hidden_state_last: Data):

        x = graph_hidden_state_last.x  # [N, F]
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

        # Crear nuevo grafo completamente conectado (como en el paper)
        new_edge_index = self._fully_connect(indices, device=x.device)

        # Pooling loss
        pool_loss = ((sig_scores * (1 - sig_scores)).mean()).to(device)

        return Data(x=new_x, edge_index=new_edge_index), pool_loss, scores

    def _fully_connect(self, indices, device):
        all_edge_index = [[], []]
        for i, src in enumerate(indices):
            for j, dst in enumerate(indices):
                if i != j:
                    all_edge_index[0].append(i)
                    all_edge_index[1].append(j)
        return torch.tensor(all_edge_index, dtype=torch.long, device=device)

class GNN_LSTM(nn.Module):
    def __init__(self, node_feat_dim, hidden_channels = 64, pool_ratio = 0.5):
        print("   ↳ Inicializando GNN_LSTM...")
        super().__init__()

        self.pool_ratio = pool_ratio
        self.hidden_channels = hidden_channels
        self.node_feat_dim = node_feat_dim

        #GraphConv para la entrada (G_t)
        self.input_gnn = GraphConv(node_feat_dim, hidden_channels)
        self.forget_gnn = GraphConv(node_feat_dim, hidden_channels)
        self.output_gnn = GraphConv(node_feat_dim, hidden_channels)
        self.modulation_gnn = GraphConv(node_feat_dim, hidden_channels)

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

    def forward(self, graph_sequence, hidden_state, cell_state, time_series):
        """
        graph_sequence: lista de grafos del timestep t
        hidden_state: tensor [num_nodes, hidden_channels]
        cell_state: tensor [num_nodes, hidden_channels]
        time_series: tensor para LSTM raw fMRI
        """
        hidden_state = hidden_state
        cell_state = cell_state


        # Normalización del hidden y cell
        if torch.isnan(hidden_state).any() or torch.isnan(cell_state).any():
            print("⚠️ NaN detectado en hidden o cell")

        # Loop sobre timesteps
        for graph in graph_sequence:
            x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
            # Normalizamos features del timestep
            x = (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-6)

            # ==== GATES ====
            input_gate = torch.sigmoid(
                self.input_gnn(x, edge_index, edge_attr) +
                self.input_gnn_hidden_state(hidden_state, edge_index)
            )
            forget_gate = torch.sigmoid(
                self.forget_gnn(x, edge_index,edge_attr) +
                self.forget_gnn_hidden_state(hidden_state, edge_index)
            )
            output_gate = torch.sigmoid(
                self.output_gnn(x, edge_index,edge_attr) +
                self.output_gnn_hidden_state(hidden_state, edge_index)
            )
            modulation = torch.relu(
                self.modulation_gnn(x, edge_index,edge_attr) +
                self.modulation_gnn_hidden_state(hidden_state, edge_index)
            )

            # ==== CELL STATE ====
            cell_state = torch.tanh(input_gate * modulation + forget_gate * cell_state)

            # ==== NEW HIDDEN STATE ====
            hidden_state = output_gate * torch.tanh(cell_state)

        # ==== DG-Pooling ====
        pooled_graph, pool_loss, pool_scores = self.dg_pool(Data(x=hidden_state, edge_index=edge_index))
        high_level_embeddings = torch.mean(pooled_graph.x, dim=0)

        # ==== LSTM raw fMRI ====
        low_level_embeddings = self.lstm_raw_time_series(time_series)

        # ==== Fusion ====
        fusion = torch.cat([high_level_embeddings, low_level_embeddings], dim=0)
        fusion = self.layer_norm(fusion.unsqueeze(0)).squeeze(0)

        # ==== Clasificación ====
        pred = self.mlp_classiffier(fusion)

        return pred, pool_scores, pool_loss


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







# --- Evaluación ---
print("7. Inicializando modelo y optimizador...")
gnn_lstm = GNN_LSTM(node_feat_dim=7).to(device=device)
gnn_lstm.load_state_dict(torch.load('modelo_final.pth'))

X = []
for site in sites:
    for subject_ts in rois_time_series[site]:
        X.append(subject_ts)
X_graphs = []
for site in sites:
    for subject_graphs in graph_data[site]:
        X_graphs.append(subject_graphs)
y = np.concatenate([rois_labels[site] for site in sites])

indices = np.arange(len(X))

_, idx_test = train_test_split(
    indices,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(idx_test)

X_tensors = [torch.tensor(ts, dtype=torch.float32) for ts in X]
y_tensor = torch.tensor(y, dtype=torch.float32)

gnn_lstm.eval()


test_hidden_state = create_starting_hidden_state_graph(num_nodes, gnn_lstm.hidden_channels).to(device)
test_cell_state = create_starting_cell_state(num_nodes, gnn_lstm.hidden_channels).to(device)

test_y_true, test_y_pred, test_y_pred_prob = [], [], []

batch_size = 32
batch_count = 0

with torch.no_grad():
    for i in range(0, len(idx_test), batch_size):
        print(f"\n   📦 Procesando batch {batch_count + 1}...")
        idxs_for_batch = idx_test[i:i+batch_size]
        time_series_batch = [X_tensors[idx].to(device) for idx in idxs_for_batch]

        for j, idx in enumerate(idxs_for_batch):
            graph_sequence = [g.to(device) for g in X_graphs[idx]]

            pred_logits, _, pool_loss = gnn_lstm(
                graph_sequence=graph_sequence,
                hidden_state=test_hidden_state.clone(),
                cell_state=test_cell_state.clone(),
                time_series=time_series_batch[j]
            )

            pred_logits = pred_logits.view(-1)
            pred_prob = torch.sigmoid(pred_logits)
            y_pred = 1 if pred_prob >= 0.3 else 0

            test_y_true.append(y_tensor[idx].item())
            test_y_pred.append(y_pred)
            test_y_pred_prob.append(pred_prob.cpu().item())

        print(f"\n   📦 Fin de procesamiento del batch {batch_count + 1}...")





# --- Calcular métricas ---
accuracy, sensitivity, precision, specificity, auc, cm = calculate_metrics(test_y_true, test_y_pred, test_y_pred_prob)
print_metrics(0, "test", accuracy, sensitivity, precision, specificity, auc, cm)
