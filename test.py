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

from config import data_path, num_nodes, num_node_features, test_site
from data_loader import load_rois_data
from model import DGPool, GNN_LSTM
from metrics import calculate_metrics, print_metrics
from utils import set_seed, create_starting_hidden_state_graph, create_starting_cell_state, get_edge_indexes_fully_connected


print("=== INICIO DEL PROGRAMA TEST ===")
print("1. Importando librerías...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando device:", device)


print("2. Configurando semilla aleatoria...")
set_seed()

print("3. Cargando configuración y datos...")
pipeline = 'cpac'
rois = 'rois_cc200'
phenotypic = 'all_cases'

csv_path = data_path / "data.csv"
df = pd.read_csv(csv_path)
origin_path = Path(data_path)


# List of all available neuroimaging sites in the dataset
all_sites = df['SITE_ID'].unique()
print("Sitios disponibles:", all_sites)

# Sites include in the analysis
sites = all_sites

print("5. Cargando datos de ROIs...")
rois_time_series, rois_labels = load_rois_data(sites, df, origin_path)

print("✅ Datos de ROIs cargados correctamente")

print("6. Cargando datos de grafos...")
lw_matrixes_data = torch.load((data_path / "lw_matrixes.pt"))

print("✅ Datos de grafos cargados correctamente")


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


edge_index = get_edge_indexes_fully_connected(num_nodes, device)

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

edge_index = get_edge_indexes_fully_connected(num_nodes, device)

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
