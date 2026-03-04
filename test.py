from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

from config import data_path, num_nodes, num_node_features, test_site, BASE_DIR
from data_loader import load_rois_data
#from model import GNN_LSTM
from model_v1 import GNN_LSTM
from metrics import calculate_metrics, print_metrics
from utils import set_seed, create_starting_hidden_state_graph, create_starting_cell_state, get_edge_indexes_fully_connected, z_score_norm


print("=== INICIO DEL PROGRAMA TEST ===")
print("1. Importando librerías...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando device:", device)


print("2. Configurando semilla aleatoria...")
set_seed()

print("3. Cargando configuración y datos...")

csv_path = data_path / "data.csv"
df = pd.read_csv(csv_path)
origin_path = Path(data_path)


# List of all available neuroimaging sites in the dataset

print("5. Cargando datos de ROIs...")
rois_time_series, rois_labels = load_rois_data([test_site], df, origin_path)

print("✅ Datos de ROIs cargados correctamente")

print("6. Cargando datos de grafos...")
lw_matrixes_data = torch.load((data_path / "lw_matrixes_test.pt"))

print("✅ Datos de grafos cargados correctamente")


# --- Evaluación ---

print("7. Inicializando modelo y optimizador...")
gnn_lstm = GNN_LSTM(num_node_features, hidden_channels=128).double()
gnn_lstm.load_state_dict(torch.load(BASE_DIR / 'best_model_pool0.15_hid128.pth', map_location=device))
gnn_lstm = gnn_lstm.to(device)

X = []
for subject_ts in rois_time_series[test_site]:
    X.append(subject_ts)
X_lw_matrixes = lw_matrixes_data

y = np.array(rois_labels[test_site])

indices = np.arange(len(X))
test_site_size = len(rois_time_series[test_site])

X_tensors = [z_score_norm(ts) for ts in rois_time_series[test_site]]
y_tensor = torch.tensor(y, dtype=torch.float64)

gnn_lstm.eval()

edge_index = get_edge_indexes_fully_connected(num_nodes, device)

batch_size = 32
thresholds = [0.3, 0.5, 0.7]

for threshold in thresholds:
    print(f"\n{'='*50}")
    print(f"Evaluando con threshold = {threshold}")
    print(f"{'='*50}")
    
    test_y_true, test_y_pred, test_y_pred_prob = [], [], []
    batch_count = 0

    with torch.no_grad():
        for i in range(0, test_site_size, batch_size):
            batch_count += 1
            print(f"\n   📦 Procesando batch {batch_count}/{(test_site_size + batch_size - 1) // batch_size}...")
            
            idxs_for_batch = indices[i:i+batch_size]
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
