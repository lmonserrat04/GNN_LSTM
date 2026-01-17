import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import os
import tempfile
import shutil

from torch_geometric.data import Data
from memory_cleanup import cleanup_batch_simple

print("=== INICIO DEL PROGRAMA ===")
print("1. Importando librerías...")

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando device:", device)

print("2. Configurando semilla aleatoria...")
import random
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

print("6. Cargando datos de matrices DFC...")
lw_matrixes_data = torch.load((data_path / "lw_matrixes.pt"))

print("✅ Datos de grafos cargados correctamente")


print("10. Definiendo funciones de checkpoint...")

def save_checkpoint(model, optimizer, scheduler, epoch, current_batch_index, loss, path='checkpoint.pth'):
    """Guarda el estado completo del entrenamiento de forma atómica"""
    print(f"   💾 Guardando checkpoint para época {epoch}...")

    try:
        # 1. Crear el diccionario de checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'batch_idx': current_batch_index,
        }

        # 2. Guardar en un archivo temporal primero
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f"checkpoint_temp_{os.getpid()}.pth")

        # Guardar con formato que permita verificación
        torch.save(checkpoint, temp_path, _use_new_zipfile_serialization=True)

        # 3. Verificar que el archivo temporal se guardó correctamente
        try:
            # Intentar cargar el archivo temporal para verificar integridad
            test_checkpoint = torch.load(temp_path, map_location='cpu', weights_only=False)

            # Verificar que todas las claves necesarias están presentes
            required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict',
                           'scheduler_state_dict', 'loss', 'batch_idx']
            for key in required_keys:
                if key not in test_checkpoint:
                    raise ValueError(f"Clave faltante en checkpoint: {key}")

            print(f"   ✅ Checkpoint verificado correctamente")

        except Exception as e:
            print(f"   ❌ Error al verificar checkpoint: {e}")
            # Limpiar archivo temporal corrupto
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

        # 4. Si existe checkpoint anterior, crear backup
        backup_path = path + '.bak'
        if os.path.exists(path):
            try:
                shutil.copy2(path, backup_path)
                print(f"   📁 Backup creado: {backup_path}")
            except Exception as e:
                print(f"   ⚠️  No se pudo crear backup: {e}")

        # 5. Mover archivo temporal a ubicación final (operación atómica)
        shutil.move(temp_path, path)

        # 6. Verificar que el archivo final existe y tiene tamaño
        if os.path.exists(path) and os.path.getsize(path) > 100:  # Mínimo 100 bytes
            print(f"   ✅ Checkpoint guardado exitosamente en época {epoch}, batch: {current_batch_index}")
            print(f"   📊 Pérdida guardada: {loss:.6f}")
        else:
            raise IOError("El archivo final no se creó correctamente")

    except Exception as e:
        print(f"   ❌ Error crítico al guardar checkpoint: {e}")
        print(f"   ⚠️  El entrenamiento continuará sin guardar este checkpoint")

        # Intentar recuperar el backup si existe
        if os.path.exists(backup_path) and not os.path.exists(path):
            try:
                shutil.copy2(backup_path, path)
                print(f"   🔄 Restaurado checkpoint desde backup")
            except:
                pass


def load_checkpoint(model, optimizer, scheduler, path):
    """Carga el estado completo del entrenamiento con recuperación ante fallos"""
    print(f"   📥 Intentando cargar checkpoint desde {path}...")

    backup_path = path + '.bak'
    temp_path = os.path.join(tempfile.gettempdir(), f"checkpoint_temp_{os.getpid()}.pth")

    # Lista de posibles ubicaciones en orden de prioridad
    possible_paths = [
        path,              # Ubicación principal
        backup_path,       # Backup
        temp_path          # Archivo temporal (si existe)
    ]

    for checkpoint_path in possible_paths:
        if os.path.exists(checkpoint_path):
            try:
                print(f"   🔍 Probando {checkpoint_path}...")

                # Intentar cargar con diferentes métodos
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                except:
                    # Fallback si el método anterior falla
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')

                # Verificar estructura básica
                if not isinstance(checkpoint, dict):
                    print(f"   ❌ Formato inválido en {checkpoint_path}")
                    continue

                required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict',
                               'scheduler_state_dict', 'loss', 'batch_idx']
                missing_keys = [k for k in required_keys if k not in checkpoint]

                if missing_keys:
                    print(f"   ❌ Claves faltantes en {checkpoint_path}: {missing_keys}")
                    continue

                # Cargar estados
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                start_epoch = checkpoint['epoch']
                current_batch_index = checkpoint['batch_idx']
                loss = checkpoint['loss']

                print(f"   ✅ Checkpoint cargado desde {checkpoint_path}")
                print(f"   📊 Continuando desde época {start_epoch}, batch: {current_batch_index}")
                print(f"   📉 Pérdida anterior: {loss:.6f}")

                # Si cargamos desde backup, restaurar como checkpoint principal
                if checkpoint_path == backup_path and os.path.exists(backup_path):
                    try:
                        shutil.copy2(backup_path, path)
                        print(f"   🔄 Backup restaurado como checkpoint principal")
                    except:
                        pass

                return start_epoch, current_batch_index, loss

            except Exception as e:
                print(f"   ❌ Error al cargar {checkpoint_path}: {e}")
                continue

    # Si llegamos aquí, ningún checkpoint fue válido
    print("   ⚠️  No se encontró checkpoint válido. Comenzando desde cero.")

    # Limpiar archivos corruptos si existen
    for p in [path, backup_path, temp_path]:
        if os.path.exists(p):
            try:
                file_size = os.path.getsize(p)
                if file_size < 100:  # Archivo demasiado pequeño, probablemente corrupto
                    os.remove(p)
                    print(f"   🗑️  Eliminado archivo corrupto: {p} ({file_size} bytes)")
            except:
                pass

    return 0, 0, float('inf')


print("7. Definiendo funciones auxiliares...")
def create_starting_hidden_state_graph(num_nodes: int, hidden_channels: int):


    return  torch.zeros((num_nodes,hidden_channels), dtype=torch.float64)

def create_starting_cell_state(num_nodes:int, hidden_channels):

    return torch.zeros((num_nodes, hidden_channels),dtype=torch.float64)


print("9. Definiendo clases del modelo...")
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
    

from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class GNN_LSTM(nn.Module):
    def __init__(self, num_node_features, hidden_channels = 64, pool_ratio = 0.15):
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
        for x in lw_matrixes_sequence:
            

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

torch.set_printoptions(threshold=torch.inf)


def get_edge_indexes_fully_connected():
    idx = torch.arange(num_nodes, device=device, dtype = torch.long)
    edge_index = torch.cartesian_prod(idx, idx).t()
    return edge_index[:, edge_index[0] != edge_index[1]]



print("11. Definiendo función de entrenamiento...")
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
import time

###################################################################
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        
    def __call__(self, model, val_loss, checkpoint_path='best_model.pth'):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Nuevo mejor modelo guardado (loss: {val_loss:.4f})")
            return False
        else:
            self.counter += 1
            print(f"⚠️  Sin mejora: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                model.load_state_dict(torch.load(checkpoint_path))
                print(f"🔙 Restaurando mejor modelo (loss: {self.best_loss:.4f})")
                return True
        return False
    
###################################################################


def train_model(checkpoint_path='checkpoint.pth'):
    print("=== INICIANDO ENTRENAMIENTO CON MONITOREO ===")

    # 🔥 CREAR MONITOR
    from gpu_memory_monitor import GPUMemoryMonitor, monitor_batch_memory
    monitor = GPUMemoryMonitor()
    monitor.snapshot("INICIO_ENTRENAMIENTO")

    print("12. Preparando datos para entrenamiento...")
    X = []
    for site in sites:
        for subject_ts in rois_time_series[site]:
            X.append(subject_ts)
            
    X_lw_matrixes = lw_matrixes_data

    y = np.concatenate([rois_labels[site] for site in sites])

    indices = np.arange(len(X))

    idx_train, idx_test = train_test_split(
        indices,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("13. Inicializando modelo y optimizador...")
    gnn_lstm = GNN_LSTM(num_node_features).to(device).double()
    optimizer = torch.optim.Adam(gnn_lstm.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.4)

    # 🔥 SNAPSHOT DESPUÉS DE CREAR MODELO
    # monitor.snapshot("MODELO_CREADO")
    # monitor.compare_snapshots()

    # Preprocesar todo antes del loop
    X_tensors = [torch.tensor(ts, dtype=torch.float64) for ts in X]
    y_tensor = torch.tensor(y, dtype=torch.float64)

    # Estados iniciales en device
    starting_hidden_state = create_starting_hidden_state_graph(num_nodes, gnn_lstm.hidden_channels).to(device)
    starting_cell_state = create_starting_cell_state(num_nodes, gnn_lstm.hidden_channels).to(device)

    # Edge index para features nodales
    edge_index = get_edge_indexes_fully_connected()

    # 🔥 SNAPSHOT DESPUÉS DE INICIALIZAR ESTADOS
    # monitor.snapshot("ESTADOS_INICIALES_CREADOS")
    # monitor.compare_snapshots()

    n_epochs_baseline = 150
    
    batch_size = 16


    start_epoch = 0
    last_batch_index = 0

    avg_loss = 0


    start_epoch, last_batch_index, _ = load_checkpoint(gnn_lstm, optimizer, scheduler, checkpoint_path)

    # Early Stopping
    early_stopping = EarlyStopping()

    print("14. Iniciando ciclo de entrenamiento...")

    # Entrenamiento
    for epoch in range(start_epoch, n_epochs_baseline):
        print(f"\n🎯 INICIANDO ÉPOCA {epoch + 1}/{n_epochs_baseline}")

        # # 🔥 SNAPSHOT AL INICIO DE ÉPOCA
        # monitor.snapshot(f"EPOCH_{epoch}_START")
        # Variables de tiempo por época (se reinician cada época)
        tiempo_total_epoch = 0
        tiempo_inicio_epoch = time.time()

        gnn_lstm.train()
        total_loss = 0
        batch_count = last_batch_index

        idxs_for_epoch = np.random.choice(idx_train, size=len(idx_train), replace=False)


         # # 🔥 SNAPSHOT PRE-BATCH
        #monitor_batch_memory(monitor, batch_count, epoch, "PRE_BATCH")
        
        for i in range(last_batch_index * batch_size, len(idxs_for_epoch), batch_size):
            current_batch_index = i // batch_size
            print(f"Iniciando procesamiento de batch {current_batch_index + 1} en Epoca: {epoch + 1} ")
            
            
            inicio_batch = time.time()

           


            idxs_for_batch = idxs_for_epoch[i:i+batch_size]
            time_series_batch = [
                X_tensors[idx].detach().clone().to(device)
                for idx in idxs_for_batch
            ]

            lw_matrixes_sequence_batch = [X_lw_matrixes[idx] for idx in idxs_for_batch]

            labels_batch = y_tensor[idxs_for_batch]

            # # 🔥 SNAPSHOT DESPUÉS DE CARGAR DATOS
            

            # Mover device
            time_series_batch = [ts.to(device) for ts in time_series_batch]
            lw_matrixes_sequence_batch = [[m.to(device) for m in subject_lw_matrixes] for subject_lw_matrixes in lw_matrixes_sequence_batch]
            labels_batch = labels_batch.to(device)

            # # 🔥 SNAPSHOT DESPUÉS DE MOVER A GPU
            #monitor_batch_memory(monitor, batch_count, epoch, "DATOS_EN_GPU")

            preds_batch = []
            pool_losses_batch = []

            for j, (time_series, lw_matrixes_sequence, _) in enumerate(zip(time_series_batch, lw_matrixes_sequence_batch, labels_batch)):
                h = starting_hidden_state.detach().clone()
                c = starting_cell_state.detach().clone()

                # Forward pass
                pred, pool_loss = gnn_lstm(
                    lw_matrixes_sequence = lw_matrixes_sequence,
                    edge_index = edge_index,
                    hidden_state=h,
                    cell_state=c,
                    time_series=time_series
                )

                pred = pred.view(-1)
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)

                del h, c, time_series

                preds_batch.append(pred)
                pool_losses_batch.append(pool_loss)

            # # 🔥 SNAPSHOT DESPUÉS DE FORWARD
            # monitor.snapshot(f"E{epoch}_B{batch_count}_FORWARD_DONE")

            # Apilar batch y mover a device
            prediction_batch = torch.stack(preds_batch).view(-1).to(device)
            pool_losses_batch_stacked  = torch.stack(pool_losses_batch).view(-1).to(device)

            # Calcular pérdida y backward
            loss = gnn_lstm.compute_loss(prediction_batch, labels_batch, pool_losses_batch_stacked )

            optimizer.zero_grad()
            loss.backward()

            # # 🔥 SNAPSHOT DESPUÉS DE BACKWARD
            # monitor.snapshot(f"E{epoch}_B{batch_count}_BACKWARD_DONE")

            torch.nn.utils.clip_grad_norm_(gnn_lstm.parameters(), max_norm=1.0)
            optimizer.step()

            # # 🔥 SNAPSHOT DESPUÉS DE OPTIMIZER STEP
            # monitor.snapshot(f"E{epoch}_B{batch_count}_OPTIMIZER_DONE")

            total_loss += loss.item()
            batch_count += 1



            if batch_count % 10 == 0:
                save_checkpoint(gnn_lstm, optimizer, scheduler, epoch, current_batch_index, loss.item(), checkpoint_path)



            # # 🔥 SNAPSHOT POST-BATCH (ANTES DE LIMPIEZA)
            # monitor_batch_memory(monitor, batch_count-1, epoch, "POST_BATCH_ANTES_DE_LIMPIEZA")

            # # 🔥 COMPARAR PRE vs POST
            # print("\n" + "="*80)
            # print(f"📊 COMPARACIÓN BATCH {batch_count-1}")
            # print("="*80)
            # # Comparar PRE_BATCH con POST_BATCH
            # monitor.compare_snapshots(-8, -1)  # PRE vs POST_PRE_CLEANUP



            # Limpieza
            # ─────────────────────────────────────────────────────────────
            # ✅ AHORA SÍ: Limpieza DESPUÉS de backward
            # ─────────────────────────────────────────────────────────────
            cleanup_batch_simple(
                time_series_batch=time_series_batch,
                lw_matrixes_sequence_batch=lw_matrixes_sequence_batch,
                preds_batch=preds_batch,
                pool_losses_batch=pool_losses_batch,
                labels_batch=labels_batch,
                model=gnn_lstm,
                optimizer=optimizer,
                extra_vars={
                    'pred': pred,
                    'pool_loss': pool_loss,
                    'prediction_batch': prediction_batch,
                    'loss': loss,
                    'pool_losses_batch_stacked': pool_losses_batch_stacked
                }
            )

            # # 🔥 SNAPSHOT POST-LIMPIEZA
            #monitor_batch_memory(monitor, batch_count-1, epoch, "POST_LIMPIEZA")

            # # 🔥 COMPARAR POST_PRE_CLEANUP vs POST_CLEANUP
            # print(f"\n🧹 EFECTIVIDAD DE LIMPIEZA:")
            # monitor.compare_snapshots(-2, -1)


             # Calcular tiempo del batch actual
            fin_batch = time.time()
            tiempo_batch = fin_batch - inicio_batch
            tiempo_total_epoch += tiempo_batch

            # Calcular tiempo promedio por batch en esta época
            batches_en_epoch = batch_count - last_batch_index
            tiempo_promedio_batch = tiempo_total_epoch / max(1, batches_en_epoch)

            new_loss = total_loss / max(1, batch_count)

            # Reporte de loss y tiempo cada x batches
            if batches_en_epoch % 10 == 0:
                print("\n" + "🔥"*40)
                print(f"Loss= {new_loss:.4f}. ΔLoss = {(new_loss - avg_loss):.4f}")
               
                print("🔥"*40 + "\n")


            avg_loss = new_loss
            print(f"   ✅ Batch {batch_count + 1} completado - Tiempo: {tiempo_batch:.2f}s | Promedio por batch en la epoca: {tiempo_promedio_batch:.2f}s")


        scheduler.step()

        # Calcular tiempo total de la época
        tiempo_fin_epoch = time.time()
        tiempo_total_epoch = tiempo_fin_epoch - tiempo_inicio_epoch

        # Calcular estadísticas finales de la época
        batches_completados_epoch = batch_count - last_batch_index
        if batches_completados_epoch > 0:
            tiempo_promedio_epoch = tiempo_total_epoch / batches_completados_epoch
        else:
            tiempo_promedio_epoch = 0

        print(f"\n📊 Epoch {epoch}/{n_epochs_baseline}")
        print(f"   Loss: {avg_loss:.4f}")
        print(f"   Tiempo total: {tiempo_total_epoch:.2f}s")
        print(f"   Tiempo promedio por batch: {tiempo_promedio_epoch:.2f}s")
        print(f"   Batches procesados: {batches_completados_epoch}")

        # Evaluación y Early Stopping
        
    
        if early_stopping(gnn_lstm, avg_loss, 'best_model.pth'):
            print(f"🛑 Early stopping en época {epoch}")
            break

        # 🔥 REPORTE AL FINAL DE ÉPOCA
        print("\n" + "="*80)
        print(f"📊 REPORTE FIN DE ÉPOCA {epoch}")
        print("="*80)
        monitor.print_detailed_report()
        monitor.snapshot(f"EPOCH_{epoch + 1}_END")

        # Comparar inicio vs fin de época
        epoch_start_idx = None
        for idx, snap in enumerate(monitor.snapshots):
            if snap['label'] == f"EPOCH_{epoch + 1}_START":
                epoch_start_idx = idx
                break

        if epoch_start_idx is not None:
            print(f"\n🔍 COMPARACIÓN ÉPOCA {epoch + 1}: INICIO vs FIN")
            monitor.compare_snapshots(epoch_start_idx, -1)

        # Reiniciar last_batch_index para la próxima época
        last_batch_index = 0

    print("17. Guardando modelo final...")
    torch.save(gnn_lstm.state_dict(), 'modelo_final.pth')
    print("✅ Modelo final guardado como 'modelo_final.pth'")

        
print("16. Iniciando entrenamiento...")
train_model('checkpoint.pth')

print("=== ENTRENAMIENTO COMPLETADO ===")
print("=== PROGRAMA FINALIZADO ===")