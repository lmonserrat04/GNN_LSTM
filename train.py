import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import os
import tempfile
import shutil


from memory_cleanup import cleanup_batch_simple
from config import data_path, num_nodes, num_node_features, test_site
from data_loader import load_rois_data
from model import DGPool, GNN_LSTM
from checkpoint import save_checkpoint, load_checkpoint
from utils import set_seed, create_starting_hidden_state_graph, create_starting_cell_state, get_edge_indexes_fully_connected

print("=== INICIO DEL PROGRAMA ===")
print("1. Importando librerías...")

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

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

print("4. Cargando datos de ROIs...")
rois_time_series, rois_labels = load_rois_data(sites, df, origin_path)

print("✅ Datos de ROIs cargados correctamente")

print("5. Cargando datos de matrices DFC...")
lw_matrixes_data = torch.load((data_path / "lw_matrixes.pt"))

print("✅ Datos de grafos cargados correctamente")

torch.set_printoptions(threshold=torch.inf)


print("6. Definiendo función de entrenamiento...")
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
        
    def __call__(self, model, val_loss, checkpoint_path):
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


def train_model():
    print("=== INICIANDO ENTRENAMIENTO CON MONITOREO ===")

    # 🔥 CREAR MONITOR
    from gpu_memory_monitor import GPUMemoryMonitor, monitor_batch_memory
    monitor = GPUMemoryMonitor()
    monitor.snapshot("INICIO_ENTRENAMIENTO")

    print("8. Preparando datos para entrenamiento...")
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

    

    # 🔥 SNAPSHOT DESPUÉS DE CREAR MODELO
    # monitor.snapshot("MODELO_CREADO")
    # monitor.compare_snapshots()

    # Preprocesar todo antes del loop
    X_tensors = [torch.tensor(ts, dtype=torch.float64) for ts in X]
    y_tensor = torch.tensor(y, dtype=torch.float64)

    

    # Edge index para features nodales
    edge_index = get_edge_indexes_fully_connected(num_nodes, device)

    # 🔥 SNAPSHOT DESPUÉS DE INICIALIZAR ESTADOS
    # monitor.snapshot("ESTADOS_INICIALES_CREADOS")
    # monitor.compare_snapshots()

    n_epochs_baseline = 150
    
    batch_size = 16

    avg_loss = 0


    pool_ratios = [0.15,0.30,0.50]

    for pool_idx in range(len(pool_ratios)):
        print(f"9. Inicializando modelo y optimizador con pool ratio = {pool_ratios[pool_idx]}....")
        # Early Stopping
        early_stopping = EarlyStopping()

        gnn_lstm = GNN_LSTM(num_node_features, pool_ratio= pool_ratios[pool_idx]).to(device).double()
        optimizer = torch.optim.Adam(gnn_lstm.parameters(), lr=1e-3, weight_decay=0.05)
        scheduler = StepLR(optimizer, step_size=20, gamma=0.4)

        checkpoint_path = f'checkpoint_pool_{pool_ratios[pool_idx]}.pth'
        best_model_path = f'best_model_pool_{pool_ratios[pool_idx]}.pth'

        # Estados iniciales en device
        starting_hidden_state = create_starting_hidden_state_graph(num_nodes, gnn_lstm.hidden_channels).to(device)
        starting_cell_state = create_starting_cell_state(num_nodes, gnn_lstm.hidden_channels).to(device)

        start_epoch = 0
        last_batch_index = 0

        start_epoch, last_batch_index, _ = load_checkpoint(gnn_lstm, optimizer, scheduler, checkpoint_path)


        print("Iniciando ciclo de entrenamiento...")

        

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

                for k, (time_series, lw_matrixes_sequence, _) in enumerate(zip(time_series_batch, lw_matrixes_sequence_batch, labels_batch)):
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
                print(f"   ✅ Batch {batch_count } completado - Tiempo: {tiempo_batch:.2f}s | Promedio por batch en la epoca: {tiempo_promedio_batch:.2f}s")


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
            
        
            if early_stopping(gnn_lstm, avg_loss, best_model_path):
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


        
print("7. Iniciando entrenamiento...")
train_model()

print("=== ENTRENAMIENTO COMPLETADO ===")
print("=== PROGRAMA FINALIZADO ===")
