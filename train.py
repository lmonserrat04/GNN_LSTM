import numpy as np
from pathlib import Path
from config import torch
import time
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from config import device
from memory_cleanup import cleanup_batch_simple
from config import data_path, num_nodes, num_node_features, sites, df
from data_loader import load_rois_data
from model import GNN_LSTM
from checkpoint import save_checkpoint, load_checkpoint
from utils import set_seed, create_starting_hidden_state_graph, create_starting_cell_state, EarlyStopping, z_score_norm
from validation import validate


print("Usando device:", device)

set_seed()

print("Cargando datos...")

rois_time_series, rois_labels = load_rois_data(sites, df, Path(data_path))
all_data = torch.load(data_path / "lw_matrixes.pt", weights_only=False)

# all_data = [(lw_sub1, nodal_sub1), (lw_sub2, nodal_sub2), ...]
lw_matrixes_data  = [lw for lw, nodal in all_data]
node_features_data = [nodal for lw, nodal in all_data]


X         = [ts for site in sites for ts in rois_time_series[site]]
y         = np.concatenate([rois_labels[site] for site in sites])

X_norm    = [z_score_norm(ts) for site in sites for ts in rois_time_series[site]]
X_tensors = [torch.tensor(ts, dtype=torch.float64) for ts in X_norm]
y_tensor  = torch.tensor(y, dtype=torch.float64)

idx_train, idx_test = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)

# BUG CORREGIDO: límites de debug eliminados — usar dataset completo
#idx_train = idx_train[:8]
#idx_test = idx_test[:8]

torch.set_printoptions(threshold=torch.inf)


def run_training(cfg: dict, run_name: str) -> float:
    print(f"\n{'='*50}")
    print(f"  Run: {run_name}")
    for k, v in cfg.items():
        print(f"  {k} = {v}")
    print(f"{'='*50}\n")

    batch_size = cfg["batch_size"]

    gnn_lstm = GNN_LSTM(
        num_node_features,
        hidden_channels=cfg["hidden_channels"],
        pool_ratio=cfg["pool_ratio"],
    ).to(device).double()

    optimizer = torch.optim.Adam([
        {'params': [p for n, p in gnn_lstm.named_parameters() if 'gnn' in n], 'lr': cfg["lr"] * 10},
        {'params': [p for n, p in gnn_lstm.named_parameters() if 'gnn' not in n], 'lr': cfg["lr"]},
    ], weight_decay=cfg["weight_decay"])
    scheduler  = StepLR(optimizer, step_size=cfg["scheduler_step_size"], gamma=cfg["scheduler_gamma"])
    early_stop = EarlyStopping(patience=cfg["patience"], min_delta=cfg["min_delta"])

    checkpoint_path = f'checkpoint_{run_name}.pth'
    best_model_path = f'best_model_{run_name}.pth'

    start_epoch, last_batch_index, _ = load_checkpoint(gnn_lstm, optimizer, scheduler, early_stop, checkpoint_path)

    if last_batch_index * batch_size >= len(idx_train):
        last_batch_index = 0

    avg_train_loss = 0.0

    for epoch in range(start_epoch, cfg["n_epochs"]):
        print(f"🎯 Época {epoch+1}/{cfg['n_epochs']}  [{run_name}]")
        t_epoch = time.time()

        gnn_lstm.train()
        total_loss  = 0.0
        batch_count = last_batch_index
        loss        = torch.tensor(0.0, device=device)

        idxs = np.random.choice(idx_train, size=len(idx_train), replace=False)

        for i in range(last_batch_index * batch_size, len(idxs), batch_size):
            idxs_batch = idxs[i:i+batch_size]
            actual_batch_size = len(idxs_batch)
            

            time_series_batch          = [X_tensors[idx].detach().clone().to(device) for idx in idxs_batch]
            lw_matrixes_sequence_batch = [[m.to(device) for m in lw_matrixes_data[idx]] for idx in idxs_batch]
            node_features_data_sequence_batch = [[f.to(device) for f in node_features_data[idx]] for idx in idxs_batch]
            labels_batch               = y_tensor[idxs_batch].to(device)

            # BUG CORREGIDO: h y c se crean con el tamaño real del batch
            # El último batch puede tener menos de batch_size individuos
            h = create_starting_hidden_state_graph(num_nodes, actual_batch_size, gnn_lstm.hidden_channels).to(device)
            c = create_starting_cell_state(num_nodes, actual_batch_size, gnn_lstm.hidden_channels).to(device)

            # BUG CORREGIDO: antes llamaba forward 1 a 1 en un loop y acumulaba en listas vacías
            # Ahora forward recibe batch completo y devuelve [batch] preds + pool_loss scalar
            preds, pool_loss = gnn_lstm(
                lw_matrixes_sequence_batch=lw_matrixes_sequence_batch,
                node_features_data_sequence_batch = node_features_data_sequence_batch,
                hidden_state_batch=h,
                cell_state_batch=c,
                time_series_batch=time_series_batch,
            )

            # BUG CORREGIDO: compute_loss ya no recibe lista de pool_losses sino scalar
            loss = gnn_lstm.compute_loss(preds, labels_batch, pool_loss)

            optimizer.zero_grad()
            loss.backward()

            if epoch == 0 and batch_count == 0:
                print("\n==== DIAGNÓSTICO GRADIENTES ====")
                gnn_norms = []
                mlp_norms = []
                for name, param in gnn_lstm.named_parameters():
                    if param.grad is not None:
                        norm = param.grad.norm().item()
                        if 'gnn' in name:
                            gnn_norms.append((name, norm))
                        else:
                            mlp_norms.append((name, norm))

                print("── GNN ──")
                for name, norm in gnn_norms:
                    bar = '█' * min(int(norm * 200), 30)
                    print(f"  {name:<45} {norm:.6f}  {bar}")

                print("── LSTM + MLP ──")
                for name, norm in mlp_norms:
                    bar = '█' * min(int(norm * 200), 30)
                    print(f"  {name:<45} {norm:.6f}  {bar}")

                gnn_mean = sum(n for _, n in gnn_norms) / len(gnn_norms) if gnn_norms else 0
                mlp_mean = sum(n for _, n in mlp_norms) / len(mlp_norms) if mlp_norms else 0
                ratio    = mlp_mean / gnn_mean if gnn_mean > 0 else float('inf')
                print(f"\n  GNN grad medio:     {gnn_mean:.6f}")
                print(f"  LSTM+MLP grad medio:{mlp_mean:.6f}")
                print(f"  Ratio MLP/GNN:      {ratio:.1f}x  ← objetivo: < 10x")
                print("================================\n")


            #Inspeccionar gradientes
            # for name, param in gnn_lstm.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: grad_norm={param.grad.norm():.6f}")
            #     else:
            #         print(f"{name}: grad=None")

            gnn_params  = [p for n, p in gnn_lstm.named_parameters() if 'gnn' in n]
            rest_params = [p for n, p in gnn_lstm.named_parameters() if 'gnn' not in n]
            torch.nn.utils.clip_grad_norm_(gnn_params,  max_norm=5.0)
            torch.nn.utils.clip_grad_norm_(rest_params, max_norm=1.0)
            optimizer.step()

            total_loss  += loss.item()
            batch_count += 1
            avg_train_loss = total_loss / max(1, batch_count)

            cleanup_batch_simple(
                time_series_batch=time_series_batch,
                lw_matrixes_sequence_batch=lw_matrixes_sequence_batch,
                preds_batch=[preds],
                pool_losses_batch=[pool_loss],
                labels_batch=labels_batch,
                model=gnn_lstm, optimizer=optimizer,
                extra_vars={
                    'preds': preds,
                    'pool_loss': pool_loss,
                    'loss': loss,
                    'h': h,
                    'c': c,
                },
            )

        scheduler.step()
        save_checkpoint(gnn_lstm, optimizer, scheduler, early_stop, epoch, batch_count, loss.item(), checkpoint_path)

        gnn_lstm.eval()

        print(f"Validando Época {epoch+1}/{cfg['n_epochs']}")

        # BUG CORREGIDO: se eliminaron val_h y val_c de aquí
        # validate los crea internamente por batch con el tamaño correcto
        val_loss = validate(
            model=gnn_lstm,
            idx_test=idx_test,
            batch_size=cfg["batch_size"],
            epoch=epoch,
            X_tensors=X_tensors,
            y_tensor=y_tensor,
            X_lw_matrixes=lw_matrixes_data,
            X_node_features=node_features_data,
            device=device,
            threshold=0.5,
            num_nodes=num_nodes,
        )

        tiempo_epoca = time.time() - t_epoch
        print(f"\n📊 Época {epoch+1} | Train loss: {avg_train_loss:.4f} | Val loss actual: {val_loss:.4f} | Mejor val loss: {early_stop.best_loss:.4f} | {tiempo_epoca:.1f}s")

        if early_stop(gnn_lstm, val_loss, best_model_path):
            print(f"🛑 Early stopping en época {epoch+1} | Val loss actual: {val_loss:.4f} | Mejor val loss (modelo guardado): {early_stop.best_loss:.4f}")
            break

        last_batch_index = 0

    print(f"\n✅ Entrenamiento finalizado [{run_name}]")
    print(f"   Mejor val loss: {early_stop.best_loss:.4f}")

    done_path = f"{run_name}.done"
    with open(done_path, 'w') as f:
        f.write(str(early_stop.best_loss))
    print(f"   📝 .done escrito: {done_path}")

    return early_stop.best_loss


if __name__ == "__main__":
    cfg = {
        "pool_ratio":          0.5,
        "hidden_channels":     64,
        "lr":                  0.001,
        "weight_decay":        0.05,
        "scheduler_step_size": 10,
        "scheduler_gamma":     0.4,
        "batch_size":          32,
        "n_epochs":            150,
        "max_grad_norm":       5.0,
        "patience":            35,
        "min_delta":           0.001,
    }
    run_name = f"pool{cfg['pool_ratio']}_hid{cfg['hidden_channels']}"
    run_training(cfg, run_name)
