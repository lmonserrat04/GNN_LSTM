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

# ── Cargar datos una sola vez ──────────────────────────────────────────────────
print("Cargando datos...")

rois_time_series, rois_labels = load_rois_data(sites, df, Path(data_path))
lw_matrixes_data              = torch.load(data_path / "lw_matrixes_reduced.pt", weights_only=False)



X         = [ts for site in sites for ts in rois_time_series[site]]
y         = np.concatenate([rois_labels[site] for site in sites])

X_norm    = [z_score_norm(ts) for site in sites for ts in rois_time_series[site]]
X_tensors = [torch.tensor(ts, dtype=torch.float64) for ts in X_norm]
y_tensor  = torch.tensor(y, dtype=torch.float64)

idx_train, idx_test = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=42)
#edge_index = get_edge_indexes_fully_connected(num_nodes, device)

idx_train = idx_train[:8]
idx_test = idx_test[:8]


torch.set_printoptions(threshold=torch.inf)

# ──────────────────────────────────────────────────────────────────────────────

def run_training(cfg: dict, run_name: str) -> float:
    print(f"\n{'='*50}")
    print(f"  Run: {run_name}")
    for k, v in cfg.items():
        print(f"  {k} = {v}")
    print(f"{'='*50}\n")

    gnn_lstm  = GNN_LSTM(
        num_node_features,
        hidden_channels=cfg["hidden_channels"],
        pool_ratio=cfg["pool_ratio"],
    ).to(device).double()

    #optimizer  = torch.optim.Adam(gnn_lstm.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    optimizer = torch.optim.Adam([
        {'params': [p for n, p in gnn_lstm.named_parameters() if 'gnn' in n], 'lr': cfg["lr"] * 10},
        {'params': [p for n, p in gnn_lstm.named_parameters() if 'gnn' not in n], 'lr': cfg["lr"]},
    ], weight_decay=cfg["weight_decay"])
    scheduler  = StepLR(optimizer, step_size=cfg["scheduler_step_size"], gamma=cfg["scheduler_gamma"])
    early_stop = EarlyStopping(patience=cfg["patience"], min_delta=cfg["min_delta"])

    checkpoint_path = f'checkpoint_{run_name}.pth'
    best_model_path = f'best_model_{run_name}.pth'

    starting_h = create_starting_hidden_state_graph(num_nodes, gnn_lstm.hidden_channels).to(device)
    starting_c = create_starting_cell_state(num_nodes, gnn_lstm.hidden_channels).to(device)

    start_epoch, last_batch_index, _ = load_checkpoint(gnn_lstm, optimizer, scheduler, early_stop, checkpoint_path)

    # Reset batch index si ya completó la época completa
    if last_batch_index * cfg["batch_size"] >= len(idx_train):
        last_batch_index = 0

    avg_train_loss = 0.0
    batch_size     = cfg["batch_size"]

    # ── Loop de entrenamiento ──────────────────────────────────────
    for epoch in range(start_epoch, cfg["n_epochs"]):
        print(f"🎯 Época {epoch+1}/{cfg['n_epochs']}  [{run_name}]")
        t_epoch = time.time()

        gnn_lstm.train()
        total_loss  = 0.0
        batch_count = last_batch_index
        loss        = torch.tensor(0.0, device=device)  # default si el loop no ejecuta

        idxs = np.random.choice(idx_train, size=len(idx_train), replace=False)

        for i in range(last_batch_index * batch_size, len(idxs), batch_size):
            idxs_batch = idxs[i:i+batch_size]

            time_series_batch          = [X_tensors[idx].detach().clone().to(device) for idx in idxs_batch]
            lw_matrixes_sequence_batch = [[m.to(device) for m in lw_matrixes_data[idx]] for idx in idxs_batch]
            labels_batch               = y_tensor[idxs_batch].to(device)

            preds_batch, pool_losses_batch = [], []

            for ts, lw_seq in zip(time_series_batch, lw_matrixes_sequence_batch):
                h = starting_h.detach().clone()
                c = starting_c.detach().clone()

                pred, pool_loss = gnn_lstm(
                    lw_matrixes_sequence=lw_seq,
                    hidden_state=h, cell_state=c,
                    time_series=ts,
                )

                pred = pred.view(-1)
                if pred.dim() == 0:
                    pred = pred.unsqueeze(0)

                del h, c, ts
                preds_batch.append(pred)
                pool_losses_batch.append(pool_loss)

            prediction_batch          = torch.stack(preds_batch).view(-1).to(device)
            pool_losses_batch_stacked = torch.stack(pool_losses_batch).view(-1).to(device)

            loss = gnn_lstm.compute_loss(prediction_batch, labels_batch, pool_losses_batch_stacked)

            optimizer.zero_grad()
            loss.backward()

            #En train.py después de loss.backward(), antes de optimizer.step()
            for name, param in gnn_lstm.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad_norm={param.grad.norm():.6f}")
                else:
                    print(f"{name}: grad=None")
            


            
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
                preds_batch=preds_batch,
                pool_losses_batch=pool_losses_batch,
                labels_batch=labels_batch,
                model=gnn_lstm, optimizer=optimizer,
                extra_vars={
                    'pred': pred, 'pool_loss': pool_loss,
                    'prediction_batch': prediction_batch,
                    'loss': loss,
                    'pool_losses_batch_stacked': pool_losses_batch_stacked,
                },
            )

        scheduler.step()
        save_checkpoint(gnn_lstm, optimizer, scheduler, early_stop, epoch, batch_count, loss.item(), checkpoint_path)

        # ── Validación al final de cada época ─────────────────────
        gnn_lstm.eval()
        val_h = create_starting_hidden_state_graph(num_nodes, gnn_lstm.hidden_channels).to(device)
        val_c = create_starting_cell_state(num_nodes, gnn_lstm.hidden_channels).to(device)

        print(f"Validando Época {epoch+1}/{cfg['n_epochs']}")

        val_loss = validate(
            model=gnn_lstm,
            idx_test=idx_test,
            batch_size=cfg["batch_size"],
            epoch=epoch,
            X_tensors=X_tensors,
            y_tensor=y_tensor,
            X_lw_matrixes=lw_matrixes_data,
            val_hidden_starting_state=val_h,
            val_cell_starting_state=val_c,
            device=device,
            threshold=0.5,
        )

        tiempo_epoca = time.time() - t_epoch
        print(f"\n📊 Época {epoch+1} | Train loss: {avg_train_loss:.4f} | Val loss actual: {val_loss:.4f} | Mejor val loss: {early_stop.best_loss:.4f} | {tiempo_epoca:.1f}s")

        if early_stop(gnn_lstm, val_loss, best_model_path):
            print(f"🛑 Early stopping en época {epoch+1} | Val loss actual: {val_loss:.4f} | Mejor val loss (modelo guardado): {early_stop.best_loss:.4f}")
            break

        last_batch_index = 0

    print(f"\n✅ Entrenamiento finalizado [{run_name}]")
    print(f"   Mejor val loss: {early_stop.best_loss:.4f}")

    import os
    done_path = f"{run_name}.done"
    with open(done_path, 'w') as f:
        f.write(str(early_stop.best_loss))
    print(f"   📝 .done escrito: {done_path}")

    return early_stop.best_loss


# ──────────────────────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = {
        "pool_ratio":          0.5,
        "hidden_channels":     128,
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