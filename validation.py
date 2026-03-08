import torch
import torch.nn.functional as F

from memory_cleanup import cleanup_batch_simple
from utils import create_starting_hidden_state_graph, create_starting_cell_state


def validate(model, idx_test, batch_size, epoch, X_tensors, y_tensor, X_lw_matrixes, device, threshold, num_nodes):

    print(f"Validando para epoca: {epoch+1} ")
    

    n_samples  = 0
    total_loss = 0

    with torch.no_grad():
        for i in range(0, len(idx_test), batch_size):

            idxs_for_batch    = idx_test[i:i+batch_size]
            actual_batch_size = len(idxs_for_batch)

            time_series_batch          = [X_tensors[idx].to(device) for idx in idxs_for_batch]
            lw_matrixes_sequence_batch = [[m.to(device) for m in X_lw_matrixes[idx]] for idx in idxs_for_batch]
            labels_batch               = y_tensor[idxs_for_batch].to(device)

            # BUG CORREGIDO: antes val_h y val_c venían de fuera con batch_size fijo
            # Ahora se crean aquí con el tamaño real del batch (último batch puede ser menor)
            h = create_starting_hidden_state_graph(num_nodes, actual_batch_size, model.hidden_channels).to(device)
            c = create_starting_cell_state(num_nodes, actual_batch_size, model.hidden_channels).to(device)

            # BUG CORREGIDO: antes llamaba model 1 a 1 en loop con edge_index hardcodeado
            # Ahora usa forward batched igual que train
            preds, pool_loss = model(
                lw_matrixes_sequence=lw_matrixes_sequence_batch,
                hidden_state_batch=h,
                cell_state_batch=c,
                time_series_batch=time_series_batch,
            )

            
            print(f"pred_prob={torch.sigmoid(preds):.4f} | label={labels_batch}")

            # BUG CORREGIDO: compute_loss ya recibe pool_loss scalar directamente
            loss = model.compute_loss(preds, labels_batch, pool_loss)

            total_loss += loss.item() * actual_batch_size
            n_samples  += actual_batch_size

            cleanup_batch_simple(
                time_series_batch=time_series_batch,
                lw_matrixes_sequence_batch=lw_matrixes_sequence_batch,
                preds_batch=[preds],
                pool_losses_batch=[pool_loss],
                labels_batch=labels_batch,
                model=model,
                optimizer=None,
                extra_vars={
                    'preds':     preds,
                    'pool_loss': pool_loss,
                    'loss':      loss,
                    'h':         h,
                    'c':         c,
                }
            )

    print(f"Fin de validacion en la epoca : {epoch+1}")
    return total_loss / n_samples
