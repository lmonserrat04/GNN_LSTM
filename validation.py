import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from memory_cleanup import cleanup_batch_simple



def validate(model,idx_test,batch_size,epoch,X_tensors,y_tensor, X_lw_matrixes,edge_index, val_hidden_starting_state, val_cell_starting_state, device,threshold):
    
    print(f"Validando para epoca: {epoch+1} ")

    
    batch_count = 0

    n_samples = 0
    total_loss = 0
    #total_batches = (len(idx_test) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(0, len(idx_test), batch_size):
            batch_count += 1
            
            
            idxs_for_batch = idx_test[i:i+batch_size]


            time_series_batch = [X_tensors[idx].to(device) for idx in idxs_for_batch]
            lw_matrixes_sequence_batch = []

            labels_batch = y_tensor[idxs_for_batch].to(device)

            preds_batch = []
            pool_losses_batch = []
            for j, idx in enumerate(idxs_for_batch):


                # Mover matrices lw a device
                lw_matrixes_sequence = [m.to(device) for m in X_lw_matrixes[idx]]
                lw_matrixes_sequence_batch.append(lw_matrixes_sequence)
                
                

                pred, pool_loss = model(
                    lw_matrixes_sequence=lw_matrixes_sequence,
                    edge_index=edge_index,
                    hidden_state=val_hidden_starting_state.clone(),
                    cell_state=val_cell_starting_state.clone(),
                    time_series=time_series_batch[j]
                )

                pred = pred.view(-1)
                pred_prob = torch.sigmoid(pred)
                y_pred = 1 if pred_prob >= threshold else 0

                

                preds_batch.append(pred)
                pool_losses_batch.append(pool_loss)
            


            # Apilar batch y mover a device
            prediction_batch = torch.stack(preds_batch).view(-1).to(device)
            pool_losses_batch_stacked  = torch.stack(pool_losses_batch).view(-1).to(device)

            # Calcular pérdida
            loss = model.compute_loss(prediction_batch, labels_batch, pool_losses_batch_stacked)

            total_loss += loss.item() * len(idxs_for_batch)
            n_samples += len(idxs_for_batch)

            cleanup_batch_simple(
                    time_series_batch=time_series_batch,
                    lw_matrixes_sequence_batch=lw_matrixes_sequence_batch,
                    preds_batch=preds_batch,
                    pool_losses_batch=pool_losses_batch,
                    labels_batch=labels_batch,
                    model=model,
                    optimizer=None,
                    extra_vars={
                        'pred': pred,
                        'pool_loss': pool_loss,
                        'prediction_batch': prediction_batch,
                        'loss': loss,
                        'pool_losses_batch_stacked': pool_losses_batch_stacked
                    }
                )

                
        

            
        
    print(f"Fin de validacion en la epoca : {epoch+1}")
    val_loss = total_loss / n_samples
    return val_loss

