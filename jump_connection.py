from config import device
from config import torch
from utils import get_edge_indexes_sparse
from torch_geometric.data import Data, Batch


def jump_connection_parallel(model, p_s, lw_matrixes_sequence_batch: list,node_features_sequence_batch:list, hidden_state_input_batch: torch.Tensor, cell_state_batch: torch.Tensor):

    if torch.isnan(hidden_state_input_batch).any() or torch.isnan(cell_state_batch).any():
        print("⚠️ NaN detectado en hidden o cell")

    num_nodes  = lw_matrixes_sequence_batch[0][0].shape[0]
    batch_size = len(lw_matrixes_sequence_batch)

    # Longitudes reales de cada sujeto — necesarias para ignorar timesteps de padding
    real_lengths  = [len(seq) for seq in lw_matrixes_sequence_batch]
    max_timesteps = max(real_lengths)
    
    # AHORA: matrices de ceros → grafo vacío, sin edges activos con threshold=0.5
    zero_lw = torch.zeros_like(lw_matrixes_sequence_batch[0][0])
    zero_nf = torch.zeros_like(node_features_sequence_batch[0][0])
    
    lw_padded = []
    nf_padded  = []
    for lw,nf in zip(lw_matrixes_sequence_batch,node_features_sequence_batch):
        pad_len = max_timesteps - len(lw)
        lw_padded.append(lw + [zero_lw] * pad_len)
        nf_padded.append(nf + [zero_nf] * pad_len)

        

    hidden_states_last_by_p = []

    for p in p_s:
        hidden_states = []
        cell_state_p  = cell_state_batch.clone()
        hidden_states.append(hidden_state_input_batch)

        for i in range(max_timesteps):

            if i > p - 1:
                hidden_state_skip_p = hidden_states[i-(p-1)]
            else:
                hidden_state_skip_p = hidden_states[0]

            data_x_list            = []
            data_hidden_state_list = []

            for j in range(batch_size):
                lw_mat = lw_padded[j][i].double()
                nf_mat = nf_padded[j][i].double()

                edge_index_ind  = get_edge_indexes_sparse(lw_mat, threshold=0.5, device=device)
                edge_weight_ind = torch.abs(lw_mat[edge_index_ind[0], edge_index_ind[1]])

                data_x_list.append(Data(
                    x          = nf_mat,
                    edge_index = edge_index_ind,
                    edge_attr  = edge_weight_ind
                ))

                hs_individual = hidden_state_skip_p[j * num_nodes:(j + 1) * num_nodes]

                data_hidden_state_list.append(Data(
                    x          = hs_individual,
                    edge_index = edge_index_ind,
                    edge_attr  = edge_weight_ind
                ))

            batch_x  = Batch.from_data_list(data_x_list)
            batch_hs = Batch.from_data_list(data_hidden_state_list)

            # ==== GATES ====
            input_gate = torch.sigmoid(
                model.gconv(model.input_gnn, batch_x) +
                model.gconv(model.input_gnn_hidden_state, batch_hs)
            )
            forget_gate = torch.sigmoid(
                model.gconv(model.forget_gnn, batch_x) +
                model.gconv(model.forget_gnn_hidden_state, batch_hs)
            )
            output_gate = torch.sigmoid(
                model.gconv(model.output_gnn, batch_x) +
                model.gconv(model.output_gnn_hidden_state, batch_hs)
            )
            mod_raw = (
                model.gconv_linear(model.modulation_gnn, batch_x) +
                model.gconv_linear(model.modulation_gnn_hidden_state, batch_hs)
            )
            modulation = torch.relu(model.mod_norm(mod_raw))

            new_cell   = torch.tanh(input_gate * modulation + forget_gate * cell_state_p)
            new_hidden = output_gate * torch.tanh(new_cell)


            # Crear máscara [N*batch, 1] — 1.0 si timestep real, 0.0 si padding
            mask = torch.ones(batch_size * num_nodes, 1, dtype=torch.float64, device=new_cell.device)
            for j in range(batch_size):
                if i >= real_lengths[j]:
                    mask[j * num_nodes:(j + 1) * num_nodes] = 0.0

            # Interpolación differenciable: usa nuevo estado si real, conserva anterior si padding
            cell_state_p = mask * new_cell   + (1.0 - mask) * cell_state_p
            hidden_state = mask * new_hidden + (1.0 - mask) * hidden_states[-1]
            hidden_states.append(hidden_state)

        # p=1 → [h_T]
        # p=2 → [h_T, h_(T-1)]
        last_p_hidden_states = hidden_states[-p:]
        hidden_states_last_by_p.append(last_p_hidden_states)

    return hidden_states_last_by_p