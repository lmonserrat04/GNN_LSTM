from config import device
from config import torch
from utils import get_edge_indexes_sparse
from torch_geometric.data import Data, Batch


def jump_connection_parallel(model, p_s, lw_matrixes_sequence_batch: list,node_features_data_sequence_batch:list, hidden_state_input_batch: torch.Tensor, cell_state_batch: torch.Tensor):

    if torch.isnan(hidden_state_input_batch).any() or torch.isnan(cell_state_batch).any():
        print("⚠️ NaN detectado en hidden o cell")

    num_nodes  = lw_matrixes_sequence_batch[0][0].shape[0]
    batch_size = len(lw_matrixes_sequence_batch)

    # Padeo lw matrixes y node features — igualar longitud de todos los individuos al máximo del batch
    max_timesteps = max(len(seq) for seq in lw_matrixes_sequence_batch)
    lw_padded = []
    nod_feat_padded = []
    for lw,n_f in zip(lw_matrixes_sequence_batch,node_features_data_sequence_batch):
        if len(lw) < max_timesteps:
            # Repetir último timestep hasta llegar a max_timesteps
            padding_lw = [lw[-1]] * (max_timesteps - len(lw))
            padding_n_f = [n_f[-1]] * (max_timesteps - len(n_f))
            lw_padded.append(lw + padding_lw)
            nod_feat_padded.append(n_f + padding_n_f)

        else:
            lw_padded.append(lw)
            nod_feat_padded.append(n_f)

   


    

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
                x_for_individual_lw = lw_padded[j][i].double()
                nod_f_for_individual = nod_feat_padded[j][i].double()

                edge_index_ind  = get_edge_indexes_sparse(x_for_individual_lw, threshold=0.5, device=device)
                edge_weight_ind = torch.abs(x_for_individual_lw[edge_index_ind[0], edge_index_ind[1]])

                data_x_list.append(Data(
                    x          = nod_f_for_individual,
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

            cell_state_p = torch.tanh(input_gate * modulation + forget_gate * cell_state_p)
            hidden_state = output_gate * torch.tanh(cell_state_p)
            hidden_states.append(hidden_state)

        # p=1 → [h_T]
        # p=2 → [h_T, h_(T-1)]
        last_p_hidden_states = hidden_states[-p:]
        hidden_states_last_by_p.append(last_p_hidden_states)

    return hidden_states_last_by_p