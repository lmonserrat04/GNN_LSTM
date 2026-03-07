from config import device
from config import torch
from utils import get_edge_indexes_sparse


def jump_connection_parallel(model, p_s, lw_matrixes_sequence, hidden_state_input, cell_state):

    if torch.isnan(hidden_state_input).any() or torch.isnan(cell_state).any():
        print("⚠️ NaN detectado en hidden o cell")

    hidden_states_last_by_p = []

    for p in p_s:
        hidden_states = []
        cell_state_p = cell_state.clone()  # copia fresca para cada p
        hidden_states.append(hidden_state_input)

        for i, x in enumerate(lw_matrixes_sequence):

            x = x.double()
            edge_index = get_edge_indexes_sparse(x, threshold=0.5, device=device)
            edge_weight = torch.abs(x[edge_index[0], edge_index[1]])

            if i > p - 1:
                hidden_state_skip_p = hidden_states[i-(p-1)]
            else:
                hidden_state_skip_p = hidden_states[0]

            # ==== GATES ====
            input_gate = torch.sigmoid(
                model.gconv(model.input_gnn, x, edge_index, edge_weight) +
                model.gconv(model.input_gnn_hidden_state, hidden_state_skip_p, edge_index, edge_weight)
            )
            forget_gate = torch.sigmoid(
                model.gconv(model.forget_gnn, x, edge_index, edge_weight) +
                model.gconv(model.forget_gnn_hidden_state, hidden_state_skip_p, edge_index, edge_weight)
            )
            output_gate = torch.sigmoid(
                model.gconv(model.output_gnn, x, edge_index, edge_weight) +
                model.gconv(model.output_gnn_hidden_state, hidden_state_skip_p, edge_index, edge_weight)
            )
            mod_raw = (
                model.gconv_linear(model.modulation_gnn, x, edge_index, edge_weight) +
                model.gconv_linear(model.modulation_gnn_hidden_state, hidden_state_skip_p, edge_index, edge_weight)
            )
            modulation = torch.relu(model.mod_norm(mod_raw))

            cell_state_p = torch.tanh(input_gate * modulation + forget_gate * cell_state_p)
            hidden_state = output_gate * torch.tanh(cell_state_p)
            hidden_states.append(hidden_state)

        # Guardamos los últimos p hidden states según ecuación 14
        # p=1 → [h_T]
        # p=2 → [h_T, h_(T-1)]
        last_p_hidden_states = hidden_states[-p:]
        hidden_states_last_by_p.append(last_p_hidden_states)

    return hidden_states_last_by_p