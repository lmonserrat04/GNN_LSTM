import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_starting_hidden_state_graph(num_nodes: int, hidden_channels: int):
    return torch.zeros((num_nodes, hidden_channels), dtype=torch.float64)


def create_starting_cell_state(num_nodes: int, hidden_channels):
    return torch.zeros((num_nodes, hidden_channels), dtype=torch.float64)


def get_edge_indexes_fully_connected(num_nodes, device):
    idx = torch.arange(num_nodes, device=device, dtype=torch.long)
    edge_index = torch.cartesian_prod(idx, idx).t()
    return edge_index[:, edge_index[0] != edge_index[1]]
