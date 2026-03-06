import random
import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def z_score_norm(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std == 0] = 1
    return (x - mean) / std


def create_starting_hidden_state_graph(num_nodes: int, hidden_channels: int):
    return torch.zeros((num_nodes, hidden_channels), dtype=torch.float64)


def create_starting_cell_state(num_nodes: int, hidden_channels: int):
    return torch.zeros((num_nodes, hidden_channels), dtype=torch.float64)


def get_edge_indexes_fully_connected(num_nodes, device):
    idx = torch.arange(num_nodes, device=device, dtype=torch.long)
    edge_index = torch.cartesian_prod(idx, idx).t()
    return edge_index[:, edge_index[0] != edge_index[1]]


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience  = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter   = 0

    def __call__(self, model, val_loss, path):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter   = 0
            torch.save(model.state_dict(), path)
            print(f"✅  Mejor modelo guardado (val_loss: {val_loss:.4f})")
            return False
        self.counter += 1
        print(f"⚠️  Sin mejora en val_loss: {self.counter}/{self.patience} | Mejor: {self.best_loss:.4f} | Actual: {val_loss:.4f}")
        if self.counter >= self.patience:
            import os
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location='cpu'))
                print(f"🔄 Modelo restaurado al mejor estado (val_loss: {self.best_loss:.4f})")
            return True
        return False
