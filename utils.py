import random
import numpy as np
from config import torch, sites
from sklearn.decomposition import PCA

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def z_score_norm(rois_time_series: dict):
    normalized_time_series_by_site = {}

    for site in sites:
        normalized_time_series_by_site[site] = []
        n_subjects_in_site = len(rois_time_series[site])
        site_time_series = []
        for i in range(n_subjects_in_site):
            site_time_series.append(rois_time_series[site][i])

        concat_series = np.concatenate(site_time_series)
        mean = concat_series.mean(axis=0)
        std = concat_series.std(axis=0)
        std[std == 0] = 1

        for ts in site_time_series:
            ts_norm_for_sub_in_site = (ts - mean) / std
            normalized_time_series_by_site[site].append(ts_norm_for_sub_in_site)

    return normalized_time_series_by_site

def reduce_dimensionality(rois_time_series:dict, n_components:int):
    reduced_time_series_by_site = {}

    for site in sites:
        pca = PCA(n_components= n_components)
        reduced_time_series_by_site[site] = []
        n_subjects_in_site = len(rois_time_series[site])
        site_time_series = []
        for i in range(n_subjects_in_site):
            site_time_series.append(rois_time_series[site][i])

        concat_series = np.concatenate(site_time_series)
        pca.fit(concat_series)
        for ts in site_time_series:
            reduced_time_series_by_site[site].append(pca.transform(ts))

    return reduced_time_series_by_site


def create_starting_hidden_state_graph(num_nodes, batch_size, hidden_channels):
    return torch.randn(num_nodes * batch_size, hidden_channels, dtype=torch.float64) * 0.01

def create_starting_cell_state(num_nodes: int,batch_size:int, hidden_channels: int):
    return torch.randn(num_nodes * batch_size, hidden_channels, dtype=torch.float64) * 0.01


def get_edge_indexes_fully_connected(num_nodes, device):
    idx = torch.arange(num_nodes, device=device, dtype=torch.long)
    edge_index = torch.cartesian_prod(idx, idx).t()
    return edge_index[:, edge_index[0] != edge_index[1]]

def get_edge_indexes_sparse(dfc_matrix, percent, device):
    n = dfc_matrix.shape[0]
    k = int(n * percent)

    flattened = torch.flatten(dfc_matrix)
    _, indices = torch.topk(flattened, k=k * 2)

    rows = indices // n
    cols = indices % n

    edge_index = torch.stack([rows, cols]).to(device)
    
    return edge_index


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
