import os
os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["MKL_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
from config import sites, df, data_path, test_site
from data_loader import load_rois_data
import torch
import numpy as np
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from multiprocessing import Pool
from utils import z_score_norm

"""
# Solo sitios de entrenamiento → guarda lw_matrixes.pt
python create_matrixes.py --modo train

# Solo test site → guarda lw_matrixes_test.pt
python create_matrixes.py --modo test

# Todos → guarda lw_matrixes_all.pt
python create_matrixes.py --modo all

"""


def create_dfc_matrix(time_fmri_series, window_size=40, step=10):
    subject_data = np.array(time_fmri_series) if not isinstance(time_fmri_series, np.ndarray) else time_fmri_series
    subject_data = z_score_norm(subject_data)
    n_time       = subject_data.shape[0]
    matrixes     = []

    for position in range(0, n_time - window_size + 1, step):
        X     = subject_data[position:position + window_size, :]
        C_reg = LedoitWolf().fit(X).covariance_
        D     = np.sqrt(np.diag(C_reg))
        R     = C_reg / np.outer(D, D)
        np.fill_diagonal(R, 0)
        matrixes.append(R)

    return matrixes


def _worker(args):
    idx, time_series = args
    try:
        return idx, create_dfc_matrix(time_series), None
    except Exception as e:
        return idx, [], str(e)


def create_matrixes(target_sites, save_filename):
    print(f"Sitios a procesar: {target_sites}")
    print("Cargando datos de ROIs...")
    rois_time_series, rois_labels = load_rois_data(target_sites, df, data_path)

    tasks = []
    idx   = 0
    for site in target_sites:
        subjects_in_site = int((df['SITE_ID'] == site).sum())
        for i in range(subjects_in_site):
            tasks.append((idx, rois_time_series[site][i]))
            idx += 1

    total = len(tasks)
    print(f"Total sujetos: {total} | Usando 6 cores\n")

    results = [None] * total
    errores = []

    with Pool(processes=6) as pool:
        with tqdm(total=total, desc="Creando matrices DFC", unit="sujeto") as pbar:
            for idx, matrixes, error in pool.imap_unordered(_worker, tasks):
                results[idx] = matrixes
                if error:
                    errores.append((idx, error))
                    print(f"\n⚠️  Error sujeto {idx}: {error}")
                pbar.update(1)

    print("\n🔄 Convirtiendo a tensores...")
    results_tensors = [
        [torch.from_numpy(m) for m in subject_matrixes]
        for subject_matrixes in tqdm(results, desc="Convirtiendo", unit="sujeto")
    ]

    save_path = data_path / save_filename
    print(f"\n💾 Guardando en {save_path}...")
    torch.save(results_tensors, save_path)

    print(f"✅ Guardado: {len(results_tensors)} sujetos en {save_filename}")
    if errores:
        print(f"⚠️  {len(errores)} errores: {errores}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modo",
        choices=["train", "test", "all"],
        default="train",
        help="train = sitios de entrenamiento | test = test_site | all = todos"
    )
    args = parser.parse_args()

    if args.modo == "train":
        create_matrixes(list(sites), "lw_matrixes.pt")

    elif args.modo == "test":
        create_matrixes([test_site], "lw_matrixes_test.pt")

    elif args.modo == "all":
        create_matrixes(list(sites) + [test_site], "lw_matrixes_all.pt")