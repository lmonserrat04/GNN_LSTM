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
from utils import z_score_norm

"""
# Solo sitios de entrenamiento → guarda lw_matrixes.pt
python create_matrixes.py --modo train

# Solo test site → guarda lw_matrixes_test.pt
python create_matrixes.py --modo test

# Todos → guarda lw_matrixes_all.pt
python create_matrixes.py --modo all

#Dummy dataset
python create_matrixes.py --modo dummy
"""


def create_dfc_matrixes(time_fmri_series: np.ndarray, window_size=40, step=10):
    subject_data: np.ndarray = np.array(time_fmri_series, dtype=np.float32) if not isinstance(time_fmri_series, np.ndarray) else time_fmri_series.astype(np.float32)
    n_time       = subject_data.shape[0]

    lw_matrixes_for_sub     = []
    nodal_features_for_sub = []

    for position in range(0, n_time - window_size + 1, step):
        #Hacer ventana
        X     = subject_data[position:position + window_size, :]
        
        
        # if position == 0:
        #     print(f"Shape original{X.shape}")
        #     print(X.T.shape)

        #Features
        nodal_features_for_sub.append(X.T)


        #lw
        C_reg = LedoitWolf().fit(X).covariance_
        D     = np.sqrt(np.diag(C_reg))
        R     = C_reg / np.outer(D, D)
        np.fill_diagonal(R, 0)
        lw_matrixes_for_sub.append(R.astype(np.float32))  # float32 desde el inicio

    return lw_matrixes_for_sub, nodal_features_for_sub


def create_matrixes(target_sites, save_filename):
    print(f"Sitios a procesar: {target_sites}")
    print("Cargando datos de ROIs...")
    rois_time_series, _ = load_rois_data(target_sites, df, data_path)
    rois_time_series = z_score_norm(rois_time_series)

    tasks = []
    for site in target_sites:
        subjects_in_site = int((df['SITE_ID'] == site).sum())
        for i in range(subjects_in_site):
            tasks.append((site, i, rois_time_series[site][i]))

    total = len(tasks)
    save_path = data_path / save_filename
    print(f"Total sujetos: {total} | Procesamiento secuencial para ahorrar RAM\n")

    # Procesar y guardar secuencialmente — sin multiprocessing
    results_tensors = []
    errores = []

    for idx, (site, i, time_series) in enumerate(tqdm(tasks, desc="Creando matrices DFC", unit="sujeto")):
        try:
            lw_matrixes_for_sub, nodal_features_for_sub = create_dfc_matrixes(time_series)

            tensors_lw = [torch.from_numpy(m) for m in lw_matrixes_for_sub]
            tensors_nodal = [torch.from_numpy(f) for f in nodal_features_for_sub]

            results_tensors.append((tensors_lw, tensors_nodal))

            # Liberar memoria inmediatamente
            del lw_matrixes_for_sub,nodal_features_for_sub, tensors_lw,tensors_nodal, time_series
            tasks[idx] = (site, i, None)  # liberar la serie temporal del task

        except Exception as e:
            errores.append((idx, str(e)))
            results_tensors.append([])
            print(f"\n⚠️ Error sujeto {idx}: {e}")

    print(f"\n💾 Guardando en {save_path}...")
    torch.save(results_tensors, save_path)

    print(f"✅ Guardado: {len(results_tensors)} sujetos en {save_filename}")
    if errores:
        print(f"⚠️ {len(errores)} errores: {errores}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modo",
        choices=["train", "test", "all","dummy"],
        default="train",
        help="train = sitios de entrenamiento | test = test_site | all = todos | dummy = Un solo sitio"
    )
    args = parser.parse_args()

    if args.modo == "train":
        create_matrixes(list(sites), "lw_matrixes.pt")
    elif args.modo == "test":
        create_matrixes([test_site], "lw_matrixes_test.pt")
    elif args.modo == "all":
        create_matrixes(list(sites) + [test_site], "lw_matrixes_all.pt")
    elif args.modo == "dummy":
        create_matrixes(['SBL'], "lw_matrixes_dummy.pt")