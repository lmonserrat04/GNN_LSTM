from config import sites, df, data_path
from data_loader import load_rois_data
import torch
import numpy as np
from sklearn.covariance import LedoitWolf
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def create_dfc_matrix(time_fmri_series, window_size=40, step=10):
    subject_data = np.array(time_fmri_series) if not isinstance(time_fmri_series, np.ndarray) else time_fmri_series
    n_time       = subject_data.shape[0]
    matrixes     = []

    for position in range(0, n_time - window_size + 1, step):
        X     = subject_data[position:position + window_size, :]
        C_reg = LedoitWolf().fit(X).covariance_
        D     = np.sqrt(np.diag(C_reg))
        R     = C_reg / np.outer(D, D)
        np.fill_diagonal(R, 0)
        matrixes.append(torch.from_numpy(R))

    return matrixes


def _worker(args):
    """Función que corre en cada proceso paralelo."""
    idx, site, i, time_series = args
    try:
        matrixes = create_dfc_matrix(time_series)
        return idx, matrixes, None
    except Exception as e:
        return idx, [], str(e)


if __name__ == "__main__":
    print("Cargando datos de ROIs...")
    rois_time_series, rois_labels = load_rois_data(sites, df, data_path)

    # Construir lista de tareas: (índice_global, site, i, time_series)
    tasks = []
    idx   = 0
    for site in sites:
        subjects_in_site = int((df['SITE_ID'] == site).sum())
        for i in range(subjects_in_site):
            tasks.append((idx, site, i, rois_time_series[site][i]))
            idx += 1

    total   = len(tasks)
    n_cores = cpu_count()
    print(f"Total sujetos: {total} | Usando {n_cores} cores\n")

    # Resultado final ordenado por índice
    results = [None] * total
    errores = []

    with Pool(processes=7) as pool:
        with tqdm(total=total, desc="Creando matrices DFC", unit="sujeto") as pbar:
            for idx, matrixes, error in pool.imap_unordered(_worker, tasks):
                results[idx] = matrixes
                if error:
                    errores.append((idx, error))
                    print(f"\n⚠️  Error sujeto {idx}: {error}")
                pbar.update(1)

    # Guardar
    save_path = data_path / 'lw_matrixes.pt'
    print(f"\n💾 Guardando en {save_path}...")
    torch.save(results, save_path)

    print(f"✅ Guardado: {len(results)} sujetos")
    if errores:
        print(f"⚠️  {len(errores)} errores")