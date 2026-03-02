from config import sites, df, data_path
from data_loader import load_rois_data
import torch
import numpy as np
from sklearn.covariance import LedoitWolf
from tqdm import tqdm


def create_dfc_matrix(time_fmri_series, window_size=40, step=10):
    subject_data = time_fmri_series
    if not isinstance(subject_data, np.ndarray):
        subject_data = np.array(subject_data)

    n_time = subject_data.shape[0]
    subject_matrixes = []

    for position in range(0, n_time - window_size + 1, step):
        X     = subject_data[position:position + window_size, :]
        C_reg = LedoitWolf().fit(X).covariance_
        D     = np.sqrt(np.diag(C_reg))
        R     = C_reg / np.outer(D, D)
        np.fill_diagonal(R, 0)
        subject_matrixes.append(torch.from_numpy(R))

    return subject_matrixes


print("Cargando datos de ROIs...")
rois_time_series, rois_labels = load_rois_data(sites, df, data_path)

# Contar total de sujetos para la barra de progreso
total_subjects = sum(int((df['SITE_ID'] == site).sum()) for site in sites)
print(f"Total de sujetos a procesar: {total_subjects}\n")

lw_matrixes = []
errores     = []

with tqdm(total=total_subjects, desc="Creando matrices DFC", unit="sujeto") as pbar:
    for site in sites:
        subjects_in_site = int((df['SITE_ID'] == site).sum())

        for i in range(subjects_in_site):
            pbar.set_postfix(site=site, sujeto=i+1)

            try:
                matrixes = create_dfc_matrix(rois_time_series[site][i])
                lw_matrixes.append(matrixes)
            except Exception as e:
                errores.append((site, i, str(e)))
                print(f"\n⚠️  Error en {site} sujeto {i}: {e}")
                lw_matrixes.append([])  # placeholder para no romper el índice

            pbar.update(1)

# Guardar
save_path = data_path / 'lw_matrixes.pt'
print(f"\n💾 Guardando en {save_path}...")
torch.save(lw_matrixes, save_path)

print(f"✅ Guardado: {len(lw_matrixes)} sujetos")
if errores:
    print(f"⚠️  {len(errores)} errores: {errores}")