from config import sites, df
from data_loader import load_rois_data
import torch
import numpy as np
from sklearn.covariance import LedoitWolf
from config import data_path, df

def create_dfc_matrix(time_fmri_series, window_size=40, step=10):
    """
    Función para crear matrices de conectividad funcional dinámica (DFC) para cada sujeto.

    Parámetros:
    -----------
    time_fmri_series : lista o array de numpy
        Serie temporal de un sujeto de shape (nro_tiempos, nro_rois).
        

    window_size : int, opcional (default=40)
        Tamaño de la ventana deslizante en cantidad de tiempos.

    step : int, opcional (default=10)
        Desplazamiento en número de tiempos entre ventanas consecutivas.

    Salida:
    -------
    tupla:
        subjects : lista
            Lista de matrices
            de correlación regularizada de tamaño (nro_rois, nro_rois).
            Si se convierte a numpy array, el shape resultante sería
            (nro_matrices, nro_rois, nro_rois).

        nro_windows: cantidad de ventanas   

    Notas:
    ------
    - La covarianza se estima usando regularización de Ledoit-Wolf.
    - A partir de esta covarianza se calcula una matriz de correlación.
    - La diagonal principal se fuerza a cero.
    """

    
    # Obtener los datos del sujeto actual
    subject_data = time_fmri_series
    
    # Convertir a array numpy si no lo es ya
    if not isinstance(subject_data, np.ndarray):
        subject_data = np.array(subject_data)
        
    n_time = subject_data.shape[0]

    

    subject_matrixes = []
    
    for position in range(0, n_time - window_size + 1, step):
        # Extraer la ventana de datos
        X = subject_data[position:position + window_size, :]

        # Calcular covarianza regularizada
        C_reg = LedoitWolf().fit(X).covariance_

        # Calcular matriz de correlación
        D = np.sqrt(np.diag(C_reg))
        R = C_reg / np.outer(D, D)
        np.fill_diagonal(R, 0)
        subject_matrixes.append(torch.from_numpy((R)))
       


    return subject_matrixes


rois_time_series, rois_labels= load_rois_data(sites, df, data_path)



lw_matrixes = []
for site in sites:
    subjects_in_site = int((df['SITE_ID'] == site).sum())

    for i in range(subjects_in_site):
        matrixes_for_subject = create_dfc_matrix(rois_time_series[site][i])
        lw_matrixes.append(matrixes_for_subject)

save_path = data_path / 'lw_matrixes.pt'
torch.save(lw_matrixes,save_path)
   