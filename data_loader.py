import numpy as np
import pandas as pd
from pathlib import Path


def load_rois_data(sites, df, origin_path):
    """
    Loads time series and diagnostic labels from neuroimaging data files for the specified sites.
    """
    print("   ↳ Ejecutando load_rois_data...")
    rois_time_series = {}  # Dictionary to store time series data for each site
    rois_labels = {}  # Dictionary to store labels for each site

    total_of_subjects = 0

    for site in sites:
        #print(f"   ↳ Procesando sitio: {site}")
        site_time_series = []  # List to store time series for each subject at the site
        site_labels = []  # List to store labels for each subject at the site

        sub_df = df[df['SITE_ID'] == site]

        parent_folder = origin_path / site
        asd_child_folder = parent_folder / "ASD"
        tc_child_folder = parent_folder / "TC"

        # Define path for the time series data file
        for file_id in sub_df['FILE_ID']:
            name = file_id + '_rois_cc200.1D'
            filtered = sub_df.loc[sub_df['FILE_ID'] == file_id, :]

            if filtered.loc[:,'DX_GROUP'].iloc[0] == 1:
                data_path = asd_child_folder / name
                if not data_path.exists():
                    print(f"   ⚠️ File Not Found Error: Data file not found at path {data_path}")
                    continue

                data = np.loadtxt(data_path)

                # Check for NaN values and add time series to the site list
                if np.isnan(data).any():
                    print(f"   ⚠️ Value Error: NaN value found for subject {file_id}")
                else:
                    site_time_series.append(data)
                    site_labels.append(1)  # Assign 1 for ASD, 0 for control

            else:
                data_path = tc_child_folder / name
                if not data_path.exists():
                    print(f"   ⚠️ File Not Found Error: Data file not found at path {data_path}")
                    continue

                data = np.loadtxt(data_path)

                # Check for NaN values and add time series to the site list
                if np.isnan(data).any():
                    print(f"   ⚠️ Value Error: NaN value found for subject {file_id}")
                else:
                    site_time_series.append(data)
                    site_labels.append(0)  # Assign 1 for ASD, 0 for control

         # Store loaded data for the current site in the dictionaries
        rois_time_series[site] = site_time_series
        rois_labels[site] = np.array(site_labels)
        loaded_subjects_from_site = len(site_time_series)
        total_of_subjects+= loaded_subjects_from_site
        #print(f"   ✅ Loaded {loaded_subjects_from_site} subjects from site {site}.")

    print(f"✅ Total loaded: {total_of_subjects} subjects")
    print("   ↳ Fin de load_rois_data")
    return rois_time_series, rois_labels
