from pathlib import Path
import pandas as pd
import numpy as np

pipeline = 'cpac'
rois = 'rois_cc200'
phenotypic = 'all_cases'

#BASE_DIR = Path.cwd()

BASE_DIR = Path.cwd()
data_path = BASE_DIR / "ABIDE_pcp" / "cpac" / "filt_noglobal" / "cc200"
csv_path = data_path / "data.csv"

df         = pd.read_csv(data_path / "data.csv")
test_site = 'YALE'
sites      = df['SITE_ID'].unique()

idxs_train = (sites != test_site)
sites  = sites[idxs_train]
#sites = ['SBL']

num_nodes = 200
num_node_features = 200
#print(sites)

