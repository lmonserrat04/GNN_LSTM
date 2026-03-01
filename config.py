from pathlib import Path
import pandas as pd

pipeline = 'cpac'
rois = 'rois_cc200'
phenotypic = 'all_cases'

#BASE_DIR = Path.cwd()

BASE_DIR = Path(r"C:\Users\marle\Desktop\AI\AENTA\Modelo en desarrollo")
data_path = BASE_DIR / "ABIDE_pcp" / "cpac" / "filt_noglobal"
csv_path = data_path / "data.csv"


num_nodes = 200
num_node_features = 200
test_site = 'YALE'

