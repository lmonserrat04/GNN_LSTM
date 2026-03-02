from pathlib import Path
import pandas as pd

pipeline = 'cpac'
rois = 'rois_cc200'
phenotypic = 'all_cases'

#BASE_DIR = Path.cwd()

BASE_DIR = Path.cwd()
data_path = BASE_DIR / "ABIDE_pcp" / "cpac" / "filt_noglobal" / "cc200"
csv_path = data_path / "data.csv"

print(data_path)

num_nodes = 200
num_node_features = 200
test_site = 'YALE'

