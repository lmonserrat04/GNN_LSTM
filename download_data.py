import requests
from pathlib import Path
import pandas as pd

BASE_DIR = Path.cwd()
fixed_path = "ABIDE_pcp" / "cpac" / "filt_noglobal" / "cc200"
df1 = pd.read_csv(fixed_path / "data.csv")


# Carpeta donde quieres guardar los archivos
dest_folder = Path("C:/Users/marle/temp_abide/ABIDE_pcp/cpac/filt_noglobal")
dest_folder.mkdir(parents=True, exist_ok=True)

problematic_subjects = []

# Iterar por cada subject en el DataFrame
for idx, subject in df['FILE_ID'].items():
    if subject == 'no_filename':
        continue
    
   
    
    # Nombre del archivo destino
    filename = f"{subject}_rois_cc200.1D"
    dest_path = dest_folder / filename

    if dest_path.exists():
        print(f"⏭ Ya existe: {dest_path}")
        continue

    # Construir la URL
    url = f'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_cc200/{subject}_rois_cc200.1D'
    
    try:
        # Descargar en chunks (mejor para archivos grandes)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:  # evitar keep-alive
                        f.write(chunk)
        print(f"✔ Descargado: {dest_path}")
    except Exception as e:
        print(f"❌ Error con {url}: {e}, sujeto con id: {subject}")
        problematic_subjects.append((idx,subject))
