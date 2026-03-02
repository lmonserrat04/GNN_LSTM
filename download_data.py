import requests
import shutil
from pathlib import Path
import pandas as pd

# ─── RUTAS AUTOMÁTICAS (funciona en cualquier PC) ─────────────────────────────
# Busca el data.csv relativo a donde está este script
SCRIPT_DIR    = Path(__file__).resolve().parent
fixed_path    = SCRIPT_DIR / "ABIDE_pcp" / "cpac" / "filt_noglobal" / "cc200"
csv_path      = fixed_path / "data.csv"
log_path      = SCRIPT_DIR / "download_errors.txt"

fixed_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

# ─── DESCARGA Y MOVE DIRECTO A SITE/ASD o SITE/TC ────────────────────────────
problematic_subjects = []

for _, row in df.iterrows():
    file_id = row['FILE_ID']
    site    = row['SITE_ID']
    dx      = row['DX_GROUP']

    if file_id == 'no_filename':
        continue

    # Carpeta destino final según site y diagnóstico
    clasificacion = "ASD" if dx == 1 else "TC"
    dest_folder   = fixed_path / site / clasificacion
    dest_folder.mkdir(parents=True, exist_ok=True)

    filename  = f"{file_id}_rois_cc200.1D"
    dest_path = dest_folder / filename

    # Si ya está en su carpeta correcta, saltear
    if dest_path.exists():
        print(f"⏭ Ya existe: {site}/{clasificacion}/{filename}")
        continue

    url = (
        "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
        f"Outputs/cpac/filt_noglobal/rois_cc200/{filename}"
    )

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print(f"✔ {site}/{clasificacion}/{filename}")
    except Exception as e:
        print(f"❌ Error {file_id}: {e}")
        problematic_subjects.append((file_id, site, clasificacion, str(e)))

# ─── LOG DE ERRORES ───────────────────────────────────────────────────────────
if problematic_subjects:
    with open(log_path, "w") as log:
        log.write(f"Sujetos con error de descarga ({len(problematic_subjects)}):\n\n")
        for file_id, site, clasificacion, error in problematic_subjects:
            line = f"  {site}/{clasificacion}/{file_id} → {error}\n"
            log.write(line)
    print(f"\n⚠ {len(problematic_subjects)} errores. Log guardado en: {log_path}")
else:
    print("\n✅ Todos los sujetos descargados y ordenados correctamente.")