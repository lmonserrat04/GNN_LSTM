import requests
import shutil
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ─── RUTAS AUTOMÁTICAS (funciona en cualquier PC) ─────────────────────────────
SCRIPT_DIR    = Path(__file__).resolve().parent
fixed_path    = SCRIPT_DIR / "ABIDE_pcp" / "cpac" / "filt_noglobal" / "cc200"
csv_path      = fixed_path / "data.csv"
log_path      = SCRIPT_DIR / "download_errors.txt"

fixed_path.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
df = df[df['FILE_ID'] != 'no_filename']  # filtrar sin filename

total = len(df)
print(f"📋 Total de sujetos: {total}\n")

# ─── DESCARGA Y MOVE DIRECTO A SITE/ASD o SITE/TC ────────────────────────────
problematic_subjects = []

for i, (_, row) in enumerate(df.iterrows(), start=1):
    file_id       = row['FILE_ID']
    site          = row['SITE_ID']
    dx            = row['DX_GROUP']
    clasificacion = "ASD" if dx == 1 else "TC"
    dest_folder   = fixed_path / site / clasificacion
    dest_folder.mkdir(parents=True, exist_ok=True)

    filename  = f"{file_id}_rois_cc200.1D"
    dest_path = dest_folder / filename

    prefix = f"[{i}/{total}]"

    url = (
        "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/"
        f"Outputs/cpac/filt_noglobal/rois_cc200/{filename}"
    )

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))

            with open(dest_path, "wb") as f, tqdm(
                desc=f"{prefix} {site}/{clasificacion}/{file_id}",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

        action = "sobreescrito" if dest_path.exists() else "descargado"
        print(f"✔ {prefix} {site}/{clasificacion}/{filename}")

    except Exception as e:
        print(f"❌ {prefix} Error {file_id}: {e}")
        problematic_subjects.append((file_id, site, clasificacion, str(e)))

# ─── LOG DE ERRORES ───────────────────────────────────────────────────────────
if problematic_subjects:
    with open(log_path, "w") as log:
        log.write(f"Sujetos con error de descarga ({len(problematic_subjects)}):\n\n")
        for file_id, site, clasificacion, error in problematic_subjects:
            log.write(f"  {site}/{clasificacion}/{file_id} → {error}\n")
    print(f"\n⚠ {len(problematic_subjects)} errores. Log guardado en: {log_path}")
else:
    print("\n✅ Todos los sujetos descargados y ordenados correctamente.")