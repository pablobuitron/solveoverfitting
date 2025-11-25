import pathlib

import h5py  # si no está instalado, te digo cómo instalarlo
from pprint import pprint

path = pathlib.Path("grano_it_dataset/dynamic_features")

# Coge el primer archivo dinámico que encuentres
files = sorted(path.glob("**/*"))
print("Encontrados archivos dinámicos:")
for f in files[:5]:
    print("  -", f)

if not files:
    print("⚠️ No hay archivos en grano_it_dataset/dynamic")
    raise SystemExit

f0 = files[0]
print("\nInspeccionando:", f0)

with h5py.File(f0, "r") as h:
    print("Keys en el HDF5:")
    pprint(list(h.keys()))
    # intenta ver si hay algo tipo 'features_names' o similar
    for k in h.keys():
        obj = h[k]
        if hasattr(obj, "shape"):
            print(f"  {k}: shape={obj.shape}, dtype={obj.dtype}")
