import pathlib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

# Imports del proyecto GranoExperiments
from GranoExperiments.grano_experiments.src.data import (
    dataset_index,
    timeseries_dataset,
)
from GranoExperiments.grano_experiments.src.utils import basic_file_utils

ROOT = pathlib.Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "grano_it_dataset"

# Usamos la misma config global del timeseries dataset
TS_CFG_PATH = (
    ROOT
    / "GranoExperiments"
    / "grano_experiments"
    / "src"
    / "config"
    / "timeseries_dataset_config.yaml"
)


def build_long_df_and_grouped_split():
    """
    Construye el DataFrame "largo" (una fila por group_id, time_idx)
    usando el mismo DatasetBuilder que el TFT, pero sin crear TimeSeriesDataSet.

    Luego hace un split 70/30 train/val agrupando por (year, field),
    es decir por "campo-año", de forma similar al split grouped del experimento.
    """

    # 1) Cargar configuración del timeseries dataset
    ts_cfg = basic_file_utils.load_yaml(TS_CFG_PATH)

    # 2) Construir el índice a partir de index.csv
    index_file = DATASET_ROOT / "index.csv"
    ds_idx = dataset_index.DatasetIndex(index_file)

    # 3) Crear el DatasetBuilder SIN splits (los haremos nosotros)
    builder = timeseries_dataset.DatasetBuilder(
        root_dir=DATASET_ROOT,
        ds_index=ds_idx,
        ds_splits=None,
        ds_splits_generator=None,
        configuration=ts_cfg,
        quick_debug=False,
    )

    # 4) Construir el DataFrame "largo"
    #    Esto es básicamente lo que hace build(), pero sin dividir en splits ni crear TimeSeriesDataSet
    data = builder._build_dataframe()

    # Cerramos caches de HDF5 como hace build()
    builder._static_data_cache.close_files()
    builder._dynamic_data_cache.close_files()

    # Preprocesado final (cast de tipos, quitar NaNs raras, mapear group_id a ints, etc.)
    # backup_group_id_col=True -> crea la columna COLUMN_TMP_GROUP_ID_STR con el ID original en string
    data = builder._preprocess_final_dataframe(data, backup_group_id_col=True)

    # 5) Construir un identificador de "campo-año" a partir de la columna de backup
    tmp_col = timeseries_dataset.COLUMN_TMP_GROUP_ID_STR
    splitter = dataset_index.ID_SPLITTER  # normalmente es '|'

    # tmp_group_id_str tiene algo como "2022|field_1|point_3"
    parts = data[tmp_col].str.split(splitter, n=2, expand=True)
    # campo-año: "2022|field_1"
    data["fieldyear_group"] = parts[0] + splitter + parts[1]

    # 6) Hacer split train/val 70/30 agrupando por campo-año (fieldyear_group)
    groups = data["fieldyear_group"]
    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=0.7,
        test_size=0.3,
        random_state=403,
    )
    train_idx, val_idx = next(gss.split(data, groups=groups))

    train_df = data.iloc[train_idx].copy()
    val_df = data.iloc[val_idx].copy()

    return train_df, val_df, ts_cfg


def make_aggregated_features(df: pd.DataFrame, ts_cfg: dict):
    """
    A partir del DataFrame "largo" (group_id, time_idx, target, features),
    construye un DataFrame "por group_id" con:

      - static_categoricals (tal cual, 1 valor por group_id)
      - static_reals
      - agregados (mean, max, min, std) de cada time_varying_unknown_real
      - target (yield) único por group_id

    Devuelve: X (features), y (target), y el DataFrame agreg. por si quieres mirarlo.
    """
    gid_col = timeseries_dataset.COLUMN_GROUP_ID
    target_col = timeseries_dataset.COLUMN_TARGET

    group = df.groupby(gid_col)

    # Target: constante por group_id -> cogemos el primero
    target = group[target_col].first()

    # Features estáticas
    static_cats = ts_cfg["static_categoricals"]
    static_reals = ts_cfg["static_reals"]
    static_cols = [c for c in static_cats + static_reals if c in df.columns]

    if static_cols:
        static_part = group[static_cols].first()
    else:
        static_part = pd.DataFrame(index=target.index)

    # Features dinámicas
    tv_cols = [c for c in ts_cfg["time_varying_unknown_reals"] if c in df.columns]

    agg_frames = []
    for col in tv_cols:
        g = group[col]
        agg = pd.DataFrame(
            {
                f"{col}_mean": g.mean(),
                f"{col}_max": g.max(),
                f"{col}_min": g.min(),
                f"{col}_std": g.std(),
            }
        )
        agg_frames.append(agg)

    if agg_frames:
        dyn_part = pd.concat(agg_frames, axis=1)
    else:
        dyn_part = pd.DataFrame(index=target.index)

    # Combinar todo
    full = pd.concat([static_part, dyn_part], axis=1)
    full[target_col] = target

    # Limpiar NaNs (muy básico, pero suficiente para el baseline)
    full = full.dropna(axis=0)

    y = full[target_col].values
    X = full.drop(columns=[target_col])

    return X, y, full


def main():
    print("Construyendo DataFrame largo y split grouped por campo-año...")
    train_df, val_df, ts_cfg = build_long_df_and_grouped_split()

    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    print("Agregando features por group_id (punto dentro del campo)...")
    X_train, y_train, agg_train = make_aggregated_features(train_df, ts_cfg)
    X_val, y_val, agg_val = make_aggregated_features(val_df, ts_cfg)

    print(f"Train groups: {X_train.shape[0]}, features: {X_train.shape[1]}")
    print(f"Val groups:   {X_val.shape[0]}, features: {X_val.shape[1]}")

    print("Entrenando RandomForestRegressor baseline (RMSE sobre campos/años no vistos)...")
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        n_jobs=-1,          # usa todos los hilos disponibles
        random_state=42,
    )
    rf.fit(X_train, y_train)

    preds_val = rf.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, preds_val))

    print("\n=== BASELINE RANDOM FOREST (grouped por campo-año) ===")
    print(f"RMSE_val: {rmse_val:.4f}")
    print("(Para comparar: TFT grouped ~ 4.02 en tu experimento 1851)")


if __name__ == "__main__":
    main()
