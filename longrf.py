import pathlib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupShuffleSplit

# Imports from GranoExperiments
from GranoExperiments.grano_experiments.src.data import (
    dataset_index,
    timeseries_dataset,
)
from GranoExperiments.grano_experiments.src.utils import basic_file_utils

ROOT = pathlib.Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "grano_it_dataset"

TS_CFG_PATH = (
    ROOT
    / "GranoExperiments"
    / "grano_experiments"
    / "src"
    / "config"
    / "timeseries_dataset_config.yaml"
)


def build_long_df_and_grouped_split_timeseries():
    """
    Versión 'long': NO agregamos por group_id.
    Trabajamos directamente con (group_id, time_idx).

    Hacemos el split 70/30 agrupado por field-year ('fieldyear_group'),
    como en los otros scripts, para que no haya leakage entre campos/años.
    """

    # 1) Load config
    ts_cfg = basic_file_utils.load_yaml(TS_CFG_PATH)

    # 2) Dataset index
    index_file = DATASET_ROOT / "index.csv"
    ds_idx = dataset_index.DatasetIndex(index_file)

    # 3) DatasetBuilder sin splits predefinidos
    builder = timeseries_dataset.DatasetBuilder(
        root_dir=DATASET_ROOT,
        ds_index=ds_idx,
        ds_splits=None,
        ds_splits_generator=None,
        configuration=ts_cfg,
        quick_debug=False,
    )

    # 4) long dataframe: (group_id, time_idx, features, target, etc.)
    data = builder._build_dataframe()

    # 5) preprocess
    data = builder._preprocess_final_dataframe(data, backup_group_id_col=True)

    # 6) añadir fieldyear_group como antes
    tmp_col = timeseries_dataset.COLUMN_TMP_GROUP_ID_STR
    splitter = dataset_index.ID_SPLITTER  # usually '|'

    # tmp_group_id_str: "year|field|point"
    parts = data[tmp_col].str.split(splitter, n=2, expand=True)
    data["fieldyear_group"] = parts[0] + splitter + parts[1]

    # 7) split 70/30 por fieldyear_group
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


def build_features_timeseries(df: pd.DataFrame, ts_cfg: dict):
    """
    Construye X, y sobre el long dataframe, usando:

      - static_categoricals
      - static_reals
      - time_varying_known_reals
      - time_varying_unknown_reals

    El target es COLUMN_TARGET (típicamente yield por group_id, repetido en el tiempo).
    """
    gid_col = timeseries_dataset.COLUMN_GROUP_ID
    target_col = timeseries_dataset.COLUMN_TARGET

    # Definimos columnas de features agregando todo lo relevante de ts_cfg
    feature_cols = []
    for key in [
        "static_categoricals",
        "static_reals",
        "time_varying_known_reals",
        "time_varying_unknown_reals",
    ]:
        cols = ts_cfg.get(key, [])
        # Nos quedamos solo con las que de verdad existen en df
        for c in cols:
            if c in df.columns and c not in feature_cols:
                feature_cols.append(c)

    if not feature_cols:
        raise RuntimeError("No se encontraron columnas de features en el DataFrame.")

    # X: todas las features numéricas/categóricas codificadas en df
    X = df[feature_cols].copy()

    # y: target a nivel de fila (misma y para todas las filas del mismo group_id)
    y = df[target_col].values

    return X, y, feature_cols


def main():
    print("Building LONG time-series DataFrame and grouped split by field-year...")
    train_df, val_df, ts_cfg = build_long_df_and_grouped_split_timeseries()

    print(f"Train rows (time steps): {len(train_df)}")
    print(f"Val rows   (time steps): {len(val_df)}")

    print("Building features on LONG dataframe (no aggregation by group_id)...")
    X_train, y_train, feature_cols = build_features_timeseries(train_df, ts_cfg)
    X_val, y_val, _ = build_features_timeseries(val_df, ts_cfg)

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val   shape: X={X_val.shape}, y={y_val.shape}")

    print("\nTraining HIGH-CAPACITY RandomForestRegressor on time steps...")
    rf = RandomForestRegressor(
        n_estimators=1200,
        max_depth=None,        # sin límite de profundidad
        min_samples_leaf=1,
        min_samples_split=2,
        max_features="sqrt",   # clásico RF
        bootstrap=True,
        max_samples=1.0,       # cada árbol ve todo el train (para máxima capacidad)
        n_jobs=-1,
        random_state=403,
    )

    rf.fit(X_train, y_train)

    # Predicciones a nivel de fila (time step)
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

    print("\n=== RANDOM FOREST TIMESERIES (per-row RMSE) ===")
    print(f"RMSE_train (per row): {rmse_train:.4f}")
    print(f"RMSE_val   (per row): {rmse_val:.4f}")
    print("Notas:")
    print(" - El split sigue siendo por field-year (no hay leakage).")
    print(" - La métrica es RMSE por fila del long_df.")
    print(" - Esto es más cercano a cómo típicamente se evalúan modelos tipo LSTM.")

    # (Opcional) Importancia de features
    try:
        importances = rf.feature_importances_
        feature_names = np.array(feature_cols)
        idx = np.argsort(importances)[::-1][:20]

        print("\nTop 20 feature importances (timeseries RF):")
        for i in idx:
            print(f"  {feature_names[i]:40s}  {importances[i]:.4f}")
    except Exception as e:
        print(f"\n[Warning] Could not compute feature importances: {e}")


if __name__ == "__main__":
    main()
