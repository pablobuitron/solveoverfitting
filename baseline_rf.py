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

# Reuse the same (global) config used for the time series dataset
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
    Build the 'long' DataFrame (one row per group_id, time_idx)
    using the same DatasetBuilder as the TFT, but without creating a TimeSeriesDataSet.

    Then perform a 70/30 train/validation split grouping by (year, field),
    i.e. by 'field-year', to mimic the grouped split used in the TFT experiment.
    """

    # 1) Load the time series dataset config
    ts_cfg = basic_file_utils.load_yaml(TS_CFG_PATH)

    # 2) Build the dataset index from index.csv
    index_file = DATASET_ROOT / "index.csv"
    ds_idx = dataset_index.DatasetIndex(index_file)

    # 3) Create the DatasetBuilder without predefined splits
    builder = timeseries_dataset.DatasetBuilder(
        root_dir=DATASET_ROOT,
        ds_index=ds_idx,
        ds_splits=None,
        ds_splits_generator=None,
        configuration=ts_cfg,
        quick_debug=False,
    )

    # 4) Build the 'long' DataFrame
    data = builder._build_dataframe()

    # Final preprocessing (type casting, removing weird NaNs,
    # mapping group_id to ints, etc.).
    data = builder._preprocess_final_dataframe(data, backup_group_id_col=True)

    # 5) Build a 'field-year' identifier from the backup column
    tmp_col = timeseries_dataset.COLUMN_TMP_GROUP_ID_STR
    splitter = dataset_index.ID_SPLITTER  # usually '|'

    # tmp_group_id_str looks like "2022|field_1|point_3"
    parts = data[tmp_col].str.split(splitter, n=2, expand=True)
    # field-year: "2022|field_1"
    data["fieldyear_group"] = parts[0] + splitter + parts[1]

    # 6) Train/validation split (70/30) grouped by field-year (fieldyear_group)
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


def _compute_slope(series: pd.Series) -> float:
    """
    Compute a simple linear slope over the sequence index (0..len-1).
    If the series is constant or too short, return 0.0.
    """
    n = len(series)
    if n < 2:
        return 0.0

    y = series.values
    if np.allclose(y, y[0]):
        return 0.0

    x = np.arange(n, dtype=float)
    a, _ = np.polyfit(x, y, 1)  # y = a*x + b
    return float(a)


def make_aggregated_features(df: pd.DataFrame, ts_cfg: dict):
    """
    'Versión simple' de features:

      - static_categoricals + static_reals
      - time_varying_unknown_reals -> mean, max, min, std, last, slope
      - target: median por group_id

    Devuelve X (DataFrame), y (np.array), y full (DataFrame con target).
    """
    gid_col = timeseries_dataset.COLUMN_GROUP_ID
    target_col = timeseries_dataset.COLUMN_TARGET

    group = df.groupby(gid_col)

    # Target: median para reducir outliers
    target = group[target_col].median()

    # Static features
    static_cats = ts_cfg.get("static_categoricals", [])
    static_reals = ts_cfg.get("static_reals", [])
    static_cols = [c for c in (static_cats + static_reals) if c in df.columns]

    if static_cols:
        static_part = group[static_cols].first()
    else:
        static_part = pd.DataFrame(index=target.index)

    # Time-varying features
    tv_cols = [c for c in ts_cfg.get("time_varying_unknown_reals", []) if c in df.columns]

    agg_frames = []
    for col in tv_cols:
        g = group[col]

        means = g.mean()
        maxs = g.max()
        mins = g.min()
        stds = g.std()
        lasts = g.last()
        slopes = g.apply(_compute_slope)

        agg = pd.DataFrame(
            {
                f"{col}_mean": means,
                f"{col}_max": maxs,
                f"{col}_min": mins,
                f"{col}_std": stds,
                f"{col}_last": lasts,
                f"{col}_slope": slopes,
            }
        )
        agg_frames.append(agg)

    if agg_frames:
        dyn_part = pd.concat(agg_frames, axis=1)
    else:
        dyn_part = pd.DataFrame(index=target.index)

    # Put everything together
    full = pd.concat([static_part, dyn_part], axis=1)
    full[target_col] = target

    # Limpieza básica
    full = full.dropna(axis=0)

    X = full.drop(columns=[target_col])
    y = full[target_col].values

    return X, y, full


def main():
    print("Building long DataFrame and grouped split by field-year...")
    train_df, val_df, ts_cfg = build_long_df_and_grouped_split()

    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    print("Aggregating SIMPLE features per group_id...")
    X_train, y_train, _ = make_aggregated_features(train_df, ts_cfg)
    X_val, y_val, _ = make_aggregated_features(val_df, ts_cfg)

    print(f"Train groups: {X_train.shape[0]}, features: {X_train.shape[1]}")
    print(f"Val groups:   {X_val.shape[0]}, features: {X_val.shape[1]}")

    # Pequeño grid de hiperparámetros razonables
    base_params = dict(
        n_estimators=1000,
        n_jobs=-1,
        random_state=403,
        bootstrap=True,
    )

    configs = [
        # max_depth, min_leaf, min_split, max_features, max_samples
        dict(max_depth=9,  min_samples_leaf=8,  min_samples_split=20, max_features="sqrt", max_samples=0.7),
        dict(max_depth=9,  min_samples_leaf=10, min_samples_split=20, max_features="sqrt", max_samples=0.7),
        dict(max_depth=10, min_samples_leaf=8,  min_samples_split=20, max_features="sqrt", max_samples=0.7),
        dict(max_depth=10, min_samples_leaf=8,  min_samples_split=30, max_features="sqrt", max_samples=0.7),
        dict(max_depth=10, min_samples_leaf=10, min_samples_split=30, max_features="sqrt", max_samples=0.6),
        dict(max_depth=11, min_samples_leaf=8,  min_samples_split=20, max_features="log2", max_samples=0.7),
        dict(max_depth=11, min_samples_leaf=10, min_samples_split=20, max_features="log2", max_samples=0.7),
        dict(max_depth=12, min_samples_leaf=10, min_samples_split=40, max_features="log2", max_samples=0.6),
    ]

    results = []

    from copy import deepcopy
    from sklearn.ensemble import RandomForestRegressor

    for i, cfg in enumerate(configs, start=1):
        params = deepcopy(base_params)
        params.update(cfg)

        print("\n--------------------------------------------------")
        print(f"Config {i}/{len(configs)}: {params}")
        rf = RandomForestRegressor(**params)

        rf.fit(X_train, y_train)

        y_train_pred = rf.predict(X_train)
        y_val_pred = rf.predict(X_val)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

        print(f"  RMSE_train = {rmse_train:.4f}")
        print(f"  RMSE_val   = {rmse_val:.4f}")

        results.append((rmse_val, rmse_train, params))

    # Ordenar por RMSE_val
    results.sort(key=lambda x: x[0])

    print("\n=========== SUMMARY (sorted by RMSE_val) ===========")
    for rmse_val, rmse_train, params in results:
        print(f"RMSE_val={rmse_val:.4f} | RMSE_train={rmse_train:.4f} | params={params}")


if __name__ == "__main__":
    main()
