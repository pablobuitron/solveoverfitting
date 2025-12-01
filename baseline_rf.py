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
    Build the "long" DataFrame (one row per group_id, time_idx)
    using the same DatasetBuilder as the TFT, but without creating a TimeSeriesDataSet.

    Then perform a 70/30 train/validation split grouping by (year, field),
    i.e. by "field-year", to mimic the grouped split used in the TFT experiment.
    """

    # 1) Load the time series dataset config
    ts_cfg = basic_file_utils.load_yaml(TS_CFG_PATH)

    # 2) Build the dataset index from index.csv
    index_file = DATASET_ROOT / "index.csv"
    ds_idx = dataset_index.DatasetIndex(index_file)

    # 3) Create the DatasetBuilder without predefined splits
    #    (we'll handle the train/val split manually here).
    builder = timeseries_dataset.DatasetBuilder(
        root_dir=DATASET_ROOT,
        ds_index=ds_idx,
        ds_splits=None,
        ds_splits_generator=None,
        configuration=ts_cfg,
        quick_debug=False,
    )

    # 4) Build the "long" DataFrame
    #    This is essentially what builder.build() does, but without
    #    creating splits or a TimeSeriesDataSet object.
    data = builder._build_dataframe()

    # Close HDF5 caches, just like builder.build() would do
    builder._static_data_cache.close_files()
    builder._dynamic_data_cache.close_files()

    # Final preprocessing (type casting, removing weird NaNs,
    # mapping group_id to ints, etc.).
    # backup_group_id_col=True -> creates COLUMN_TMP_GROUP_ID_STR
    # with the original string group_id.
    data = builder._preprocess_final_dataframe(data, backup_group_id_col=True)

    # 5) Build a "field-year" identifier from the backup column
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


def make_aggregated_features(df: pd.DataFrame, ts_cfg: dict):
    """
    Starting from the "long" DataFrame (group_id, time_idx, target, features),
    build a "per group_id" DataFrame with:

      - static_categoricals (as is: one value per group_id)
      - static_reals
      - aggregated (mean, max, min, std) time_varying_unknown_real features
      - a single target (yield) per group_id

    Returns: X (features), y (target), and the aggregated DataFrame for inspection.
    """
    gid_col = timeseries_dataset.COLUMN_GROUP_ID
    target_col = timeseries_dataset.COLUMN_TARGET

    group = df.groupby(gid_col)

    # Target: assumed constant per group_id -> take the first value
    target = group[target_col].first()

    # Static features
    static_cats = ts_cfg["static_categoricals"]
    static_reals = ts_cfg["static_reals"]
    static_cols = [c for c in static_cats + static_reals if c in df.columns]

    if static_cols:
        static_part = group[static_cols].first()
    else:
        static_part = pd.DataFrame(index=target.index)

    # Time-varying features
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

    # Put everything together
    full = pd.concat([static_part, dyn_part], axis=1)
    full[target_col] = target

    # Basic NaN cleaning (simple but good enough for a baseline)
    full = full.dropna(axis=0)

    y = full[target_col].values
    X = full.drop(columns=[target_col])

    return X, y, full


def main():
    print("Building long DataFrame and grouped split by field-year...")
    train_df, val_df, ts_cfg = build_long_df_and_grouped_split()

    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    print("Aggregating features per group_id (point within a field)...")
    X_train, y_train, agg_train = make_aggregated_features(train_df, ts_cfg)
    X_val, y_val, agg_val = make_aggregated_features(val_df, ts_cfg)

    print(f"Train groups: {X_train.shape[0]}, features: {X_train.shape[1]}")
    print(f"Val groups:   {X_val.shape[0]}, features: {X_val.shape[1]}")

    print("Training RandomForestRegressor baseline (RMSE on unseen field-years)...")
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=3,
        n_jobs=-1,          # use all available cores
        random_state=42,
    )
    rf.fit(X_train, y_train)

    preds_val = rf.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, preds_val))

    print("\n=== RANDOM FOREST BASELINE (grouped by field-year) ===")
    print(f"RMSE_val: {rmse_val:.4f}")
    print("(For reference: TFT grouped ~ 4.02 in experiment 1851)")


if __name__ == "__main__":
    main()
