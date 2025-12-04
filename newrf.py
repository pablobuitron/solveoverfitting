import argparse
import pathlib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Grano.IT pipeline imports
from GranoExperiments.grano_experiments.src.experiments_manager import experiments_manager
from GranoExperiments.grano_experiments.src.data import (
    dataset_index,
    timeseries_dataset,
    dataset_constants,
)
from GranoExperiments.grano_experiments.src.data.splits import generators_factory
from feature_engineering_rf import build_rf_features


ROOT = pathlib.Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "grano_it_dataset"
EXPERIMENTS_ROOT = ROOT / "GranoExperiments" / "grano_experiments" / "experiments"


# =============================================================================
# Load the long dataframe and the exact train/val splits used by the LSTM
# =============================================================================
def build_train_val_from_experiment(exp_dir: pathlib.Path,
                                    dataset_root: pathlib.Path,
                                    exp_name: str | None,
                                    quick_debug: bool = False):
    """
    Loads a Grano.IT experiment (latest or a specific one), and reconstructs:

        - the full long dataframe (points × timesteps × features)
        - the train and validation dataframes using EXACTLY the same splits
          used by the LSTM pipeline.

    Returns:
        train_df, val_df, dataset_config
    """
    # Instantiate ExperimentManager
    mgr = experiments_manager.ExperimentManager(
        experiments_root_dir=str(exp_dir),
        dataset_root_dir=str(dataset_root),
        quick_debug=quick_debug,
    )

    # Load experiment
    if exp_name is None:
        mgr.load_latest()
        print(f"Loaded latest experiment: {mgr.experiment.dir_structure.experiment_root_dir.name}")
    else:
        exp_path = exp_dir / exp_name
        mgr.load(exp_path)
        print(f"Loaded experiment: {exp_name}")
    print("\n=== Split information from experiment ===")
    print("Split generator data:", mgr.experiment.splits_generator_data)
    print("Pre-generated split files:", mgr.experiment.splits_files)
    print("========================================\n")
    # Load dataset / training configs as the original trainer would
    mgr._load_configuration_for_training(training_configuration_override=None)
    mgr._check_configuration_for_training()

    ds_idx = dataset_index.DatasetIndex(mgr.dataset_index_path)

    # Determine whether to load pre-generated split files or generate splits
    if mgr.experiment.splits_files:
        splits = mgr._fetch_splits(
            [dataset_constants.TRAIN_SPLIT_NAME, dataset_constants.VALIDATION_SPLIT_NAME]
        )
        splits_gen = None
        print("Using pre-generated experiment CSV splits.")
    else:
        splits = None
        splits_gen = generators_factory.from_dict(mgr.experiment.splits_generator_data)
        print("Using split generator defined in the experiment configuration.")

    # Build dataset with the same configuration as the LSTM experiment
    ds_builder = timeseries_dataset.DatasetBuilder(
        root_dir=dataset_root,
        ds_index=ds_idx,
        ds_splits=splits,
        ds_splits_generator=splits_gen,
        configuration=mgr.experiment.dataset_config,
        quick_debug=mgr.quick_debug,
    )

    print("Building long dataframe (points × timesteps × features)...")
    data = ds_builder._build_dataframe()
    data = ds_builder._preprocess_final_dataframe(data, backup_group_id_col=True)

    # Split the dataframe exactly as the LSTM saw it
    splits_to_data = ds_builder._divide_in_splits(data, use_group_id_backup_col=True)

    train_df = splits_to_data[dataset_constants.TRAIN_SPLIT_NAME].copy()
    val_df = splits_to_data[dataset_constants.VALIDATION_SPLIT_NAME].copy()

    print(f"Train rows (timesteps): {len(train_df)}")
    print(f"Val rows   (timesteps): {len(val_df)}")

    # Close caches if still open
    try:
        ds_builder._static_data_cache.close()
    except Exception:
        pass

    try:
        ds_builder._dynamic_data_cache.close()
    except Exception:
        pass

    return train_df, val_df, mgr.experiment.dataset_config


# =============================================================================
# Extract features from long dataframe consistent with the TimeSeriesDataSet
# =============================================================================
def build_features_timeseries(df: pd.DataFrame, ts_cfg: dict):
    """
    Builds X and y from the long dataframe using the SAME feature lists that
    the PyTorch TimeSeriesDataSet uses in the LSTM pipeline.

    Ensures perfect compatibility with the LSTM experiment.
    """
    target_col = timeseries_dataset.COLUMN_TARGET
    feature_cols: list[str] = []

    for key in [
        "static_categoricals",
        "static_reals",
        "time_varying_known_reals",
        "time_varying_unknown_reals",
    ]:
        cols = ts_cfg.get(key, [])
        for c in cols:
            if c in df.columns and c not in feature_cols:
                feature_cols.append(c)

    if not feature_cols:
        raise RuntimeError("No valid feature columns found using the dataset config.")

    X = df[feature_cols].copy()
    y = df[target_col].values

    return X, y, feature_cols


# =============================================================================
# Main: trains a RandomForest baseline aligned with the experiment splits
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Random Forest baseline aligned with a Grano.IT experiment")
    parser.add_argument("--exp_dir", type=str, default=str(EXPERIMENTS_ROOT),
                        help="Root directory of the experiments.")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name to load. If omitted, loads the latest.")
    parser.add_argument("--dataset_path", type=str, default=str(DATASET_ROOT),
                        help="Dataset root directory.")
    parser.add_argument("--quick_debug", action="store_true", default=False,
                        help="Debug mode (mirrors the original pipeline).")

    args = parser.parse_args()

    exp_dir = pathlib.Path(args.exp_dir)
    dataset_root = pathlib.Path(args.dataset_path)

    # -------------------------------------------------------------------------
    # Load train/val splits and build long dataframe exactly as the LSTM saw it
    # -------------------------------------------------------------------------
    train_df, val_df, ts_cfg = build_train_val_from_experiment(
        exp_dir=exp_dir,
        dataset_root=dataset_root,
        exp_name=args.exp_name,
        quick_debug=args.quick_debug,
    )

    # Build features
    #print("Extracting features (long timeseries)...")
    #X_train, y_train, feature_cols = build_features_timeseries(train_df, ts_cfg)
    #X_val, y_val, _ = build_features_timeseries(val_df, ts_cfg)

    print("Building engineered RF features (long timeseries)...")
    X_train, y_train, feature_cols = build_rf_features(train_df, ts_cfg)
    X_val, y_val, _ = build_rf_features(val_df, ts_cfg)

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val   shape: X={X_val.shape}, y={y_val.shape}")


    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Val   shape: X={X_val.shape}, y={y_val.shape}")

    # -------------------------------------------------------------------------
    # High-capacity RandomForest baseline
    # -------------------------------------------------------------------------
    print("\nTraining RandomForestRegressor (aligned with experiment splits)...")
    rf = RandomForestRegressor(
        n_estimators=1400,
        max_depth=18,
        min_samples_leaf=2,
        min_samples_split=10,
        max_features="sqrt",
        bootstrap=True,
        max_samples=0.9,
        n_jobs=-1,
        random_state=403,
    )

    rf.fit(X_train, y_train)

    # Predictions
    y_train_pred = rf.predict(X_train)
    y_val_pred = rf.predict(X_val)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_val_pred))

    print("\n=== RANDOM FOREST (Experiment-Aligned) ===")
    print(f"RMSE_train (per timestep): {rmse_train:.4f}")
    print(f"RMSE_val   (per timestep): {rmse_val:.4f}")
    print("Notes:")
    print(" - Uses identical splits as the LSTM experiment (same CSVs or same generator).")
    print(" - Uses exactly the same feature lists as the original TimeSeriesDataSet.")
    print(" - Metric is RMSE per timestep on the long dataframe.")

    # -------------------------------------------------------------------------
    # Feature importances
    # -------------------------------------------------------------------------
    try:
        importances = rf.feature_importances_
        feature_names = np.array(feature_cols)
        idx = np.argsort(importances)[::-1][:20]

        print("\nTop 20 feature importances:")
        for i in idx:
            print(f"  {feature_names[i]:40s}  {importances[i]:.4f}")
    except Exception as e:
        print(f"[Warning] Failed to compute feature importances: {e}")


if __name__ == "__main__":
    main()
