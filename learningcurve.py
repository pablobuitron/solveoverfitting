import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Grano.IT imports
from GranoExperiments.grano_experiments.src.experiments_manager import experiments_manager
from GranoExperiments.grano_experiments.src.data import (
    dataset_index,
    timeseries_dataset,
    dataset_constants,
)
from GranoExperiments.grano_experiments.src.data.splits import generators_factory


ROOT = pathlib.Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "grano_it_dataset"

# NOTE: using dev_experiments as default root for experiments
DEFAULT_EXPERIMENTS_ROOT = ROOT / "GranoExperiments" / "grano_experiments" / "dev_experiments"


def build_train_val_from_experiment(exp_dir: pathlib.Path,
                                    dataset_root: pathlib.Path,
                                    exp_name: str | None,
                                    quick_debug: bool = False):
    """
    Load a Grano.IT experiment (latest or specific) and reconstruct:

      - the long dataframe (points × timesteps × features)
      - train/val dataframes using the same splits as the LSTM pipeline.
    """
    mgr = experiments_manager.ExperimentManager(
        experiments_root_dir=str(exp_dir),
        dataset_root_dir=str(dataset_root),
        quick_debug=quick_debug,
    )

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
    mgr._load_configuration_for_training(training_configuration_override=None)
    mgr._check_configuration_for_training()

    ds_idx = dataset_index.DatasetIndex(mgr.dataset_index_path)

    if mgr.experiment.splits_files:
        splits = mgr._fetch_splits(
            [dataset_constants.TRAIN_SPLIT_NAME, dataset_constants.VALIDATION_SPLIT_NAME]
        )
        splits_gen = None
        print("Using pre-generated CSV splits from experiment.")
    else:
        splits = None
        splits_gen = generators_factory.from_dict(mgr.experiment.splits_generator_data)
        print("Using split generator from experiment configuration.")

    ds_builder = timeseries_dataset.DatasetBuilder(
        root_dir=dataset_root,
        ds_index=ds_idx,
        ds_splits=splits,
        ds_splits_generator=splits_gen,
        configuration=mgr.experiment.dataset_config,
        quick_debug=mgr.quick_debug,
    )

    print("Building long dataframe...")
    data = ds_builder._build_dataframe()
    data = ds_builder._preprocess_final_dataframe(data, backup_group_id_col=True)

    splits_to_data = ds_builder._divide_in_splits(data, use_group_id_backup_col=True)

    train_df = splits_to_data[dataset_constants.TRAIN_SPLIT_NAME].copy()
    val_df = splits_to_data[dataset_constants.VALIDATION_SPLIT_NAME].copy()

    print(f"Train rows (timesteps): {len(train_df)}")
    print(f"Val rows   (timesteps): {len(val_df)}")

    return train_df, val_df, mgr.experiment.dataset_config


def build_features_timeseries(df: pd.DataFrame, ts_cfg: dict):
    """
    Build X, y using the same feature lists as the TimeSeriesDataSet config.
    """
    feature_cols: list[str] = []
    for key in ["static_categoricals", "static_reals",
                "time_varying_known_reals", "time_varying_unknown_reals"]:
        cols = ts_cfg.get(key, [])
        for c in cols:
            if c in df.columns and c not in feature_cols:
                feature_cols.append(c)

    if not feature_cols:
        raise RuntimeError("No feature columns found from dataset config.")

    X = df[feature_cols].copy()
    y = df[timeseries_dataset.COLUMN_TARGET].values
    return X, y, feature_cols


def plot_learning_curves(model: RandomForestRegressor,
                         X_train: pd.DataFrame,
                         y_train: np.ndarray,
                         X_val: pd.DataFrame,
                         y_val: np.ndarray):
    """
    Plot a learning curve by gradually increasing the fraction of training data
    and computing train/validation RMSE.
    """
    train_sizes = np.linspace(0.1, 1.0, 8)

    train_errors = []
    val_errors = []

    print("\nGenerating learning curve...\n")

    for frac in train_sizes:
        n = int(len(X_train) * frac)

        X_sub = X_train[:n]
        y_sub = y_train[:n]

        m = RandomForestRegressor(**model.get_params())
        m.fit(X_sub, y_sub)

        train_pred = m.predict(X_sub)
        val_pred = m.predict(X_val)

        train_rmse = np.sqrt(mean_squared_error(y_sub, train_pred))
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))

        train_errors.append(train_rmse)
        val_errors.append(val_rmse)

        print(f"Fraction {frac:.2f} → Train RMSE={train_rmse:.4f}, Val RMSE={val_rmse:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_errors, "o-", label="Train RMSE")
    plt.plot(train_sizes, val_errors, "o-", label="Validation RMSE")
    plt.xlabel("Fraction of training data")
    plt.ylabel("RMSE")
    plt.title("Random Forest Learning Curve (experiment-aligned)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("rf_learning_curve.png")
    print("\nSaved figure: rf_learning_curve.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="RF learning curve aligned with a Grano.IT experiment")
    parser.add_argument("--exp_dir", type=str, default=str(DEFAULT_EXPERIMENTS_ROOT),
                        help="Root directory of experiments (default: dev_experiments).")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name to load. If omitted, loads the latest.")
    parser.add_argument("--dataset_path", type=str, default=str(DATASET_ROOT),
                        help="Path to dataset root (where index.csv lives).")

    args = parser.parse_args()

    exp_dir = pathlib.Path(args.exp_dir)
    dataset_root = pathlib.Path(args.dataset_path)

    train_df, val_df, ts_cfg = build_train_val_from_experiment(
        exp_dir=exp_dir,
        dataset_root=dataset_root,
        exp_name=args.exp_name,
        quick_debug=False,
    )

    print("Building features...")
    X_train, y_train, feat_cols = build_features_timeseries(train_df, ts_cfg)
    X_val, y_val, _ = build_features_timeseries(val_df, ts_cfg)

    # Same RF configuration as your baseline
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

    plot_learning_curves(rf, X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
