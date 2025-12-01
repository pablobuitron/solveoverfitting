import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from baseline_rf import (
    build_long_df_and_grouped_split,
    make_aggregated_features,
)


def encode_categoricals_together(X_train: pd.DataFrame, X_val: pd.DataFrame):
    """
    Take train and validation DataFrames and encode all 'object' columns
    as categorical integer codes, using the SAME category mapping
    for both train and validation.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()

    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    if cat_cols:
        print(f"Detected categorical (object) columns: {cat_cols}")
    else:
        print("No categorical (object) columns detected.")

    for c in cat_cols:
        # Join train + val so we see the full set of categories
        all_vals = pd.concat([X_train[c], X_val[c]], axis=0)
        cats = all_vals.astype("category").cat.categories

        X_train[c] = pd.Categorical(X_train[c], categories=cats).codes
        X_val[c] = pd.Categorical(X_val[c], categories=cats).codes

    return X_train, X_val


def main():
    print("Building long DataFrame and grouped split by field-year (XGB)...")
    train_df, val_df, ts_cfg = build_long_df_and_grouped_split()

    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    print("Aggregating features per group_id (point within a field)...")
    X_train, y_train, agg_train = make_aggregated_features(train_df, ts_cfg)
    X_val, y_val, agg_val = make_aggregated_features(val_df, ts_cfg)

    # Make sure we have DataFrames (in case make_aggregated_features changes in the future)
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(
            X_train,
            columns=agg_train.drop(columns=["target"]).columns,
        )
    if not isinstance(X_val, pd.DataFrame):
        X_val = pd.DataFrame(
            X_val,
            columns=agg_val.drop(columns=["target"]).columns,
        )

    print(f"Train groups: {X_train.shape[0]}, features: {X_train.shape[1]}")
    print(f"Val groups:   {X_val.shape[0]}, features: {X_val.shape[1]}")

    # Handle categorical (static_*) columns that are still of dtype object
    X_train, X_val = encode_categoricals_together(X_train, X_val)

    # Convert to numpy float32 for XGBoost
    X_train_np = X_train.to_numpy(dtype=np.float32)
    X_val_np = X_val.to_numpy(dtype=np.float32)

    print("Training XGBRegressor baseline (RMSE on unseen field-years)...")
    xgb = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        tree_method="hist",
    )

    xgb.fit(X_train_np, y_train)

    preds_val = xgb.predict(X_val_np)
    rmse_val = np.sqrt(mean_squared_error(y_val, preds_val))

    print("\n=== XGBOOST BASELINE (grouped by field-year) ===")
    print(f"RMSE_val: {rmse_val:.4f}")
    print("(For reference: RF grouped ~ 1.1651, TFT grouped ~ 4.02 in experiment 1851)")


if __name__ == "__main__":
    main()
