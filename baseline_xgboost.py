import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from baseline_rf import (
    build_long_df_and_grouped_split,
    make_aggregated_features,
)

def encode_categoricals_together(X_train, X_val):
    """
    Toma dos DataFrames y convierte las columnas 'object' en códigos categóricos
    usando el MISMO conjunto de categorías para train y val.
    """
    X_train = X_train.copy()
    X_val = X_val.copy()

    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    if cat_cols:
        print(f"Columnas categóricas detectadas: {cat_cols}")
    else:
        print("No se detectaron columnas categóricas (object).")

    for c in cat_cols:
        # Unir train+val para tener el universo de categorías
        all_vals = pd.concat([X_train[c], X_val[c]], axis=0)
        cats = all_vals.astype("category").cat.categories

        X_train[c] = pd.Categorical(X_train[c], categories=cats).codes
        X_val[c] = pd.Categorical(X_val[c], categories=cats).codes

    return X_train, X_val


def main():
    print("Construyendo DataFrame largo y split grouped por campo-año (XGB)...")
    train_df, val_df, ts_cfg = build_long_df_and_grouped_split()

    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    print("Agregando features por group_id (punto dentro del campo)...")
    X_train, y_train, agg_train = make_aggregated_features(train_df, ts_cfg)
    X_val, y_val, agg_val = make_aggregated_features(val_df, ts_cfg)

    # Asegurarnos de que son DataFrames (por si algún día cambias make_aggregated_features)
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=agg_train.drop(columns=["target"]).columns)
    if not isinstance(X_val, pd.DataFrame):
        X_val = pd.DataFrame(X_val, columns=agg_val.drop(columns=["target"]).columns)

    print(f"Train groups: {X_train.shape[0]}, features: {X_train.shape[1]}")
    print(f"Val groups:   {X_val.shape[0]}, features: {X_val.shape[1]}")

    # 🔴 Aquí arreglamos las categóricas (static_*) que están como object
    X_train, X_val = encode_categoricals_together(X_train, X_val)

    # Pasar a numpy float32 para XGBoost
    X_train_np = X_train.to_numpy(dtype=np.float32)
    X_val_np = X_val.to_numpy(dtype=np.float32)

    print("Entrenando XGBRegressor baseline (RMSE sobre campos/años no vistos)...")
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

    print("\n=== BASELINE XGBOOST (grouped por campo-año) ===")
    print(f"RMSE_val: {rmse_val:.4f}")
    print("(Para comparar: RF grouped ~ 1.1651, TFT grouped ~ 4.02 en tu experimento 1851)")


if __name__ == "__main__":
    main()
