import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from baseline_rf import (
    build_long_df_and_grouped_split,
    make_aggregated_features,
)


class MLPRegressor(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(X_train, y_train, X_val, y_val, max_epochs=300, batch_size=64, patience=30):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")

    # Escalador
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Tensores
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MLPRegressor(in_features=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # --- Val ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {
                "model": model.state_dict(),
                "scaler_mean": scaler.mean_.copy(),
                "scaler_scale": scaler.scale_.copy(),
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping en epoch {epoch+1}")
                break

    # Cargar mejor modelo
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        scaler.mean_ = best_state["scaler_mean"]
        scaler.scale_ = best_state["scaler_scale"]

    # Predicciones en val para RMSE
    model.eval()
    with torch.no_grad():
        X_val_scaled = scaler.transform(X_val)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        preds_val = model(X_val_t).cpu().numpy()

    rmse_val = np.sqrt(mean_squared_error(y_val, preds_val))
    return rmse_val


def main():
    print("Construyendo DataFrame largo y split grouped por campo-año (MLP)...")
    train_df, val_df, ts_cfg = build_long_df_and_grouped_split()

    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}")

    print("Agregando features por group_id (punto dentro del campo)...")
    X_train_df, y_train, agg_train = make_aggregated_features(train_df, ts_cfg)
    X_val_df, y_val, agg_val = make_aggregated_features(val_df, ts_cfg)

    # Asegurarnos de que no queden NaNs
    X_train_df = X_train_df.copy()
    X_val_df = X_val_df.copy()
    X_train_df = X_train_df.fillna(X_train_df.mean(numeric_only=True))
    X_val_df = X_val_df.fillna(X_train_df.mean(numeric_only=True))

    print(f"Train groups: {X_train_df.shape[0]}, features: {X_train_df.shape[1]}")
    print(f"Val groups:   {X_val_df.shape[0]}, features: {X_val_df.shape[1]}")

    rmse_val = train_mlp(
        X_train_df.values,
        y_train,
        X_val_df.values,
        y_val,
        max_epochs=300,
        batch_size=64,
        patience=40,
    )

    print("\n=== BASELINE MLP (grouped por campo-año) ===")
    print(f"RMSE_val: {rmse_val:.4f}")
    print("(Para comparar: RF ~ 1.17, XGB ~ 1.47, TFT grouped ~ 4.02)")


if __name__ == "__main__":
    main()
