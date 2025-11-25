import csv
from pathlib import Path

def summarize_metrics(exp_path: str):
    metrics_file = Path(exp_path) / "training_results" / "version_0" / "metrics.csv"
    print(f"\n=== {exp_path} ===")
    if not metrics_file.exists():
        print("  metrics.csv NO encontrado:", metrics_file)
        return

    by_epoch = {}
    with metrics_file.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            epoch = int(float(row["epoch"]))
            by_epoch.setdefault(epoch, []).append(row)

    best = None  # almacenará (epoch, val_RMSE, train_loss_epoch)
    for epoch, rows in by_epoch.items():
        val_rows = [r for r in rows if r.get("val_RMSE")]
        if not val_rows:
            continue
        val_row = sorted(val_rows, key=lambda r: int(r["step"]))[-1]
        val_rmse = float(val_row["val_RMSE"])
        train_loss = val_row.get("train_loss_epoch") or val_row.get("train_loss_step")
        train_loss = float(train_loss) if train_loss not in (None, "") else None
        if best is None or val_rmse < best[1]:
            best = (epoch, val_rmse, train_loss)

    if best is None:
        print("  No se encontraron val_RMSE.")
        return

    print(f"  Mejor epoch: {best[0]}")
    print(f"  Mejor val_RMSE: {best[1]:.4f}")
    print(f"  train_loss_epoch en esa epoch: {best[2]}")

    print("  Últimos 5 epochs (epoch, val_RMSE):")
    for ep in sorted(by_epoch.keys())[-5:]:
        rows = by_epoch[ep]
        val_rows = [r for r in rows if r.get("val_RMSE")]
        if not val_rows:
            continue
        val_row = sorted(val_rows, key=lambda r: int(r["step"]))[-1]
        print(f"    {ep:3d}  ->  {float(val_row['val_RMSE']):.4f}")

# CAMBIA ESTOS:
old_exp = "experiment|20251121-1050"
new_exp = "experiment|20251125-0129"   # ← pon aquí tu experimento grouped
summarize_metrics(old_exp)
summarize_metrics(new_exp)
