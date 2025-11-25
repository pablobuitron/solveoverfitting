from pathlib import Path

from GranoExperiments.grano_experiments.src.data import dataset_index, timeseries_dataset, dataset_constants
from GranoExperiments.grano_experiments.src.data.splits import grouped_split_generator

def main():
    root = Path("grano_it_dataset")
    idx = dataset_index.DatasetIndex(root / "index.csv")

    # Usamos el mismo split que en tus experimentos: grouped por campo-año
    splits_gen = grouped_split_generator.GroupedSplitGenerator()

    # quick_debug=True para que no sea gigante pero sí representativo
    builder = timeseries_dataset.DatasetBuilder(
        root_dir=root,
        ds_index=idx,
        ds_splits=None,
        ds_splits_generator=splits_gen,
        configuration=None,   # usa timeseries_dataset_config.yaml del proyecto
        quick_debug=True,
    )

    # Construimos SOLO el dataframe (sin pasar aún a TimeSeriesDataSet)
    df = builder._build_dataframe()

    print("\n=== Columnas del DataFrame final (antes de TimeSeriesDataSet) ===")
    cols = list(df.columns)
    print(cols)

    print("\nTotal columnas:", len(cols))

    # Algunas vistas útiles
    print("\nStatic categoricals:")
    print([c for c in cols if c.startswith("static_")])

    print("\nDynamic (observed_*) features:")
    dyn_cols = [c for c in cols if c.startswith("observed_")]
    print(dyn_cols)
    print("\nTotal dynamic cols:", len(dyn_cols))

if __name__ == "__main__":
    main()

