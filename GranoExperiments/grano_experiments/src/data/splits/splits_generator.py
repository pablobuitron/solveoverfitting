import pathlib
from typing import List, Dict

import numpy as np
import pandas as pd

from .. import dataset_index, dataset_constants
from .generators_factory import register_splits_generator

COLUMN_ID = "id"


class SplitsGenerator:
    """
    Clase base vacía para generadores de splits.

    Se usa sólo para tipado y para mantener compatibilidad con el código
    que hace `from ..data.splits.splits_generator import SplitsGenerator`.
    """
    pass


def from_csv(path: str) -> "DatasetSplit":
    """
    Allows to instantiate a DatasetSplit from a CSV file.

    NOTE: kept for backward compatibility. For pre-generated splits
    you are already using `dataset_split.from_csv`, so this helper is
    essentially redundant but harmless.
    """
    split_name = pathlib.Path(path).stem
    df = pd.read_csv(path)
    ids = df[COLUMN_ID].tolist()
    return DatasetSplit(split_name, ids)


class DatasetSplit:
    """Represents a dataset split containing a list of IDs.

    This class encapsulates the information for a single data split (e.g.,
    train, validation, test), including its name and the list of IDs
    belonging to that split. It provides a method to persist the split
    to a CSV file.
    """

    def __init__(self, split_name: str, ids: List[str]):
        """Initializes the DatasetSplit.

        Args:
            split_name: The name of the split (e.g., 'train', 'val', 'test').
            ids: A list of IDs belonging to this split.
        """
        self.split_name = split_name
        self.ids = ids

    def persist(self, output_dir: pathlib.Path):
        """Saves the split to a CSV file.

        The CSV file will contain a single column named 'id' with the list
        of IDs for the split. The file will be named '<split_name>.csv'.

        Args:
            output_dir: The directory where the CSV file will be saved.
        """
        df = pd.DataFrame()
        df[COLUMN_ID] = self.ids
        output_path = output_dir / (self.split_name + ".csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def __getitem__(self, idx):
        return self.ids[idx]

    def __len__(self):
        return len(self.ids)


# ---------------------------------------------------------------------------
# Nuevo generador de splits: "per_field_point" (shuffle de puntos dentro
# de cada campo-año, como en el experimento “bueno” de Giovanni)
# ---------------------------------------------------------------------------


@register_splits_generator
class PerFieldPointShuffleSplitGenerator(SplitsGenerator):
    """
    Splitter 'simple' al estilo Giovanni:

    - Para cada campo (y anualidad) toma todos los group_id (= puntos en el campo),
      los mezcla aleatoriamente de forma reproducible.
    - Asigna ~70% al train, 15% al validation y 15% al test.

    Este esquema hace que train/val sean muy parecidos (útil para debugging
    y tuning), aunque es menos realista como escenario de producción que
    un split estrictamente por campo-año.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 403,
    ):
        if not np.isclose(train_ratio + validation_ratio + test_ratio, 1.0):
            raise ValueError("train_ratio + validation_ratio + test_ratio must be 1.0")

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    # ------------------------------------------------------------------
    # Interfaz que usa DatasetBuilder: generate_splits(ids)
    #   ids es la lista de group_id (strings tipo 'year<SPLITTER>field<SPLITTER>point')
    #   devolvemos un dict: split_name -> DatasetSplit
    # ------------------------------------------------------------------
    def generate_splits(self, ids: List[str]) -> Dict[str, DatasetSplit]:
        rng = np.random.RandomState(self.seed)

        # agrupamos por (year, field) usando el separador que define dataset_index
        by_field: Dict[tuple, List[str]] = {}
        for gid in ids:
            gid_str = str(gid)
            parts = gid_str.split(dataset_index.ID_SPLITTER)
            if len(parts) < 3:
                raise ValueError(f"Unexpected group_id format: {gid_str}")
            year, field = parts[0], parts[1]
            key = (year, field)
            by_field.setdefault(key, []).append(gid_str)

        train_ids: List[str] = []
        val_ids: List[str] = []
        test_ids: List[str] = []

        for (year, field), group_list in by_field.items():
            # orden estable + sin duplicados
            group_list = sorted(set(group_list))
            rng.shuffle(group_list)

            n = len(group_list)
            if n == 0:
                continue

            n_train = int(round(self.train_ratio * n))
            n_val = int(round(self.validation_ratio * n))
            # el resto a test
            n_test = max(0, n - n_train - n_val)

            train_ids.extend(group_list[:n_train])
            val_ids.extend(group_list[n_train : n_train + n_val])
            test_ids.extend(group_list[n_train + n_val : n_train + n_val + n_test])

        splits: Dict[str, DatasetSplit] = {
            dataset_constants.TRAIN_SPLIT_NAME: DatasetSplit(
                dataset_constants.TRAIN_SPLIT_NAME, train_ids
            ),
            dataset_constants.VALIDATION_SPLIT_NAME: DatasetSplit(
                dataset_constants.VALIDATION_SPLIT_NAME, val_ids
            ),
            dataset_constants.TEST_SPLIT_NAME: DatasetSplit(
                dataset_constants.TEST_SPLIT_NAME, test_ids
            ),
        }
        return splits

    # ------------------------------------------------------------------
    # Interfaz que usa ExperimentManager si se piden splits pre-generados:
    #   generate_splits_files(index.csv, experiment_dir)
    # ------------------------------------------------------------------
    def generate_splits_files(self, dataset_index_path: pathlib.Path, output_dir: pathlib.Path):
        """
        Genera ficheros CSV de splits (train/val/test) a partir del index del dataset.

        Este método se usa solo si llamas a new_experiment(..., pre_generate_splits=True).
        En tus ejecuciones actuales no lo estás usando, pero lo dejamos implementado
        para coherencia con el resto de generadores.
        """
        df = pd.read_csv(dataset_index_path)
        if dataset_index.COLUMN_ID in df.columns:
            all_ids = df[dataset_index.COLUMN_ID].tolist()
        elif COLUMN_ID in df.columns:
            all_ids = df[COLUMN_ID].tolist()
        else:
            raise RuntimeError(
                f"No se encuentra columna de IDs ni '{dataset_index.COLUMN_ID}' "
                f"ni '{COLUMN_ID}' en {dataset_index_path}"
            )

        splits = self.generate_splits(all_ids)
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for split in splits.values():
            split.persist(output_dir)
