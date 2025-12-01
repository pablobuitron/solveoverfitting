import pathlib
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .. import dataset_index, dataset_constants
from . import generators_factory

COLUMN_ID = "id"


def from_csv(path: str) -> "DatasetSplit":
    """
    Permite instanciar un DatasetSplit desde un CSV con columna 'id'.

    (Compatibilidad hacia atrás; en otros puntos del código se usa
    también dataset_split.from_csv, no pasa nada por tener ambas cosas.)
    """
    split_name = pathlib.Path(path).stem
    df = pd.read_csv(path)
    ids = df[COLUMN_ID].tolist()
    return DatasetSplit(split_name, ids)


class DatasetSplit:
    """Representa un dataset split con una lista de IDs."""

    def __init__(self, split_name: str, ids: List[str]):
        """
        Args:
            split_name: nombre del split ('train', 'val', 'test', etc.).
            ids: lista de IDs pertenecientes a este split.
        """
        self.split_name = split_name
        self.ids = ids

    def persist(self, output_dir: pathlib.Path):
        """
        Guarda el split en un csv '<split_name>.csv' con una columna 'id'.
        """
        df = pd.DataFrame()
        df[COLUMN_ID] = self.ids
        output_path = pathlib.Path(output_dir) / (self.split_name + ".csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    def __getitem__(self, idx):
        return self.ids[idx]

    def __len__(self):
        return len(self.ids)


# ---------------------------------------------------------------------------
# Clase base para generadores de splits (lo que busca experiment.py)
# ---------------------------------------------------------------------------


class SplitsGenerator:
    """
    Clase base simple. Las subclases deben implementar generate_splits().

    Además, damos una implementación por defecto de generate_splits_files()
    y to_dict(), que es lo que usa Experiment para serializar el generador.
    """

    def generate_splits(self, ids: List[str]) -> Dict[str, DatasetSplit]:
        raise NotImplementedError

    def generate_splits_files(self, dataset_index_path: pathlib.Path, output_dir: pathlib.Path):
        """
        Genera ficheros de splits (train/val/test) leyendo el index.csv
        y llamando a generate_splits().

        Esto se usa solo si llamas a new_experiment(..., pre_generate_splits=True).
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Representación mínima para guardar en el Experiment.
        Las subclases pueden extender este diccionario.
        """
        return {"class_name": self.__class__.__name__}


# ---------------------------------------------------------------------------
# Tu generador “simple” tipo Giovanni: PerFieldPointShuffleSplitGenerator
#   – para cada (year, field) mezcla group_id
#   – 70% train, 15% val, 15% test
# ---------------------------------------------------------------------------


@generators_factory.register_splits_generator
class PerFieldPointShuffleSplitGenerator(SplitsGenerator):
    """
    Split 'simple' al estilo Giovanni:

    - Para cada campo (y anualidad) toma todos los group_id (= puntos en el campo),
      los mezcla aleatoriamente de forma reproducible.
    - Asigna ~70% al train, 15% al validation y 15% al test.

    Es útil para tener train/val muy parecidos y ver claramente si
    el modelo está overfitteando o no.
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
            group_list = sorted(set(group_list))
            rng.shuffle(group_list)

            n = len(group_list)
            if n == 0:
                continue

            n_train = int(round(self.train_ratio * n))
            n_val = int(round(self.validation_ratio * n))
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Guardamos también los parámetros del split por si luego queremos
        reconstruirlo exactamente igual.
        """
        return {
            "class_name": self.__class__.__name__,
            "train_ratio": self.train_ratio,
            "validation_ratio": self.validation_ratio,
            "test_ratio": self.test_ratio,
            "seed": self.seed,
        }
    
@generators_factory.register_splits_generator
class PerFieldYearHoldoutSplitGenerator(SplitsGenerator):
    """
    Split más realista: holdout por anualidad dentro del mismo campo.

    Lógica:
      - Los IDs son tipo 'year|owner|field|pX'.
      - Para cada (owner, field) agrupamos por año.
      - Elegimos un año de holdout (por defecto el más reciente -> 'latest').
      - Todos los puntos de ese año van a validation.
      - Los puntos de los otros años van a train.
      - No devolvemos 'test' (solo train/val) para evitar splits vacíos.
    """

    def __init__(self, holdout_strategy: str = "latest"):
        """
        Args:
            holdout_strategy: 'latest' (año más reciente) o 'earliest' (año más antiguo).
        """
        if holdout_strategy not in ("latest", "earliest"):
            raise ValueError("holdout_strategy must be 'latest' or 'earliest'")
        self.holdout_strategy = holdout_strategy

    def generate_splits(self, ids: List[str]) -> Dict[str, DatasetSplit]:
        # Mapa: (owner, field) -> {year -> [ids...]}
        by_field_year: Dict[tuple, Dict[str, List[str]]] = {}

        for gid in ids:
            gid_str = str(gid)
            parts = gid_str.split(dataset_index.ID_SPLITTER)
            # esperamos: year | owner | field | punto
            if len(parts) < 4:
                raise ValueError(f"Unexpected group_id format: {gid_str}")
            year, owner, field = parts[0], parts[1], parts[2]
            key = (owner, field)

            if key not in by_field_year:
                by_field_year[key] = {}
            by_field_year[key].setdefault(year, []).append(gid_str)

        train_ids: List[str] = []
        val_ids: List[str] = []

        for (owner, field), year_dict in by_field_year.items():
            years = sorted(year_dict.keys())
            if not years:
                continue

            # Elegimos el año de holdout
            if self.holdout_strategy == "latest":
                holdout_year = years[-1]
            else:  # 'earliest'
                holdout_year = years[0]

            if len(years) == 1:
                # Solo una anualidad para este campo: todo a train, nada a val
                train_years = years
                val_years: List[str] = []
            else:
                val_years = [holdout_year]
                train_years = [y for y in years if y != holdout_year]

            for y in train_years:
                train_ids.extend(year_dict[y])
            for y in val_years:
                val_ids.extend(year_dict[y])

        # Seguridad: si por lo que sea no hay validación, explota explícito
        if len(val_ids) == 0:
            raise RuntimeError(
                "PerFieldYearHoldoutSplitGenerator produjo 0 muestras de validación. "
                "Probablemente todos los campos tienen una sola anualidad; "
                "en ese caso este tipo de split no es aplicable."
            )

        splits: Dict[str, DatasetSplit] = {
            dataset_constants.TRAIN_SPLIT_NAME: DatasetSplit(
                dataset_constants.TRAIN_SPLIT_NAME, train_ids
            ),
            dataset_constants.VALIDATION_SPLIT_NAME: DatasetSplit(
                dataset_constants.VALIDATION_SPLIT_NAME, val_ids
            ),
        }
        return splits

    def to_dict(self) -> Dict[str, Any]:
        """
        Para que Experiment pueda serializar el generador y reconstruirlo
        con generators_factory.from_dict.
        """
        return {
            "class_name": self.__class__.__name__,
            "holdout_strategy": self.holdout_strategy,
        }

