import pathlib
from typing import Tuple
import random

import pandas as pd

from .dataset_split import DatasetSplit
from .. import dataset_constants
from .. import dataset_index
from . import splits_generator
from . import dataset_split
from . import generators_factory

@generators_factory.register_splits_generator
class RandomSplitGenerator(splits_generator.SplitsGenerator):

    _MAX_SEED_ALLOWED = 1000
    _DEFAULT_TRAIN_SPLIT_RATIO = 0.7
    _DEFAULT_VAL_SPLIT_RATIO = 0.15

    @classmethod
    def from_dict(cls, input_dict) -> 'RandomSplitGenerator':
        seed = input_dict.get('seed')
        if seed is None:
            raise ValueError('Seed is required when instantiating from dict.')
        train_split_ratio = input_dict.get('train_split_ratio') or cls._DEFAULT_TRAIN_SPLIT_RATIO
        val_split_ratio = input_dict.get('val_split_ratio') or cls._DEFAULT_VAL_SPLIT_RATIO
        return cls(
            seed=seed,
            train_split_ratio=train_split_ratio,
            val_split_ratio=val_split_ratio,
        )

    def __init__(self,
                 seed: int = None,
                 train_split_ratio: float = 0.7,
                 val_split_ratio: float = 0.15):
        """
        This class implements the SplitsGenerator interface to create train, validation, and test splits by
         randomly shuffling the dataset indices.
        Args:
            seed: The seed for the random number generator to ensure reproducibility.
            If not provided, a random seed will be generated.
            train_split_ratio: Percentage of the dataset to use for training. Must be expressed as a float between 0 and 1.
            val_split_ratio: Percentage of the dataset to use for training. Must be expressed as a float between 0 and 1.
        """
        if train_split_ratio + val_split_ratio >= 1:
            raise ValueError('train_split_ratio and val_split_ratio cannot add up to 1 or more')

        self.seed = seed or random.randint(0, self._MAX_SEED_ALLOWED)
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        random.seed(self.seed)

    def generate_splits_files(self, dataset_index_path: str | pathlib.Path, output_dir: str | pathlib.Path):
        output_dir = pathlib.Path(output_dir)
        self._output_dir_exists(output_dir)

        ds_index = dataset_index.DatasetIndex(dataset_index_path)
        splits = self.generate_splits(ds_index.get_all_ids())

        splits_dir = self._create_splits_dir(output_dir)
        [split.persist(splits_dir) for split in splits.values()]


    def generate_splits(self, ids: list[str]) -> dict[str, DatasetSplit]:

        train_split, val_split, test_split = self._create_splits(ids)
        train_split = dataset_split.DatasetSplit(dataset_constants.TRAIN_SPLIT_NAME, train_split)
        val_split = dataset_split.DatasetSplit(dataset_constants.VALIDATION_SPLIT_NAME, val_split)
        test_split = dataset_split.DatasetSplit(dataset_constants.TEST_SPLIT_NAME, test_split)

        result = {
            train_split.split_name: train_split,
            val_split.split_name: val_split,
            test_split.split_name: test_split,
        }

        return result

    def _create_splits(self, ids: list[str]) -> Tuple[list, list, list]:
        ids = list(set(ids))
        random.shuffle(ids)

        if len(ids) == 3:
            return [ids[0]], [ids[1]], [ids[2]]

        train_split_index = round(len(ids) * self.train_split_ratio)
        val_split_index = train_split_index + round(len(ids) * self.val_split_ratio)

        train_split = ids[:train_split_index]
        val_split = ids[train_split_index:val_split_index]
        test_split = ids[val_split_index:]

        return train_split, val_split, test_split


