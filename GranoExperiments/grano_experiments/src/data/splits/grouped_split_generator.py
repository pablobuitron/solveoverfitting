import pathlib
import random
from collections import defaultdict
from typing import Iterable, List

from .dataset_split import DatasetSplit
from .. import dataset_constants, dataset_index
from . import splits_generator, generators_factory


@generators_factory.register_splits_generator
class GroupedSplitGenerator(splits_generator.SplitsGenerator):
    """Split generator that keeps entire identifier groups together.

    Groups are derived from the sample IDs by splitting them with
    ``dataset_index.ID_SPLITTER`` and selecting the components indicated by
    ``groupby_id_indices`` (default: year + field). All samples that share the
    resulting key are assigned to the same split, preventing leakage across
    train/validation/test.
    """

    _DEFAULT_TRAIN_SPLIT_RATIO = 0.7
    _DEFAULT_VAL_SPLIT_RATIO = 0.15
    _DEFAULT_GROUPBY_INDICES = (0, 1)

    @classmethod
    def from_dict(cls, input_dict) -> "GroupedSplitGenerator":
        seed = input_dict.get("seed")
        if seed is None:
            raise ValueError("Seed is required when instantiating from dict.")
        train_ratio = input_dict.get("train_split_ratio") or cls._DEFAULT_TRAIN_SPLIT_RATIO
        val_ratio = input_dict.get("val_split_ratio") or cls._DEFAULT_VAL_SPLIT_RATIO
        indices = input_dict.get("groupby_id_indices") or list(cls._DEFAULT_GROUPBY_INDICES)
        return cls(
            seed=seed,
            train_split_ratio=train_ratio,
            val_split_ratio=val_ratio,
            groupby_id_indices=indices,
        )

    def __init__(
        self,
        seed: int = 42,
        train_split_ratio: float = _DEFAULT_TRAIN_SPLIT_RATIO,
        val_split_ratio: float = _DEFAULT_VAL_SPLIT_RATIO,
        groupby_id_indices: Iterable[int] = _DEFAULT_GROUPBY_INDICES,
    ):
        if train_split_ratio + val_split_ratio >= 1:
            raise ValueError("train_split_ratio and val_split_ratio cannot add up to 1 or more")

        self.seed = seed
        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio
        self.groupby_id_indices = list(groupby_id_indices)
        random.seed(self.seed)

    def generate_splits_files(self, dataset_index_path: str | pathlib.Path, output_dir: str | pathlib.Path):
        output_dir = pathlib.Path(output_dir)
        self._output_dir_exists(output_dir)

        ds_index = dataset_index.DatasetIndex(dataset_index_path)
        splits = self.generate_splits(ds_index.get_all_ids())

        splits_dir = self._create_splits_dir(output_dir)
        [split.persist(splits_dir) for split in splits.values()]

    def generate_splits(self, ids: List[str]) -> dict[str, DatasetSplit]:
        grouped_ids = self._group_ids(ids)
        split_groups = self._split_group_keys(list(grouped_ids.keys()))

        train_ids = self._collect_ids(grouped_ids, split_groups[dataset_constants.TRAIN_SPLIT_NAME])
        val_ids = self._collect_ids(grouped_ids, split_groups[dataset_constants.VALIDATION_SPLIT_NAME])
        test_ids = self._collect_ids(grouped_ids, split_groups[dataset_constants.TEST_SPLIT_NAME])

        return {
            dataset_constants.TRAIN_SPLIT_NAME: DatasetSplit(dataset_constants.TRAIN_SPLIT_NAME, train_ids),
            dataset_constants.VALIDATION_SPLIT_NAME: DatasetSplit(dataset_constants.VALIDATION_SPLIT_NAME, val_ids),
            dataset_constants.TEST_SPLIT_NAME: DatasetSplit(dataset_constants.TEST_SPLIT_NAME, test_ids),
        }

    def _group_ids(self, ids: List[str]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for id_ in ids:
            parts = id_.split(dataset_index.ID_SPLITTER)
            key_parts = [parts[i] for i in self.groupby_id_indices if i < len(parts)]
            key = dataset_index.ID_SPLITTER.join(key_parts)
            grouped[key].append(id_)
        return grouped

    def _split_group_keys(self, group_keys: List[str]) -> dict[str, list[str]]:
        random.shuffle(group_keys)
        train_limit = round(len(group_keys) * self.train_split_ratio)
        val_limit = train_limit + round(len(group_keys) * self.val_split_ratio)

        return {
            dataset_constants.TRAIN_SPLIT_NAME: group_keys[:train_limit],
            dataset_constants.VALIDATION_SPLIT_NAME: group_keys[train_limit:val_limit],
            dataset_constants.TEST_SPLIT_NAME: group_keys[val_limit:],
        }

    @staticmethod
    def _collect_ids(groups: dict[str, list[str]], keys: List[str]) -> list[str]:
        ids: list[str] = []
        for key in keys:
            ids.extend(groups.get(key, []))
        return ids