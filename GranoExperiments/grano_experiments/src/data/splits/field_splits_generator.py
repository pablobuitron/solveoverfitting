import pathlib
from collections import defaultdict

from .dataset_split import DatasetSplit
from .. import dataset_index
from . import splits_generator
from .. import dataset_constants
from . import generators_factory

@generators_factory.register_splits_generator
class FieldSplitsGenerator(splits_generator.SplitsGenerator):
    """Generates splits that divide data based on wheat field and year"""

    @classmethod
    def from_dict(cls, input_dict) -> 'FieldSplitsGenerator':
        return cls()

    def generate_splits_files(self, dataset_index_path: str | pathlib.Path, output_dir: str | pathlib.Path):
        output_dir = pathlib.Path(output_dir)
        self._output_dir_exists(output_dir)

        ds_index = dataset_index.DatasetIndex(dataset_index_path)
        splits = self.generate_splits(ds_index.get_all_ids())

        splits_dir = self._create_splits_dir(output_dir)
        [split.persist(splits_dir) for split in splits.values()]

    def generate_splits(self, ids: list[str]) -> dict[str, DatasetSplit]:
        field_to_count = defaultdict(int)
        field_to_ids = defaultdict(list)
        for id_ in ids:
            year_and_field = id_.rsplit(dataset_index.ID_SPLITTER, 1)[0]
            field_to_count[year_and_field] += 1
            field_to_ids[year_and_field].append(id_)

        sorted_fields = sorted(field_to_count.keys(), key=lambda x: field_to_count[x])
        sorted_fields.reverse()
        chunk_len = len(sorted_fields) // 3

        train_fields = sorted_fields[:chunk_len]
        train_ids = []
        for field in train_fields:
            train_ids += field_to_ids[field]
        train_split = DatasetSplit(dataset_constants.TRAIN_SPLIT_NAME, train_ids)

        val_fields = sorted_fields[chunk_len:chunk_len * 2]
        val_ids = []
        for field in val_fields:
            val_ids += field_to_ids[field]
        val_split = DatasetSplit(dataset_constants.VALIDATION_SPLIT_NAME, val_ids)

        test_fields = sorted_fields[chunk_len * 2:]
        test_ids = []
        for field in test_fields:
            test_ids += field_to_ids[field]
        test_split = DatasetSplit(dataset_constants.TEST_SPLIT_NAME, test_ids)

        result = {
            train_split.split_name: train_split,
            val_split.split_name: val_split,
            test_split.split_name: test_split
        }

        return result