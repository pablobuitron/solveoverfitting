from collections import defaultdict
from .. import dataset_index
from . import splits_generator
from . import generators_factory

@generators_factory.register_splits_generator
class FieldSplitsGenerator(splits_generator.SplitsGenerator):
    """Generates splits that divide data based on wheat field and year"""

    @classmethod
    def from_dict(cls, input_dict) -> 'FieldSplitsGenerator':
        return cls()


    def _create_splits(self, ids: list[str]) -> tuple[list, list, list]:
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
        train_split = []
        for field in train_fields:
            train_split += field_to_ids[field]

        val_fields = sorted_fields[chunk_len:chunk_len * 2]
        val_split = []
        for field in val_fields:
            val_split += field_to_ids[field]

        test_fields = sorted_fields[chunk_len * 2:]
        test_split = []
        for field in test_fields:
            test_split += field_to_ids[field]

        return train_split, val_split, test_split