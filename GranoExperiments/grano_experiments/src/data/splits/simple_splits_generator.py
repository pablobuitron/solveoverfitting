from collections import defaultdict
from random import shuffle

from .. import dataset_index
from .splits_generator import SplitsGenerator
from . import generators_factory

@generators_factory.register_splits_generator
class SimpleSplitsGenerator(SplitsGenerator):
    DEFAULT_TRAIN_SPLIT_RATIO = 0.7
    DEFAULT_VAL_SPLIT_RATIO = 0.15

    @classmethod
    def from_dict(cls, input_dict) -> 'SplitsGenerator':
        train_split_ratio = input_dict.get('train_split_ratio') or cls.DEFAULT_TRAIN_SPLIT_RATIO
        val_split_ratio = input_dict.get('val_split_ratio') or cls.DEFAULT_VAL_SPLIT_RATIO
        return cls(train_split_ratio=train_split_ratio, val_split_ratio=val_split_ratio)

    def __init__(self, train_split_ratio = DEFAULT_TRAIN_SPLIT_RATIO, val_split_ratio = DEFAULT_VAL_SPLIT_RATIO):
        """Split generator that creates train/validation/test sets while keeping
                samples from the same field and year together.

                The generator takes a list of sample IDs and:
                  - Groups samples belonging to the same field and year.
                  - Applies the train/validation/test ratios within each (shuffled) group.
                  - Merges all groups to produce the final splits.
            """
        if train_split_ratio + val_split_ratio >= 1:
            raise ValueError('train_split_ratio and val_split_ratio cannot add up to 1 or more')

        self.train_split_ratio = train_split_ratio
        self.val_split_ratio = val_split_ratio


    def _create_splits(self, ids: list[str]) -> tuple[list, list, list]:
        aggregated_samples = defaultdict(list)  # key: field and year
        for id_ in ids:
            year_and_field = id_.rsplit(dataset_index.ID_SPLITTER, 1)[0]
            aggregated_samples[year_and_field].append(id_)

        train_split = []
        val_split = []
        test_split = []
        for samples in aggregated_samples.values():
            shuffle(samples)
            samples_split = self._split_train_val_test(samples, self.train_split_ratio, self.val_split_ratio)
            train_split += samples_split[0]
            val_split += samples_split[1]
            test_split += samples_split[2]

        return train_split, val_split, test_split