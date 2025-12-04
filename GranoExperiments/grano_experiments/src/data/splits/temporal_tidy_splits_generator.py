import logging
from collections import defaultdict
from email.policy import default

from .generators_factory import register_splits_generator
from .splits_generator import SplitsGenerator
from .. import dataset_index

@register_splits_generator
class TemporalTidySplitGenerator(SplitsGenerator):
    # TODO <doc>

    @staticmethod
    def _populate_split(year_to_ids: dict, years_to_use) -> list[str]:
        #
        split = []

        if years_to_use is None:
            split = year_to_ids.pop(max(year_to_ids.keys()))
            split = [str(id_) for id_ in split]
            return split

        for year in years_to_use:
            if year not in year_to_ids:
                logging.warning('Year {} not found in the provided ids, '
                                'no spilt will be created with it'.format(year))
                continue
            split += [str(id_) for id_ in year_to_ids.pop(year)]

        return split

    @classmethod
    def from_dict(cls, input_dict) -> 'SplitsGenerator':
        training_years = input_dict.get('training_years')
        val_years = input_dict.get('validation_years')
        test_years = input_dict.get('test_years')
        return cls(training_years, val_years, test_years)

    def __init__(self, training_years: list[int] = None, val_years: list[int] = None, test_years: list[int] = None):
        self.training_years = training_years
        self.val_years = val_years
        self.test_years = test_years

    def _create_splits(self, ids: list[str]) -> tuple[list, list, list]:
        year_to_ids = defaultdict(list)
        for id_ in ids:
            id_ = dataset_index.Id.from_string(id_)
            year_to_ids[id_.year].append(id_)

        if len(year_to_ids) < 3:
            raise ValueError("Not enough years to create splits")

        test_split = self._populate_split(year_to_ids, self.test_years)
        val_split = self._populate_split(year_to_ids, self.val_years)

        training_split = []
        if self.training_years is not None:
            training_split = self._populate_split(year_to_ids, self.training_years)
        else:
            for remainder_ids in year_to_ids.values():
                training_split += [str(id_) for id_ in remainder_ids]

        everything_ok = bool(training_split) and bool(val_split) and bool(test_split)
        if not everything_ok:
            raise ValueError("Not all splits were created correctly. Some may be empty.")

        return training_split, val_split, test_split
