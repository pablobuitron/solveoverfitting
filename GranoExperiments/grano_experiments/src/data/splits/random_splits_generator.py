from typing import Tuple
import random
from . import splits_generator
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


