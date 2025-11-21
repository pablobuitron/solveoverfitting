import abc
import pathlib

import pandas as pd

from .. import dataset_constants
from ..splits import dataset_split


class SplitsGenerator(metaclass=abc.ABCMeta):
    """Abstract base class for data split generators.

    This class defines the interface for splitting a dataset index into
    training, validation, and test sets. The resulting splits are saved to
    files, each containing a list of indices from the original dataset index
    file (e.g., 'index.csv').
    """

    @staticmethod
    def _output_dir_exists(output_dir: pathlib.Path):
        output_dir = pathlib.Path(output_dir)
        if not output_dir.exists():
            raise FileNotFoundError("The provided output dir '%r' does not exist", output_dir)
        if not output_dir.is_dir():
            raise NotADirectoryError("The provided output dir '%r' is not a directory", output_dir)
        return output_dir

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, input_dict) -> 'SplitsGenerator':
        ...

    @abc.abstractmethod
    def generate_splits_files(self, dataset_index_path: str | pathlib.Path, output_dir: str | pathlib.Path):
        """Generates and saves the dataset splits.

        This method takes a dataset index file and creates a subdirectory inside the output dir where the train,
        validation, and test split files will be saved.

        Args:
            dataset_index_path: Path to the dataset index file (e.g., 'index.csv').
            output_dir: Path to the directory where the split files will be
              saved.
        """
        ...


    @abc.abstractmethod
    def generate_splits(self, ids: list[str]) -> dict[str, dataset_split.DatasetSplit]:
        """
        Generates dataset splits from a list of sample IDs.

        The implementation partitions `ids` into disjoint splits (e.g., 'train',
        'val', 'test') according to the specific splitting strategy.

        Args:
            ids: List of unique sample identifiers.

        Returns:
            A mapping from split name to `DatasetSplit`.
        """
        ...

    # noinspection PyMethodMayBeStatic
    def _create_splits_dir(self, output_dir: pathlib.Path) -> pathlib.Path:
        splits_dirname = f'{dataset_constants.SPLITS_DINRAME}'
        splits_dir = output_dir / splits_dirname
        splits_dir.mkdir(parents=False, exist_ok=False)
        return splits_dir