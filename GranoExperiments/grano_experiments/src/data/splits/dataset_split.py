import pathlib
from typing import List

import pandas as pd

COLUMN_ID = 'id'


def from_csv(path: str) -> 'DatasetSplit':
    """Allows to instantiate a DatasetSplit from a CSV file."""
    spilt_name = pathlib.Path(path).stem
    df = pd.read_csv(path)
    ids = df[COLUMN_ID].tolist()
    return DatasetSplit(spilt_name, ids)

class DatasetSplit:
    """Represents a dataset split containing a list of IDs.

    This class encapsulates the information for a single data split (e.g.,
    train, validation, test), including its name and the list of IDs
    belonging to that split. It provides a method to persist the split
    to a CSV file.
    """


    def __init__(self, split_name: str, ids: List[str]):
        """Initializes the SplitFile.

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
        df['id'] = self.ids
        output_path = output_dir / (self.split_name + '.csv')
        df.to_csv(output_path, index=False)

    def __getitem__(self, idx):
        return self.ids[idx]

    def __len__(self):
        return len(self.ids)
