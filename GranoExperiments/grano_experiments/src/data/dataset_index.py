import pathlib
from dataclasses import dataclass

import pandas as pd
from typing import List, Tuple
import ast

ID_SPLITTER = '|'
COLUMN_ID = "id"
COLUMN_LON = 'lon'
COLUMN_LAT = 'lat'
COLUMN_YEAR = 'year'
COLUMN_YIELD = 'yield'
COLUMN_YIELD_FILE = 'yield_file'
COLUMN_STATIC_FILES = 'static_files'
COLUMN_DYNAMIC_FILES = 'dynamic_files'
COLUMN_WHEAT_FIELD_OWNER = 'wheat_field_owner'
COLUMN_WHEAT_FIELD_NAME = 'wheat_field_name'


def get_points(df: pd.DataFrame, to_list: bool=True):
    result = df.index
    if to_list:
        result = result.tolist()
    return result

def get_lon_column(df: pd.DataFrame, to_list: bool=True):
    result = df[COLUMN_LON]
    if to_list:
        result = result.tolist()
    return result

def get_lat_column(df: pd.DataFrame, to_list: bool=True):
    result = df[COLUMN_LAT]
    if to_list:
        result = result.tolist()
    return result

def get_yield_files(df: pd.DataFrame, to_list: bool=True):
    result = df[COLUMN_YIELD_FILE]
    if to_list:
        result = result.to_list()
    return result

def get_associated_files(df: pd.Series) -> Tuple[List[str], List[str]]:
    """Memo: returns static and dynamic files, not the yield files"""
    static_files = df[COLUMN_STATIC_FILES]
    dynamic_files = df[COLUMN_DYNAMIC_FILES]
    # noinspection PyTypeChecker
    return static_files, dynamic_files


def get_point_to_yields(df: pd.DataFrame) -> dict:
    yields = df[COLUMN_YIELD]
    return yields.to_dict()


def get_year(row: pd.Series) -> int:
    # noinspection PyTypeChecker
    return row[COLUMN_YEAR]

def get_yield_file(row: pd.Series) -> str:
    return row[COLUMN_YIELD_FILE]


@dataclass(frozen=True)
class WheatField:
    year: int
    owner: str
    field_name: str

    def __str__(self):
        return ID_SPLITTER.join([str(self.year), self.owner, self.field_name])

    def __hash__(self):
        return hash((self.year, self.owner, self.field_name))

@dataclass
class Id:
    _source: str
    year: int
    owner: str
    field_name: str
    point: str

    @classmethod
    def from_string(cls, string: str) -> "Id":
        split_string = string.split(ID_SPLITTER)
        year = int(split_string[0])
        owner = split_string[1]
        field_name = split_string[2]
        point = split_string[3]
        return cls(string, year, owner, field_name, point)

    def __str__(self):
        return self._source

class DatasetIndex:
    """
    Encapsulates the content of a dataset index CSV file.
    Provides utilities to access metadata and sample identifiers.
    """

    @staticmethod
    def _read_and_preprocess(index_file: pathlib.Path) -> pd.DataFrame:
        df = pd.read_csv(index_file, index_col=COLUMN_ID)

        # Static files and dynamic files should be lists of paths.
        # However, they are originally saved as strings inside the dataframe, so they must be converted to lists.
        df[COLUMN_STATIC_FILES] = df[COLUMN_STATIC_FILES].apply(ast.literal_eval)
        df[COLUMN_DYNAMIC_FILES] = df[COLUMN_DYNAMIC_FILES].apply(ast.literal_eval)
        return df

    def __init__(self, index_file: str | pathlib.Path):
        # noinspection GrazieInspection
        """
        Initialize the DatasetIndex.

        Parameters
        ----------
        index_file : str | Path
            Path to the index CSV file.
        """
        self._df = self._read_and_preprocess(index_file)

    def __len__(self):
        return len(self._df)

    def get_all_ids(self) -> List[str]:
        """
        Returns all sample IDs in the dataset.

        Returns
        -------
        List[str]
            List of IDs corresponding to each sample in the index.
        """
        return self._df.index.tolist()

    def get_entry(self, entry_id: str) -> pd.DataFrame:
        entry = self._df.loc[entry_id]
        return entry

    def get_wheat_fields(self) -> List[WheatField]:
        wheat_fields = []
        for i, row in self._df[[COLUMN_YEAR, COLUMN_WHEAT_FIELD_OWNER, COLUMN_WHEAT_FIELD_NAME]].drop_duplicates().iterrows():
            wheat_field = WheatField(row[COLUMN_YEAR], row[COLUMN_WHEAT_FIELD_OWNER], row[COLUMN_WHEAT_FIELD_NAME])
            wheat_fields.append(wheat_field)
        return wheat_fields

    def get_rows(self, wheat_field: WheatField) -> pd.DataFrame:
        same_year_mask = self._df[COLUMN_YEAR] == wheat_field.year
        same_owner_mask = self._df[COLUMN_WHEAT_FIELD_OWNER] == wheat_field.owner
        same_field_mask = self._df[COLUMN_WHEAT_FIELD_NAME] == wheat_field.field_name
        return self._df[same_year_mask & same_owner_mask & same_field_mask]



