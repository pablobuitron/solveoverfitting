import json

import yaml
import pathlib
from glob import glob

def load_yaml(path):
    file_check(path)
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(path: str | pathlib.Path, content: dict):
    with open(path, 'w') as f:
        yaml.dump(content, f)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(path, content):
    with open(path, 'w') as f:
        json.dump(content, f)


def directory_check(directory_path: str | pathlib.Path):
    directory_path = pathlib.Path(directory_path)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory_path} does not exist")
    elif not directory_path.is_dir():
        raise NotADirectoryError(f"Directory {directory_path} is not a directory")

def file_check(file_path: str | pathlib.Path):
    file_path = pathlib.Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist")
    elif file_path.is_dir():
        raise IsADirectoryError(f"File {file_path} is not a file")

def glob_single_file(search_string: str | pathlib.Path) -> pathlib.Path:
    files = glob(str(search_string))
    if len(files) == 0:
        raise FileNotFoundError(f"File searched with {search_string} does not exist")
    elif len(files) > 1:
        raise RuntimeError(f"More than one file searched with {search_string} found")
    file = files[0]
    return pathlib.Path(file)