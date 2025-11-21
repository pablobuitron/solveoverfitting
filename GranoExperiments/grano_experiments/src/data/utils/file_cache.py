import pathlib
from typing import Iterable

import h5py


class Hdf5FilesCache:

    def __init__(self, root_dir: str | pathlib.Path):
        root_dir = pathlib.Path(root_dir)
        if not root_dir.exists() or not root_dir.is_dir():
            raise ValueError(f"Root directory {root_dir} should be an existing directory.")

        self.root_dir = root_dir
        self.cache = {}

    def iter_files(self, files: Iterable[str], sort_files: bool = True):
        files = sorted(files) if sort_files else files
        for f in files:
            if f not in self.cache:
                loaded_file = h5py.File(self.root_dir / f, "r")
                self.cache[f] = loaded_file
            yield f, self.cache[f]

    def close_files(self):
        for file in self.cache.values():
            file.close()
        self.cache = {}
