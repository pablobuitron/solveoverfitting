import abc
import pathlib
from collections import OrderedDict
from typing import Iterable, Any
import geopandas as gpd
import h5py


class FilesCache(metaclass=abc.ABCMeta):
    """Generic file-based cache with LRU eviction and context-manager support."""
    # TODO <doc> scrivere da qualche parte che è bene non lasciare in giro riferimenti a file descriptor
    #            onde evitare che quando vengono chiesti nuovi files, quelli vecchi vengano chiusi e ci si ritrovi
    #           con degli oggetti 'fantasma'. Se ci sono dei dati da mantenere, li si preleva e li si salva
    DEFAULT_MAX_OPEN_FILES: int | None = None  # subclasses may override

    def __init__(
        self,
        root_dir: str | pathlib.Path,
        max_open_files: int | None = None,
    ):
        root_dir = pathlib.Path(root_dir)
        if not root_dir.exists() or not root_dir.is_dir():
            raise ValueError(
                f"Root directory {root_dir} should be an existing directory."
            )

        self.root_dir = root_dir
        # If user provided a value → use it
        # Otherwise → use subclass default
        self.max_open_files = (
            max_open_files
            if max_open_files is not None
            else self.DEFAULT_MAX_OPEN_FILES
        )

        self._cache: OrderedDict[str, Any] = OrderedDict()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    @abc.abstractmethod
    def _load_file(self, full_path: pathlib.Path):
        ...

    @abc.abstractmethod
    def _close_file(self, file_obj: Any):
        ...

    def _ensure_capacity(self):
        if self.max_open_files is None:
            return

        while len(self._cache) > self.max_open_files:
            key, file_obj = self._cache.popitem(last=False)
            self._close_file(file_obj)

    def get_file(self, filename: str):
        """Retrieve a file from the cache, loading it if necessary."""
        if filename in self._cache:
            # Move to end (mark as recently used)
            file_obj = self._cache.pop(filename)
            self._cache[filename] = file_obj
            return file_obj

        # Load file
        full_path = self.root_dir / filename
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")

        file_obj = self._load_file(full_path)
        self._cache[filename] = file_obj

        # LRU eviction if needed
        self._ensure_capacity()

        return file_obj

    def iter_files(self, filenames: Iterable[str], sort_files: bool = True):
        """Iterates over files, using get_file() to load/cache each."""
        filenames = sorted(filenames) if sort_files else filenames
        for fname in filenames:
            yield fname, self.get_file(fname)

    def close(self):
        for file_obj in self._cache.values():
            self._close_file(file_obj)
        self._cache.clear()



class Hdf5FilesCache(FilesCache):
    """File cache specialization for HDF5 files using h5py."""

    DEFAULT_MAX_OPEN_FILES = 50

    def _load_file(self, full_path: pathlib.Path):
        return h5py.File(full_path, "r")

    def _close_file(self, file_obj: Any):
        file_obj.close()



class GeoDataFrameFilesCache(FilesCache):
    """
    Cache for loading GeoDataFrames using geopandas.
    """

    DEFAULT_MAX_OPEN_FILES = 300

    def _load_file(self, path: pathlib.Path):
        if not path.exists():
            raise FileNotFoundError(f"GeoDataFrame file not found: {path}")
        return gpd.read_file(path)

    def _close_file(self, file_obj: Any):
        # geopandas handles it autonomously
        pass
