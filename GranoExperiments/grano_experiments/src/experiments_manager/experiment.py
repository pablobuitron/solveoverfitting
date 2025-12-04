import os.path
import pathlib
from pathlib import PosixPath
from typing import Optional
from glob import glob
import logging
import pandas as pd

from ..data import dataset_constants
from ..data.splits.splits_generator import SplitsGenerator
from ..utils import basic_file_utils


class ExperimentDirStructure:
    _TRAINING_RESULTS_DIR = 'training_results'
    _FILENAME_SPLITS_GENERATOR = 'splits_generator.json'
    _DATASET_CONFIG_FILE = 'timeseries_dataset_config.yaml'
    _TRAINING_CONFIG_FILE = 'training_config.yaml'
    _MODEL_ARCHITECTURE_CONFIG_FILE = 'model_architecture_config.yaml'
    _METRICS_CSV_FILE = 'metrics.csv'
    _EXPERIMENT_LOG_FILE = 'experiment.log'

    @staticmethod
    def _find_latest_training_results_version(training_results_dir: pathlib.Path) -> Optional[pathlib.Path]:
        if not training_results_dir.is_dir():
            return None
        versions_dirs = os.listdir(str(training_results_dir))
        if not versions_dirs:
            return None
        else:
            versions_dirs = [vd for vd in versions_dirs if os.path.isdir(training_results_dir / vd)]
        versions_dirs.sort(key=lambda x: int(x.split('_')[1]))
        latest_version = versions_dirs[-1]
        return training_results_dir / latest_version

    def __init__(self, experiment_dir: str | pathlib.Path):
        self.experiment_root_dir = pathlib.Path(experiment_dir)

        self.split_generator_path = self.experiment_root_dir / self._FILENAME_SPLITS_GENERATOR
        self.splits_files = [pathlib.Path(sf) for sf in
                             glob(str(self.experiment_root_dir / dataset_constants.SPLITS_DINRAME / '*'))]
        self.dataset_config_path = self.experiment_root_dir / self._DATASET_CONFIG_FILE
        self.training_config_path = self.experiment_root_dir / self._TRAINING_CONFIG_FILE
        self.model_architecture_config_path = self.experiment_root_dir / self._MODEL_ARCHITECTURE_CONFIG_FILE
        self.log_file_path = self.experiment_root_dir / self._EXPERIMENT_LOG_FILE

        self.training_results_dir = self.experiment_root_dir / self._TRAINING_RESULTS_DIR
        self.train_res_latest_version_dir = self._find_latest_training_results_version(self.training_results_dir)
        self.metrics_csv_path = self.train_res_latest_version_dir / self._METRICS_CSV_FILE if self.train_res_latest_version_dir else None

    def new_training_res_version(self) -> pathlib.Path:
        """Creates a new training_results/version_i directory and returns its path."""
        latest = self._find_latest_training_results_version(self.training_results_dir)
        if latest is None:
            new = self.training_results_dir / 'version_0'
            new.mkdir(parents=True, exist_ok=False)
            return  new
        else:
            latest_number = int(latest.name.split('_')[1])
            new_number = str(latest_number + 1)
            new = self.training_results_dir / f'version_{new_number}'
            new.mkdir(parents=True, exist_ok=False)
            return new

# ------------------------------------------------------------


class Experiment:
    """
    Represents one experiment run and manages its persistence.

    An Experiment encapsulates all on-disk artifacts of a single run.

    The class exposes `persist()` to snapshot the experiment run to disk.

    Attributes:
        experiment_dir (pathlib.Path): Root directory of this experiment. TODO <doc> aggiornare
        splits_generator (Optional[SplitsGenerator]): Split generator used to produce split files, if available.
        splits_generator_data (Optional[dict]): Dictionary with data about the used splits generator, if available.
        splits_files (Optional[list[pathlib.Path]]): Paths to split files (e.g., train/val/test).
        dataset_config (Optional[dict]): Dataset configuration content.
        training_config (Optional[dict]): Training configuration content .
        model_architecture_config (Optional[dict]): Model architecture configuration content.
    """

    @classmethod
    def load(cls, experiment_path: str | pathlib.Path) -> 'Experiment':
        """
        Load an Experiment from an existing directory on disk.

        The loader is intentionally self-contained: it only loads what is present
        in the experiment directory and leaves missing pieces as `None`.

        Args:
            experiment_path (str | pathlib.Path): Path to the experiment directory
                to be loaded.

        Returns:
            Experiment: A populated `Experiment` instance with file paths and any
                available configurations attached.

        Raises:
            FileNotFoundError: If `experiment_path` is not an existing directory.
        """
        experiment_path = pathlib.Path(experiment_path)
        basic_file_utils.directory_check(experiment_path)

        dir_structure = ExperimentDirStructure(experiment_path)

        split_generator_data = basic_file_utils.load_json(dir_structure.split_generator_path) \
            if os.path.exists(dir_structure.split_generator_path) else None

        splits_files = dir_structure.splits_files  # left as is so that the client code is free to choose how to load those files

        dataset_config = basic_file_utils.load_yaml(dir_structure.dataset_config_path) \
            if os.path.exists(dir_structure.dataset_config_path) else None

        training_config = basic_file_utils.load_yaml(dir_structure.training_config_path) \
            if os.path.exists(dir_structure.training_config_path) else None

        model_architecture_config = basic_file_utils.load_yaml(dir_structure.model_architecture_config_path) \
            if os.path.exists(dir_structure.model_architecture_config_path) else None

        training_metrics = basic_file_utils.load_csv(dir_structure.metrics_csv_path) \
            if dir_structure.metrics_csv_path and os.path.exists(dir_structure.metrics_csv_path) else None

        experiment = cls(dir_structure,
                         splits_files=splits_files,
                         splits_generator_data=split_generator_data,
                         dataset_config=dataset_config,
                         training_config=training_config,
                         model_architecture_config=model_architecture_config,
                         training_metrics=training_metrics)

        return experiment

    def __init__(self,
                 dir_structure: ExperimentDirStructure,
                 splits_generator: Optional[SplitsGenerator] = None,
                 splits_generator_data: Optional[dict] = None,
                 splits_files: Optional[list[pathlib.Path]] = None,
                 dataset_config: dict = None,
                 training_config: dict = None,
                 model_architecture_config: dict = None,
                 training_metrics: pd.DataFrame = None) -> None:
        """
        Initialize an Experiment and ensure its directory exists.

        Args:
            experiment_dir (str | pathlib.Path): Path to the experiment directory. TODO <doc> aggiornare
            splits_generator (Optional[SplitsGenerator]):
            splits_generator_data (Optional[dict]):
            splits_files (Optional[list[pathlib.Path]]): List of split file paths
                (e.g., train.csv, val.csv, test.csv). Optional. TODO <doc> spiegare perché questa proprietà fa eccezione,
                                                                           ovvero sono dei path anziché dei file caricati
            dataset_config (Optional[dict]): In-memory dataset configuration. If provided,
                it will be written when calling `persist()`.
            training_config (Optional[dict]): In-memory training configuration. If provided,
                it will be written when calling `persist()`.
            model_architecture_config (Optional[dict]): In-memory model architecture configuration. If provided,
                it will be written when calling `persist()`.
        """

        # --- experiment dir ---
        self.dir_structure = dir_structure
        dir_structure.experiment_root_dir.mkdir(parents=False, exist_ok=True)

        # --- experiment configuration and status ---
        self.splits_generator = splits_generator
        self.splits_generator_data = splits_generator_data
        self.splits_files = splits_files

        self.dataset_config = dataset_config
        self.training_config = training_config
        self.model_architecture_config = model_architecture_config

        self.training_metrics = training_metrics

        self._setup_experiment_logging()

    def persist(self) -> None:
        """
        Persist the experiment state to disk.

        It is safe to call multiple times, but be careful: files are overwritten atomically if they already exist.
        """
        if self.splits_generator:
            self._persist_splits_generator()
        if self.dataset_config:
            basic_file_utils.save_yaml(self.dir_structure.dataset_config_path, self.dataset_config)
        if self.training_config:
            basic_file_utils.save_yaml(self.dir_structure.training_config_path, self.training_config)
        if self.model_architecture_config:
            basic_file_utils.save_yaml(self.dir_structure.model_architecture_config_path,
                                       self.model_architecture_config)

    def _setup_experiment_logging(self):
        """
        Redirect all root logger messages to the experiment logger.

        - Creates a separate FileHandler for the root logger pointing to the current experiment's log file.
        - Ensures only one root FileHandler per experiment.
        - Leaves other root handlers untouched.
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)

        # Rimuove eventuali FileHandler del root logger già associati a esperimenti precedenti
        for handler in list(root_logger.handlers):
            if getattr(handler, "is_experiment_handler", False):
                root_logger.removeHandler(handler)
                handler.close()

        # Crea un nuovo FileHandler separato per il root logger
        file_handler = logging.FileHandler(self.dir_structure.log_file_path)
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        file_handler.is_experiment_handler = True  # marca come handler per esperimenti

        root_logger.addHandler(file_handler)

        # Aggiunge uno StreamHandler se non presente
        if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

    def _persist_splits_generator(self):
        generator_class = self.splits_generator.__class__.__name__
        generator_state = self.splits_generator.__dict__

        dict_to_save = {'class_name': generator_class}
        dict_to_save.update(generator_state)
        for key, value in generator_state.items():
            if isinstance(value, PosixPath):
                dict_to_save[key] = str(value)


        json_path = str(self.dir_structure.split_generator_path)
        basic_file_utils.save_json(json_path, dict_to_save)



