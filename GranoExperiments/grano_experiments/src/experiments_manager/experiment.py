import os.path
import pathlib
from typing import Optional
from glob import glob
from ..data import dataset_constants
from ..data.splits.splits_generator import SplitsGenerator
from ..utils import basic_file_utils

# ------------------------------ STRUCTURAL STUFF  ------------------------------
_TRAINING_RESULTS_DIR = 'training_results'
_FILENAME_SPLITS_GENERATOR = 'splits_generator.json'
_DATASET_CONFIG_FILE = 'dataset_config.yaml'
_TRAINING_CONFIG_FILE = 'training_config.yaml'
_MODEL_ARCHITECTURE_CONFIG_FILE = 'model_architecture_config.yaml'


def _training_results_dir(experiment_dir: pathlib.Path) -> pathlib.Path:
    return experiment_dir / _TRAINING_RESULTS_DIR

def _split_generator_path(experiment_dir: pathlib.Path) -> pathlib.Path:
    return experiment_dir / _FILENAME_SPLITS_GENERATOR

def _dataset_config_path(experiment_dir: pathlib.Path) -> pathlib.Path:
    return experiment_dir / _DATASET_CONFIG_FILE


def _training_config_path(experiment_dir: pathlib.Path) -> pathlib.Path:
    return experiment_dir / _TRAINING_CONFIG_FILE


def _model_architecture_config_path(experiment_dir: pathlib.Path) -> pathlib.Path:
    return experiment_dir / _MODEL_ARCHITECTURE_CONFIG_FILE


# ------------------------------------------------------------


def load(experiment_path: str | pathlib.Path) -> 'Experiment':
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

    splits_files = [pathlib.Path(sf) for sf in glob(str(experiment_path / dataset_constants.SPLITS_DINRAME / '*.csv'))]

    split_generator_path = _split_generator_path(experiment_path)
    split_generator_data = basic_file_utils.load_json(split_generator_path) if os.path.exists(split_generator_path) else None

    dataset_config_path = _dataset_config_path(experiment_path)
    dataset_config = basic_file_utils.load_yaml(dataset_config_path) if os.path.exists(dataset_config_path) else None

    training_config_path = _training_config_path(experiment_path)
    training_config = basic_file_utils.load_yaml(training_config_path) if os.path.exists(training_config_path) else None

    model_arch_conf_path = _model_architecture_config_path(experiment_path)
    model_architecture_config = basic_file_utils.load_yaml(model_arch_conf_path) if os.path.exists(model_arch_conf_path) else None

    experiment = Experiment(experiment_path,
                            splits_files=splits_files,
                            splits_generator_data=split_generator_data,
                            dataset_config=dataset_config,
                            training_config=training_config,
                            model_architecture_config=model_architecture_config)

    return experiment


class Experiment:
    """
    Represents one experiment run and manages its persistence.

    An Experiment encapsulates all on-disk artifacts of a single run.

    The class exposes `persist()` to snapshot the experiment run to disk.

    Attributes:
        experiment_dir (pathlib.Path): Root directory of this experiment.
        training_results_dir (pathlib.Path): Directory where training logs/results are written.
        dataset_config_path (pathlib.Path): Path where the dataset config is stored.
        training_config_path (pathlib.Path): Path where the training config is stored.
        model_architecture_config_path (pathlib.Path): Path where the model architecture config is stored.
        splits_generator (Optional[SplitsGenerator]): Split generator used to produce split files, if available.
        splits_generator_data (Optional[dict]): Dictionary with data about the used splits generator, if available.
        splits_files (Optional[list[pathlib.Path]]): Paths to split files (e.g., train/val/test).
        dataset_config (Optional[dict]): Dataset configuration content.
        training_config (Optional[dict]): Training configuration content .
        model_architecture_config (Optional[dict]): Model architecture configuration content.
    """

    def __init__(self,
                 experiment_dir: str | pathlib.Path,
                 splits_generator: Optional[SplitsGenerator] = None,
                 splits_generator_data: Optional[dict] = None,
                 splits_files: Optional[list[pathlib.Path]] = None,
                 dataset_config: dict = None,
                 training_config: dict = None,
                 model_architecture_config: dict = None) -> None:
        """
         Initialize an Experiment and ensure its directory exists.

        Args:
            experiment_dir (str | pathlib.Path): Path to the experiment directory.
            splits_generator (Optional[SplitsGenerator]):
            splits_generator_data (Optional[dict]):
            splits_files (Optional[list[pathlib.Path]]): List of split file paths
                (e.g., train.csv, val.csv, test.csv). Optional.
            dataset_config (Optional[dict]): In-memory dataset configuration. If provided,
                it will be written when calling `persist()`.
            training_config (Optional[dict]): In-memory training configuration. If provided,
                it will be written when calling `persist()`.
            model_architecture_config (Optional[dict]): In-memory model architecture configuration. If provided,
                it will be written when calling `persist()`.
        """

        # --- experiment dir and internal dir structure---
        experiment_dir = pathlib.Path(experiment_dir)
        experiment_dir.mkdir(parents=False, exist_ok=True)
        self.experiment_dir = experiment_dir

        self.training_results_dir = _training_results_dir(experiment_dir)
        self.splits_generator_path = _split_generator_path(experiment_dir)
        self.dataset_config_path = _dataset_config_path(experiment_dir)
        self.training_config_path = _training_config_path(experiment_dir)
        self.model_architecture_config_path = _model_architecture_config_path(experiment_dir)

        # --- experiment configuration and status ---
        self.splits_generator = splits_generator
        self.splits_generator_data = splits_generator_data
        self.splits_files = splits_files

        self.dataset_config = dataset_config
        self.training_config = training_config
        self.model_architecture_config = model_architecture_config

    def persist(self) -> None:
        """
        Persist the experiment state to disk.

        It is safe to call multiple times, but be careful: files are overwritten atomically if they already exist.
        """
        if self.splits_generator:
            self._persist_splits_generator()
        if self.dataset_config:
            basic_file_utils.save_yaml(self.dataset_config_path, self.dataset_config)
        if self.training_config:
            basic_file_utils.save_yaml(self.training_config_path, self.training_config)
        if self.model_architecture_config:
            basic_file_utils.save_yaml(self.model_architecture_config_path, self.model_architecture_config)

    def _persist_splits_generator(self):
        generator_class = self.splits_generator.__class__.__name__
        generator_state = self.splits_generator.__dict__

        dict_to_save = {'class_name': generator_class}
        dict_to_save.update(generator_state)

        json_path = self.experiment_dir / _FILENAME_SPLITS_GENERATOR
        basic_file_utils.save_json(json_path, dict_to_save)
