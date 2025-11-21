import datetime
from typing import Optional

import pathlib
from . import experiment
from ..data import dataset_index, dataset_constants, timeseries_dataset
from ..data.splits import random_splits_generator, dataset_split, field_splits_generator, grouped_split_generator
from ..utils import basic_file_utils
from ..data.splits import splits_generator
from ..train import train
from ..data.splits import generators_factory

# constants related to project structure
_TIMESERIES_DATASET_CONFIG_PATH = pathlib.Path(__file__).parent.parent / 'config/timeseries_dataset_config.yaml'
_TRAINING_CONFIG_PATH = pathlib.Path(__file__).parent.parent / 'config/training_config.yaml'
_MODEL_ARCHITECTURE_CONFIG_PATH = pathlib.Path(__file__).parent.parent / 'config/model_architecture_config.yaml'

# strings supported for instantiating splits generators
SPLITS_GENERATOR_RANDOM = 'random'
SPLITS_GENERATOR_FIELD_AND_YEAR = 'field_and_year'
SPLITS_GENERATOR_GROUPED = 'grouped'

class ExperimentManager:
    """Manager for experiment lifecycle and orchestration within the Grano.IT pipeline.

    The ExperimentManager handles the creation, configuration, and execution of machine learning
    experiments. It acts as the main controller that ties together dataset generation, split
    management, configuration loading, and training routines. Each experiment is encapsulated
    in its own directory and persisted with configuration, dataset splits, and training/testing results.

    Typical workflow:
      1. Initialize the ExperimentManager with dataset and experiment root directories.
      2. Create a new experiment with a specified split generator.
      3. Load the latest or a specific experiment if needed.
      4. Train the experiment, automatically handling dataset building and configuration loading.

    Attributes:
        experiments_root_dir (pathlib.Path): Root directory where experiments are created and stored.
        dataset_root_dir (pathlib.Path): Root directory of the dataset, containing subfolders for yields,
            static and dynamic features.
        dataset_index_path (pathlib.Path): Path to the dataset index CSV used to map samples.
        quick_debug (bool): If True, enables a lightweight mode for quick testing (e.g., reduced dataset size,
            fewer epochs).
        experiment (Optional[experiment.Experiment]): The currently active experiment instance, or None
            if no experiment is loaded.

    Constants:
        SPLITS_GENERATOR_RANDOM (str): Supported split generator string identifier for random splits.
    """

    @staticmethod
    def _new_experiment_name() -> str:
        """Generate a unique, timestamp-based experiment directory name.

        The format is `experiment|YYYYMMDD-HHMM`. The lexicographical ordering guarantees
        that the most recent experiment name sorts last, enabling simple "latest" lookup.

        Returns:
            str: The generated experiment name, e.g., "experiment|20250130-1735".
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        return f"experiment|{timestamp}"

    @staticmethod
    def _find_latest_experiment(experiments_root_dir: pathlib.Path) -> Optional[str]:
        """Find the most recent experiment folder under a given root directory.

        An experiment is considered valid if its directory name starts with "experiment|".
        Given the timestamp naming convention, the latest experiment is the max() by name.

        Args:
            experiments_root_dir (pathlib.Path): Root directory containing experiment subdirectories.

        Returns:
            Optional[str]: The latest experiment directory name if found; otherwise, None.
        """
        subdirs = [d for d in experiments_root_dir.iterdir() if d.is_dir()]
        names = [d.name for d in subdirs if d.name.startswith("experiment|")]

        if not names:
            return None

        return max(names)  # thanks to YYYYMMDD-HHMM format

    @staticmethod
    def _init_splits_generator(generator_name: str) -> splits_generator.SplitsGenerator:
        """Initialize a dataset splits generator.

        Args:
            generator_name (str): The name of the generator to initialize.
                                  The supported generators strings are module
                                   level constants that start with SPLIT_GENERATOR

        Returns:
            SplitsGenerator: An instance of the requested split generator.

        Raises:
            ValueError: If the generator_name is not supported.
        """
        if generator_name == SPLITS_GENERATOR_RANDOM:
            return random_splits_generator.RandomSplitGenerator()
        if generator_name == SPLITS_GENERATOR_FIELD_AND_YEAR:
            return field_splits_generator.FieldSplitsGenerator()
        if generator_name == SPLITS_GENERATOR_GROUPED:
            return grouped_split_generator.GroupedSplitGenerator()
        else:
            raise ValueError(f'Generator "{generator_name}" is not supported.')

    def __init__(self, experiments_root_dir: str,
                 dataset_root_dir: str,
                 dataset_index_relative_path: str = 'index.csv',
                 quick_debug: bool = False):
        """Initialize an ExperimentManager with a root directory.

        Ensures the root directory exists and sets the internal state.

        Args:
            experiments_root_dir : Path to the root directory where experiments will be created and managed.
            dataset_root_dir: Path to the root directory of the dataset.
            dataset_index_relative_path (str): Path to the dataset index file relative to the root.
            quick_debug (bool): If true, enable debug mode (e.g. use a tiny dataset and other related stuff to make debugging and testing quicker.)

        Raises:
            FileNotFoundError: If the root directory cannot be accessed.
        """
        experiments_root_dir = pathlib.Path(experiments_root_dir)
        basic_file_utils.directory_check(experiments_root_dir)
        experiments_root_dir.mkdir(parents=True, exist_ok=True)

        basic_file_utils.directory_check(dataset_root_dir)
        dataset_root_dir = pathlib.Path(dataset_root_dir)

        dataset_index_path = dataset_root_dir / dataset_index_relative_path
        basic_file_utils.file_check(dataset_index_path)

        # configuration
        self.experiments_root_dir = experiments_root_dir
        self.dataset_root_dir = dataset_root_dir
        self.dataset_index_path = dataset_index_path
        self.quick_debug = quick_debug

        # status
        self.experiment = None

    def new_experiment(self, split_generator: str, pre_generate_splits: bool = False):
        """Create and persist a new experiment.

        This method:
        - Creates a new experiment directory with a unique timestamped name.
        - Initializes the chosen split generator.
        - Creates and persists the experiment data.
        - Generates dataset splits and persists them.

        Args:
            split_generator (str): The name of the split generator to use.
            Available names are stored as constants of this module.
            pre_generate_splits (bool): If true, generates splits files.

        Raises:
            FileNotFoundError: If the dataset index file does not exist.
            ValueError: If the split generator is unsupported.
        """
        # Create new experiment directory
        experiment_name = self._new_experiment_name()
        experiment_path = self.experiments_root_dir / experiment_name
        experiment_path.mkdir(parents=False, exist_ok=False)

        # Initialize split generator
        splits_gen = self._init_splits_generator(split_generator)
        if pre_generate_splits:
            splits_gen.generate_splits_files(self.dataset_index_path, experiment_path)

        exp = experiment.Experiment(experiment_path, splits_generator=splits_gen)
        exp.persist()
        self.experiment = exp

    def load_latest(self):
        """Load the latest experiment from the experiments root directory."""
        latest = self._find_latest_experiment(self.experiments_root_dir)
        self.experiment = experiment.load(self.experiments_root_dir / latest)

    def train_experiment(self):

        if not self.experiment:
            raise RuntimeError('Experiment not initialized.')

        self._load_configuration_for_training()
        self._check_configuration_for_training()

        if self.quick_debug:
            self.experiment.training_config['trainer']['max_epochs'] = 3

        self.experiment.persist()

        ds_index = dataset_index.DatasetIndex(self.dataset_index_path)

        if self.experiment.splits_files:
            splits = self._fetch_splits([dataset_constants.TRAIN_SPLIT_NAME, dataset_constants.VALIDATION_SPLIT_NAME])
            splits_gen = None
        else:
            splits = None
            splits_gen = generators_factory.from_dict(self.experiment.splits_generator_data)

        ds_builder = timeseries_dataset.DatasetBuilder(self.dataset_root_dir,
                                                       ds_index,
                                                       ds_splits=splits,
                                                       ds_splits_generator=splits_gen,
                                                       configuration=self.experiment.dataset_config,
                                                       quick_debug=self.quick_debug)

        dataset = ds_builder.build()
        trainer = train.Trainer(training_config=self.experiment.training_config,
                                model_architecture_config=self.experiment.model_architecture_config)
        trainer.train(dataset[dataset_constants.TRAIN_SPLIT_NAME],
                      dataset[dataset_constants.VALIDATION_SPLIT_NAME],
                      output_dir=self.experiment.training_results_dir)

    def _load_configuration_for_training(self):
        # if dataset config is not inside the experiment dir, load the project one
        if self.experiment.dataset_config is None:
            dataset_config = basic_file_utils.load_yaml(_TIMESERIES_DATASET_CONFIG_PATH)
            self.experiment.dataset_config = dataset_config

        # if training config is not inside the experiment dir, load the project one
        if self.experiment.training_config is None:
            training_config = basic_file_utils.load_yaml(_TRAINING_CONFIG_PATH)
            self.experiment.training_config = training_config

        # if model architecture config is not inside the experiment dir, load the project one
        if self.experiment.model_architecture_config is None:
            model_architecture_config = basic_file_utils.load_yaml(_MODEL_ARCHITECTURE_CONFIG_PATH)
            self.experiment.model_architecture_config = model_architecture_config

    def _check_configuration_for_training(self):
        # field_splits_generator_name = field_splits_generator.FieldSplitsGenerator.__name__

        # is_field_split_generator = self.experiment.splits_generator_data['class_name'] == field_splits_generator_name
        # look at the configuration file for more info
        # field_level_aggregation = self.experiment.dataset_config['field_level_aggregation'] or False
        # if  field_level_aggregation and not is_field_split_generator:
        #     raise ValueError(f'Dataset has field level aggregation set to True, '
        #                      f'but split generator is not {field_splits_generator_name}.')
        pass

    def _fetch_splits(self, split_names: list[str]) -> Optional[dict[str, dataset_split.DatasetSplit]]:
        """
        Fetch pre-generated splits with an all-or-nothing policy.

        Loads pre-generated splits from the experiment and returns a
        mapping for the requested `split_names`. If any requested split is missing,
        the method raises instead of returning a partial result.

        Args:
            split_names: Names of the splits to retrieve (e.g., ['train', 'val', 'test']).

        Returns:
            A mapping from split name to `DatasetSplit`.

        Raises:
            RuntimeError: If pre-generated splits are expected but at least one of
                the requested `split_names` is not found.
        """

        all_splits = [dataset_split.from_csv(sp) for sp in self.experiment.splits_files]
        all_splits = {s.split_name: s for s in all_splits}

        result = {}
        for split_name in split_names:
            split = all_splits.get(split_name)
            if not split:
                raise RuntimeError(f'Splits where pre-generated, but dataset split {split_name} was not found.')
            result[split_name] = split

        return result