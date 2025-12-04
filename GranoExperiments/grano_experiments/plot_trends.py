import os
from grano_experiments.src.experiments_manager.experiments_manager import ExperimentManager
default_experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")
default_dataset_dir = os.path.join(os.path.dirname(__file__), "grano_it_dataset")

manager = ExperimentManager(default_experiments_dir, default_dataset_dir)
manager.plot_experiment_data(True)