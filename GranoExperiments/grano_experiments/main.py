import os
import pathlib
from argparse import ArgumentError

from src.experiments_manager import experiments_manager
import argparse


def new_experiment(experiments_dir, dataset_dir, split_generator, pre_generate_splits):
    manager = experiments_manager.ExperimentManager(experiments_dir, dataset_dir)
    manager.new_experiment(split_generator, pre_generate_splits=pre_generate_splits)


def train_experiment(experiments_dir, experiment_name, dataset_dir, quick_debug):
    manager = experiments_manager.ExperimentManager(experiments_dir, dataset_dir, quick_debug=quick_debug)
    if experiment_name is None:
        manager.load_latest()
    else:
        experiment_path = pathlib.Path(experiments_dir) / experiment_name
        manager.load(experiment_path)
    manager.train_experiment()


def main():
    default_experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")
    default_dataset_dir = os.path.join(os.path.dirname(__file__), "grano_it_dataset")

    parser = argparse.ArgumentParser(description="Grano.IT Experiment Manager")
    parser.add_argument("--exp_dir", type=str, default=default_experiments_dir, help="Directory for experiments.")
    parser.add_argument("--exp_name", type=str, default=None, help="Name of the experiment to load.")
    parser.add_argument("--dataset_path", type=str, default=default_dataset_dir, help="Path to the dataset.")
    parser.add_argument("--quick_debug", default=False, action="store_true", help="Enable quick debug mode.")
    parser.add_argument("--new", default=False, action="store_true", help="Create a new experiment.")
    parser.add_argument("--pre_generate_splits", default=False, action="store_true", help="Generate splits files during the new experiment phase instead of at runtime.")
    parser.add_argument("--split_generator", type=str,
                        default=experiments_manager.SPLITS_GENERATOR_FIELD_AND_YEAR,
                        help="What kind of split generator to use. Possible values:"
                             f" {experiments_manager.SPLITS_GENERATOR_RANDOM},"
                             f" {experiments_manager.SPLITS_GENERATOR_FIELD_AND_YEAR},"
                             f" {experiments_manager.SPLITS_GENERATOR_TEMPORAL},"
                             f" {experiments_manager.SPLITS_GENERATOR_TEMPORAL_TIDY},")

    args = parser.parse_args()

    if args.new and args.exp_name is not None:
        raise ArgumentError("Cannot specify both --new and --exp_name.")

    if args.new:
        new_experiment(args.exp_dir, args.dataset_path, args.split_generator, args.pre_generate_splits)

    train_experiment(args.exp_dir, args.exp_name, args.dataset_path, args.quick_debug)


if __name__ == '__main__':
    main()

