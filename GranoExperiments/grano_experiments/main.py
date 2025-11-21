import os
from src.experiments_manager import experiments_manager
import argparse


def new_experiment(experiments_dir, dataset_dir, split_generator):
    manager = experiments_manager.ExperimentManager(experiments_dir, dataset_dir)
    manager.new_experiment(split_generator)


def train_experiment(experiments_dir, dataset_dir, quick_debug=False):
    manager = experiments_manager.ExperimentManager(experiments_dir, dataset_dir, quick_debug=quick_debug)
    manager.load_latest()
    manager.train_experiment()


def main():
    default_experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")
    default_dataset_dir = os.path.join(os.path.dirname(__file__), "grano_it_dataset")

    parser = argparse.ArgumentParser(description="Grano.IT Experiment Manager")
    parser.add_argument("--exp_dir", type=str, default=default_experiments_dir, help="Directory for experiments.")
    parser.add_argument("--dataset_path", type=str, default=default_dataset_dir, help="Path to the dataset.")
    parser.add_argument("--quick_debug", default=False, action="store_true", help="Enable quick debug mode.")
    parser.add_argument("--new", default=False, action="store_true", help="Create a new experiment.")
    parser.add_argument("--split_generator", type=str,
                        default=experiments_manager.SPLITS_GENERATOR_FIELD_AND_YEAR,
                        help="What kind of split generator to use. Possible values:"
                             f" {experiments_manager.SPLITS_GENERATOR_RANDOM}, {experiments_manager.SPLITS_GENERATOR_FIELD_AND_YEAR}")

    args = parser.parse_args()

    if args.new:
        new_experiment(args.exp_dir, args.dataset_path, args.split_generator)

    train_experiment(args.exp_dir, args.dataset_path, quick_debug=args.quick_debug)


if __name__ == '__main__':
    main()

