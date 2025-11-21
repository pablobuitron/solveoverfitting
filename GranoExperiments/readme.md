To get more infos on how to run the code:
```
 python grano_experiments/main.py --help
```
## Setting up dataset and experiments directories

By default, the script searches for the dataset directory and the experiments directory at the same level of main.

Example:
```
.
├── conda_environment.yaml
├── grano_experiments
|       ├── __init__.py
│       ├── experiments -> (symlink) /Users/gtaddei/Desktop/Progetti/Grano.IT/Risorse/Experiments
│       ├── grano_it_dataset -> (symlink )/Users/gtaddei/Desktop/Progetti/Grano.IT/Risorse/Datasets/grano_it_dataset
│       ├── main.py
│       └── src
└── readme.md
```

So, tu run the code, there are two possibilities:
1. create the symlinks 'experiments' and 'grano_it_dataset' inside 'grano_experiments'
2. specify the paths to the experiments and dataset directories as parameters of main.py
```
--dataset_path '/path/to/the/hdf5/files/dataset' --exp_dir '/path/to/parent/directories/where/experiments/will/be/stored'
```

## Run An Experiment
To create and run a new experiment:
```
python grano_experiments/main.py --new 
```

To re-run latest experiment:
To create and run a new experiment:
```
python grano_experiments/main.py
```

To run a new experiment with a specified split generator:
```
python grano_experiments/main.py --new --split_generator <name_of_a_supported_split_generator>
```

## Experiment configuration
When creating a new experiment, the configuration will be extracted from the yaml files inside grano_experiment/src/config.
Inside the experiment directory, those yaml files will be stored. When the latest experiment is run, the yaml configuration files are took from the experiment directory and NOT from the src/config directory.