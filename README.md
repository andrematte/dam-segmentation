# Dam Segmentation

[![DOI](https://zenodo.org/badge/849929315.svg)](https://zenodo.org/doi/10.5281/zenodo.13776999)

This repository contains the source code for embankment dam land-cover segmentation based on multispectral remote sensing imagery.
The work was submitted to PeerJ Computer Science as a scientific article.

## Setup

Project requirements are specified in the `pyproject.toml` file. The environment is set up using Poetry.

1. Clone this repository and navigate to its directory
```sh
git clone https://github.com/andrematte/dam-segmentation
cd dam-segmentation
```

2. Create the virtual environment and install dependencies

```sh
poetry install
```

3. Initiate the environment
```sh
poetry shell
```

## Instructions

In order to create the dataset and execute the experiments:

1. Download the dataset from the HuggingFace repository and store it in the `./data/` folder while keeping the file structure. 

> **Dataset available at: https://huggingface.co/datasets/andrematte/dam-segmentation**

Make sure you have git-lfs installed before cloning the dataset repository.

```sh
mkdir data
cd data
git clone https://huggingface.co/datasets/andrematte/dam-segmentation
```

2. Execute `scripts/create_tabular_dataset.py` to create the tabular dataset:

```sh
python scripts/create_tabular_dataset.py
```

Tabular data will be stored in the `.parquet` format and split into train and test data.

## Replicating the Experiments

To replicate each experiment described in the paper:

1. Navigate to the experiment folder with `cd`. Example: `cd experiments/1-feature-subsets`
2. Run the experiment script with `python`. Example: `python subsets_binary.py`

## License

This project is licensed under the terms of MIT License.
