## Setup

Project requirements are specifiec in the `pyproject.toml` file. The environment is set up using Poetry.

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
![Dataset Link](https://huggingface.co/datasets/andrematte/dam-segmentation)

2. Execute `scripts/create_tabular_dataset.py` to create the tabular dataset:

```sh
python scripts/create_tabular_dataset.py
```

Tabular data will be stored in the `.parquet` format and split into train and test data.
