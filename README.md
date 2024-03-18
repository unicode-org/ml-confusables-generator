# ML Confusables Generator

A pair of confusables is a pair of characters which might be used in spoofing
attacks due to their visual similarity (for example ‘ν’ and ‘v’). The wide range
of characters supported by Unicode poses security vulnerabilities. Security
mechanisms listed in UTS#39 (UTS #39) use confusable data
(https://www.unicode.org/Public/security/latest/confusables.txt)
to combat such attacks. The purpose of this project is to identify novel pairs
of confusables using representation learning and custom distance metrics.

![](./pictures/confusable_pair.png)

## Table of Contents
- [Getting Started](#getting-started)
    1. [Prerequisite](#prerequisite)
    2. [Installation](#installation)
    3. [Launch Jupyter Notebook Container](#launch-jupyter-notebook-container)
    4. [Launch Command Line Environment Container](#launch-command-line-environment-container)
    5. [Get Interactive Shell in Running Container](#interactive-shell-in-running-container)
    6. [Exit Docker Container](#exit-docker-container)
- [Usage](#usage)
    1. [Han Script Confusable Generation](#han-script-confusable-generation)
    2. [Full Walk-through](#full-walk-through)
    3. [Pre-trained CNN model](#pre-trained-cnn-model)
    4. [Source File Generation](#source-file-generation)
- [Repo Contents](#repo-contents)
    1. [Main Components](#main-components)
    2. [CNN Model Training Scripts](#cnn-model-training-scripts)
    3. [Dataset Source File](#dataset-source-file)
    4. [Shell Scripts](#shell-scripts)
    5. [Unit Tests](#unit-tests)
    6. [Utility Functions](#utility-functions-in-utilspy)
- [Testing](#testing)
    1. [Run All Unit Tests](#run-all-unit-tests)
    2. [Run Individual Unit Test](#run-individual-unit-test)

## Getting Started

### Prerequisite
- [Docker](https://www.docker.com/)
- [Python_FreeType](https://github.com/ldo/python_freetype)
- [Qahirah](https://github.com/ldo/qahirah)
- [pybidi](https://github.com/ldo/pybidi)

### Installation
1. Download and install Docker: [Get Docker Here](https://docs.docker.com/get-docker/).
2. `git clone` and `cd` into git repository.
3. Make sure all submodules are updated: `git submodule update --init --recursive`.

### Launch Jupyter Notebook Container
1. In project source folder, run `./scripts/start.sh`.
2. In any browser, go to `localhost:8888`.
3. Copy the token from terminal to browser to access Jupyter Notebook.

### Launch Command Line Environment Container
1. In project source folder, run `./scripts/start_cli.sh`.
2. Execute setup script `./scripts/setup.sh`.

### Interactive Shell in Running Container
1. Run `docker ps` to get container id/name.
2. Run `docker exec -it [CONTAINER_NAME/ID] /bin/bash`.

### Exit Docker Container
- In Jupyter Notebook terminal, type `ctrl` + `c`.
- In command-line interface, `exit`.

## Usage

### Han Script Confusable Generation
1. From [link](https://drive.google.com/drive/folders/1AEjzkWi9eq8Nxqa99qhoMRbQYl8p5D4M?usp=sharing),
download `full_data.zip` (pre-generated images) file and unzip in `data/` directory.
2. From [link](https://drive.google.com/drive/folders/1QWUDridC499uqmXJJZYKgUf2fPmlgAeB),
download `full_data_triplet1.0_meta.tsv` and `full_data_triplet1.0_vec.tsv` (pre-generated embeddings and labels) into `embeddings/` directory.
3. Create representation clustering object:
    ```python
    from rep_cls import RepresentationClustering
    rc = RepresentationClustering(embedding_file='embeddings/full_data_triplet1.0_vec.tsv',
                                  label_file='embeddings/full_data_triplet1.0_meta.tsv',
                                  img_dir='data/full_data/')
    ```
4. Generate confusables for specific chracter:
    ```python
    rc.get_confusables_for_char('褢')
    >>> ['裹', '裏', '裛', '裏']
    ```

### Full Walk-through
Check `main.ipynb`.

### Pre-trained CNN model
From [link](https://drive.google.com/drive/folders/1ipofZ-BiQzZemFI-aaVnNsciuMUn0zjb?usp=sharing),
download `TripletTransferTF` (pre-trained model) folder into `ckpts/` directory.

### Source file generation
- To regenerate source files, in `source/` directory, run `python generate_source_file.py`.
- To check how the source file is selected, see `source/Radical-stroke_Index_Analysis.ipynb`.

## Repo Contents

#### Main Components
- `main.ipynb`: Notebook for setting up, building and deploying confusable detector. Also serves as tutorial script.
- `vis_gen.py`: Contains VisualGenerator, class for generating visualization of characters.
- `rep_gen.py`: Contains RepresentationGenerator, class for generating representations (embeddings) used for clustering.
- `rep_cls.py`: Contains RepresentationClustering, class for clustering representations and finding confusables.
- `distance_metrics.py`: Contains Distance, factory class that defines distance metrics for different image format. Also contains enumeration class ImgFormat.

#### CNN Model Training Scripts
- `configs/sample_config.ini`: Sample configuration for model training. To start your own training procedure, create new configuration file following the same format.
- `custom_train.py`: Contains ModelTrainer, class that executes training procedure.
- `dataset_builder.py`: Contains DatasetBuilder, class that invokes data pre-processing functions for TensorFlow dataset generation.
- `model_builder.py`: Contains ModelBuilder, class that creates and initialize TensorFlow models.
- `data_preprocessing.py`: Image pre-processing functions.

#### Dataset Source File
- `source/Radical-stroke_Index_Analysis.ipynb`: Jupyter Notebook for radical-stroke analysis and dataset selection.
- `source/generate_source_file.py`: Contains functions that produces the same result as Jupyter Notebook file.
- `source/charset_*k.txt`: Selected Unicode code points.
- `source/randset_*k.txt`: Randomly selected Unicode code points.
- `source/full_dataset.txt`: Full dataset containing 21028 code points, used for clustering.

#### Shell Scripts
Expect all scripts to be executed in base directory. For example, `./scripts/start.sh` instead of `./start.sh`.
- `scripts/start.sh`: Launch a Docker container with Jupyter Notebook.
- `scripts/start_cli.sh`: Launch a Docker container with bash.
- `scripts/setup.sh`: Should run inside the container, setting up the environment and install all packages.
- `scritps/install_fonts.sh`: Install required fonts, included in setup.sh.
- `scripts/download_*.sh`: Scripts for downloading pre-established data, model or embeddings from Google Drive.

#### Unit Tests
- `*_test.py`: Run `python [MODULE]_test.py` for all the unit tests for `[MODULE].py`.

#### Utility functions (in `utils.py`)
- `calculate_from_path`: Calculate distance between the two images specified by file path.
- `train_test_split`: Split dataset (already created) into training and testing datasets.

#### Placeholder Folders
- `data/`: Default visualization directory.
- `ckpts/`: Default model directory.
- `embeddings/`: Default embedding directory.

## Testing
Expect all tests to be run under the CLI container [setup](#launch-command-line-environment-container).

### Run All Unit Tests
In root folder, run `python -m unittest discover -s . -p '*_test.py'`.

### Run Individual Unit Test
In root folder, run `python [MODULE]_test.py`

### Copyright & Licenses

Copyright © 2020-2024 Unicode, Inc. Unicode and the Unicode Logo are registered trademarks of Unicode, Inc. in the United States and other countries.

The project is released under [LICENSE](./LICENSE).

A CLA is required to contribute to this project - please refer to the [CONTRIBUTING.md](https://github.com/unicode-org/.github/blob/main/.github/CONTRIBUTING.md) file (or start a Pull Request) for more information.
