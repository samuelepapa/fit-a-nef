# 🚀 `fit-a-nef`

*Quickly fit neural fields to an entire dataset.*

Creators: [Samuele Papa](https://samuelepapa.github.io), [Riccardo Valperga](https://twitter.com/RValperga), [David Knigge](https://twitter.com/davidmknigge), [Phillip Lippe](https://phlippe.github.io/).

[![License: MIT](https://img.shields.io/badge/License-MIT-purple)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Style](https://img.shields.io/badge/code%20style-black-000000)](https://github.com/psf/black)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](CODE_OF_CONDUCT.md)
![Schema](assets/fig-1.png)

Using the ability of JAX to easily parallelize the operations on a GPU with `vmap`, a sizeable set of neural fields can be fit to distinct samples.

The `fit-a-nef` library is designed to easily allow the user to add their own *training task*, *dataset*, and *model*. It provides a uniform format to store and load large amounts of neural fields in a platform-agnostic way. Whether you use PyTorch, Jax or any other framework, the neural fields can be loaded and used in your project.

This repository also provides a simple interface that uses [optuna](https://optuna.org/) to find the best parameters for any neural field while tracking all relevant metrics using [wandb](https://wandb.ai/).

<!-- TABLE OF CONTENTS -->

<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#add-your-own-dataset">Add your own dataset</a></li>
        <li><a href="#add-your-own-model">Add your own model</a></li>
        <li><a href="#find-the-best-parameters">Find the best parameters</a></li>
        <li><a href="#simple-parallelization">Simple parallelization</a></li>
      </ul>
    </li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

## Getting started

### The library

To use the `fit-a-nef` library, simply clone the repository and run:

```bash
python setup.py install
```

This will install the `fit-a-nef` library and all its dependencies. Problems might arise with using jax. In that case, simply install jax manually first, and then run the command above.

### The repository

The repository contains also examples of how to use the library to fit neural fields to images and shapes. For this, more dependencies are required, as the datasets are not included in the library, and some plotting functions are used to visualize the results.

Follow the instructions from INSTALL.md for all the details.

*Note that both `optuna` and `wandb` are optional dependencies, which are required only when running the `tune.py` script. Additionally, the dependencies required for the datasets are not necessary if you are not going to use them.*

## Repository structure

The repository follows the "make it simple, not easy" philosophy.

We prioritize extensibility, and strong independence between packages.
This means that we prefer to have several simple components that have a small set of functionalities and leave the
onus of building powerful software to the end user.

The code does not need to be as concise as it could be.
However, it must always be easy to add new tasks, new neural fields, and new datasets.
Additionally, the dataset format must be standardized across tasks.
Finally, always provide clear documentation and error messages.

The repository is structured as follows:

- `./config`. Configuration files for the tasks.
- `./fit_a_nef`. **Library** for quickly fitting and storing neural fields. *Here you can add the trainer for your own task and your own NeF models*.
- `./dataset`. **Package** to load the targets and inputs used during training. *Here you can add your own dataset*.
- `./tasks`. **Collection** of the tasks that we want to carry out. Both fitting and downstream tasks fit here. *Here you can add the fitting scripts for your own tasks*.
- `./tests`. Tests for the code in the repository.
- `./assets`. Contains the images used in this README.

## Usage

The basic usage of this repository is to fit neural fields to a collection of signals. The current signals supported are images and shapes (through occupancy).

Each task has its own `fit.py` file which is called to fit the neural fields to the provided signals. The `fit.py` file is optimized to provide maximum speed when fitting. Therefore, all logging options have been removed. **NOTE:** The repository will provide example scripts for tracking metrics and large-scale hyperparameter tuning in a future release.

Let us look at a simple example. From the root folder we can run:

```bash
PYTHONPATH="." DATA_PATH="./data/" python tasks/image/fit.py --task=config/image.py:fit --nef=config/nef.py:SIREN --task.seeds='(0,1,2,3,4)' --task.train.start_idx=0 --task.train.end_idx=140000 --task.train.num_parallel_nefs=2000 --task.dataset.name="MNIST" --task.dataset.out_channels=1 --task.dataset.path="." --task.nef_dir="saved_models/example"
```

First, we set `PYTHONPATH="."` to let Python find all the relevant packages (when we provide an install option, this won't be necessary). Then, we set the `DATA_PATH="./data/"` environment variable to indicate the root folder where all our datasets are stored. For MNIST and CIFAR10 the dataset is downloaded in that folder automatically.

Now, we take a look at the other options:

- `--task=config/image.py:fit`: indicates which config file to use and that we are currently trying to `fit` multiple nefs.
- `--nef=config/nef.py:SIREN`: we select `SIREN` as the nef. The `nef.py` config file contains a list of all nefs currently implemented.
- `--task.seeds='(0,1,2,3,4)'`: the seeds that we want to use to initialize the models. These are used to augment the dataset with nefs that use new random initialization. Given that MNIST has 70k images, the total size of the neural dataset with 5 seeds will be 350k nefs.
- `--task.train.start_idx=0 --task.train.end_idx=140000`: with this, we train only the first 140k nefs out of 350k. These two options are used to allow for very simple parallelization of training across multiple devices or even nodes.
- `--task.train.num_parallel_nefs=2000`: the trainer will fit 2k nefs in parallel in the GPU. This is where the speed-up happens, the trainer will `vmap` across 2k nefs. This parameter is dependent on the GPU used, as the processing speed will saturate at a certain point.
- `--task.dataset.name="MNIST" --task.dataset.out_channels=1 --task.dataset.path="."`: these define the dataset used.
- `--task.nef_dir="saved_models/example"`: this is the folder where the neural dataset is stored. If the folder does not exist, it gets created.

### Simple parallelization

To run the program, use:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="." DATA_PATH="./data/" python tasks/image/fit.py --task=config/image.py:fit --nef=config/nef.py:SIREN --task.train.multi_gpu=True --task.seeds='(0,1,2,3,4)' --task.train.start_idx=0 --task.train.end_idx=140000 --task.train.num_parallel_nefs=2000 --task.dataset.name="MNIST" --task.dataset.out_channels=1 --task.dataset.path="." --task.nef_dir="saved_models/example" &
```

and then:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="." DATA_PATH="./data/" python tasks/image/fit.py --task=config/image.py:fit --nef=config/nef.py:SIREN --task.train.multi_gpu=True --task.seeds='(0,1,2,3,4)' --task.train.start_idx=140000 --task.train.end_idx=210000 --task.train.num_parallel_nefs=2000 --task.dataset.name="MNIST" --task.dataset.out_channels=1 --task.dataset.path="." --task.nef_dir="saved_models/example" &
```

These two commands will fit 140k nefs on each GPU, allowing for a direct 2x speed-up. The `start_idx` and `end_idx` options are what ultimately allow this to happen. The user should make sure that no overlap is happening, and that the indices are correct.

## Contributing

Please help us improve this repository by providing your own suggestions or bug report through this repository's GitHub issues system.
Before committing, please ensure to have `pre-commit` installed. This will ensure that the code is formatted correctly and that the tests pass. To install it, run:

```bash
pip install pre-commit
pre-commit install
```

## Code of conduct

Please note that this project has a Code of Conduct. By participating in this project, you agree to abide by its terms.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Acknowledgements and Contributions

We thank [Miltiadis Kofinas](https://mkofinas.github.io/), and David Romero for the feedback during development.
