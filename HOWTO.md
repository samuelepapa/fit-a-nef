# How to use this repository

Here are more details on this repository, with some more examples of how to use it to the max.

## Config management

We use `ml_collections` to manage the configuration files.

## DATA_PATH and NEF_PATH

To make running this code easier on different machines or clusters, we use environment variables to specify the paths to the datasets and neural fields.

The `DATA_PATH` environment variable specifies the path to the datasets. The `NEF_PATH` environment variable specifies the path to the neural fields.

Given a `NEF_PATH`, the neural fields are stored in folders with the following structure:

```bash
NEF_PATH/CIFAR10/SIREN
NEF_PATH/CIFAR10/MLP
NEF_PATH/MNIST/SIREN
```

## Neural fields, optimizers, and schedulers

The neural fields, optimizer, and scheduler are interchangeable and independent from the task. This is why they have a separate configuration file each.

To select a specific neural field architecture, optimizer, or scheduler, we use the following syntax:

```bash
--nef=config/nef.py:SIREN
--optimizer=config/optimizer.py:Adam
--scheduler=config/scheduler.py:constant_scheduler
```

Then, to specify the parameters of the neural field, optimizer, or scheduler, we use the following syntax:

```bash
--nef.params.hidden_size=256
--optimizer.params.eps=1e-10
--scheduler.params.value=1000
```

### Tasks

Each task is meant to be a different modality, or sufficiently different method of training. In this reposutory we show image and shape fitting, but you can add your own tasks.

We also allow for large-scale tuning of hyperparameters, on top of being also able to just visualize the reconstructions of the neural fields.

To switch between fitting, tuning or visualization, we use the following syntax:

```bash
--task=config/image.py:fit
--task=config/image.py:tune
--task=config/image.py:inspect
```

The same can be done for the shape task.

### Datasets

Datasets are mostly consistent given a task. Therefore, we do not have separate configuration files for them. Instead, we use the following syntax:

```bash
--task.dataset.name="MNIST"
--task.dataset.out_channels=1
--task.dataset.path="mnist"
```

The `--task.dataset.path` is the path to the dataset, relative to the `DATA_PATH` environment variable. This is to ensure that the datasets are always in the same place, regardless of the machine.

## Fitting

```bash
DATA_PATH="./data/" python tasks/image/fit.py --task=config/image.py:fit --nef=config/nef.py:SIREN --task.seeds='(0,1,2,3,4)' --task.train.start_idx=0 --task.train.end_idx=140000 --task.train.num_parallel_nefs=2000 --task.dataset.name="MNIST" --task.dataset.out_channels=1 --task.dataset.path="." --task.nef_dir="saved_models/example"
```

First, we set the `DATA_PATH="./data/"` environment variable to indicate the root folder where all our datasets are stored. For MNIST and CIFAR10 the dataset is downloaded in that folder automatically.

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

## Logging

We use `wandb` for logging. Simply log into you account and use the `--task.wandb.(...)` config options to specify the project and entity.

## Hyperparameter tuning

TODO
