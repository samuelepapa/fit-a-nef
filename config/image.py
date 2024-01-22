import pathlib
from datetime import datetime
from typing import Literal, Type

from absl import logging
from ml_collections import ConfigDict, config_dict

from config.utils import find_env_path


def get_config(mode: Literal["fit", "inspect", "tune", "find_init"] = None):
    if mode is None:
        mode = "fit"
        logging.info(f"No mode provided, using '{mode}' as default")

    cfg = ConfigDict()
    cfg.task = "image"
    cfg.nef_dir = find_env_path("NEF_PATH", "saved_models")
    # Create dir for saving meta-learned initialisations
    cfg.meta_nef_dir = find_env_path("NEF_PATH", "saved_meta_models")
    cfg.seeds = tuple(list(range(20)))

    # Train
    cfg.train = ConfigDict()
    cfg.train.start_idx = 0
    cfg.train.end_idx = 50
    cfg.train.num_steps = 500
    cfg.train.num_parallel_nefs = 5000
    cfg.train.masked_portion = 1.0
    cfg.train.multi_gpu = False
    # put train_to_target_psnr is an optional argument
    cfg.train.train_to_target_psnr = config_dict.placeholder(float)
    cfg.train.check_every = 10
    cfg.train.fixed_init = True
    cfg.train.verbose = True

    # Whether to use meta-learned initialization
    cfg.train.from_meta_init = False
    cfg.train.meta_init_epoch = 10

    # Logging
    cfg.log = ConfigDict()
    cfg.log.images = 500
    cfg.log.metrics = 10
    cfg.log.loss = 10
    cfg.log.use_wandb = False

    # Wandb
    cfg.wandb = ConfigDict()
    cfg.wandb.entity = "neuralfield-wandb"
    cfg.wandb.project = "Classification_tuning_MNIST"
    cfg.wandb.name = "logging_MNIST_dset"

    # Dataset
    cfg.dataset = ConfigDict()
    cfg.dataset.path = "."
    cfg.dataset.name = "CIFAR10"
    cfg.dataset.out_channels = 3

    if mode == "tune":
        cfg.job_id = 0

        cfg.store_models = False
        cfg.temporary_dir = "tmp"
        cfg.study_objective = "mnist_siren_classification"

        # Optuna
        cfg.optuna = ConfigDict()
        cfg.optuna.num_trials = 20
        cfg.optuna.seed = 43
        cfg.optuna.n_jobs = 1
        cfg.optuna.direction = ("maximize",)
        cfg.optuna.study_name = "tuning_MNIST_dset"
    elif mode == "inspect":
        cfg.experiment_dir = str(
            pathlib.Path("experiment_dir") / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
    elif mode == "find_init":
        # Meta learning an initialisation
        cfg.meta = ConfigDict()
        cfg.meta.num_outer_steps = 1000
        cfg.meta.num_inner_steps = 2
        cfg.meta.inner_learning_rate = 1e-2
        cfg.meta.inner_optimizer_name = "sgd"

        cfg.store_model = True

    elif mode == "histograms_num_steps":
        cfg.histograms = ConfigDict()
        cfg.histograms.list_num_steps = (20, 50, 100, 200, 500, 1000, 2000, 5000)
        cfg.histograms.num_bins = 100

        cfg.histograms.num_nefs = 1000

        cfg.train.end_idx = cfg.histograms.num_nefs
        cfg.train.num_steps = cfg.histograms.list_num_steps[-1]
        cfg.train.num_parallel_nefs = cfg.histograms.num_nefs

    elif mode == "histograms_general":
        cfg.histograms = ConfigDict()
        cfg.histograms.num_bins = 100

        cfg.histograms.list_lr = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
        cfg.histograms.list_omega_0 = (1, 2, 3, 4, 5, 6)

        cfg.histograms.num_nefs = 1000
        cfg.histograms.num_steps = 100
        cfg.histograms.target_psnr = 12.5
        cfg.histograms.check_every = 100

        cfg.histogram.end_idx = cfg.histograms.num_nefs
        cfg.histogram.num_steps = cfg.histograms.num_steps
        cfg.histogram.num_parallel_nefs = cfg.histograms.num_nefs

    elif mode != "fit":
        logging.warning(
            f"Unknown mode '{mode}' provided, only 'fit', 'inspect', 'tune', and 'find_init' are supported."
        )

    logging.debug(f"Loaded config: {cfg}")

    return cfg
