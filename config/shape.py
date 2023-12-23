import pathlib
from datetime import datetime
from typing import Literal

from absl import logging
from ml_collections import ConfigDict

from config.utils import find_env_path


def get_config(
    mode: Literal["fit", "tune", "inspect", "histograms_num_steps", "histograms_general"] = None
):
    if mode is None:
        mode = "fit"
        logging.info(f"No mode provided, using '{mode}' as default")

    cfg = ConfigDict()
    cfg.task = "shape"
    cfg.nef_dir = find_env_path("NEF_PATH", "saved_models")
    cfg.meta_nef_dir = find_env_path("META_NEF_PATH", "saved_models")
    cfg.seeds = tuple(list(range(10)))

    # Train
    cfg.train = ConfigDict()
    cfg.train.start_idx = 0
    cfg.train.end_idx = 100
    cfg.train.num_parallel_nefs = 10
    cfg.train.num_steps = 5000
    cfg.train.multi_gpu = True  # Whether we're using multiple GPUs
    cfg.train.fixed_init = False
    cfg.train.verbose = False
    cfg.train.num_points = (2048, 2048)  # Number of points in the mesh and outside the mesh
    # Whether to use meta-learned initialization
    cfg.train.from_meta_init = False
    cfg.train.meta_init_epoch = 10
    cfg.train.train_to_target_iou = False
    cfg.train.check_every = 10

    cfg.mode = mode
    cfg.task = "shape"

    # Dataset
    cfg.dataset = ConfigDict()
    cfg.dataset.path = "."
    cfg.dataset.name = "ShapeNet"
    cfg.dataset.out_channels = 1

    # Logging
    cfg.log = ConfigDict()
    cfg.log.meshes = 2500
    cfg.log.metrics = 1000
    cfg.log.loss = 250
    cfg.log.use_wandb = False

    # Wandb
    cfg.wandb = ConfigDict()
    cfg.wandb.use = True
    cfg.wandb.entity = "neuralfield-wandb"
    cfg.wandb.project = "ShapeNetVal"
    cfg.wandb.name = "logging_ShapeNet_dset"

    if mode == "inspect":
        cfg.experiment_dir = pathlib.Path("visualizations") / datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    elif mode == "tune":
        cfg.job_id = 0

        cfg.store_models = False
        cfg.store_meshes = True

        cfg.log.shapes_temp_dir = find_env_path("SHAPES_TEMP_DIR", "shapes_temp_dir")

        # Optuna
        cfg.optuna = ConfigDict()
        cfg.optuna.num_trials = 1
        cfg.optuna.n_jobs = 1
        cfg.optuna.direction = ("maximize",)
        cfg.study_objective = "simple_shape"
        cfg.optuna.study_name = "validation_iou"

    elif mode == "histograms_num_steps":
        cfg.histograms = ConfigDict()
        cfg.histograms.list_num_steps = (20, 50, 100, 200, 500, 1000, 2000, 5000)
        cfg.histograms.num_bins = 100

        cfg.histograms.num_nefs = 10

        cfg.histograms.end_idx = cfg.histograms.num_nefs
        cfg.histograms.num_steps = cfg.histograms.list_num_steps[-1]
        cfg.histograms.num_parallel_nefs = cfg.histograms.num_nefs

    elif mode == "histograms_general":
        cfg.histograms = ConfigDict()
        cfg.histograms.num_bins = 100

        # cfg.histograms.list_lr = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
        # cfg.histograms.list_omega_0 = (1, 2, 3, 4, 5, 6)
        cfg.histograms.list_num_steps = (20, 200, 500, 1000, 2000, 5000)

        cfg.histograms.num_nefs = 500
        cfg.histograms.start_idx = 0
        cfg.histograms.end_idx = 500
        cfg.histograms.num_parallel_nefs = 50
        cfg.histograms.num_steps = 1000
        cfg.histograms.target_iou = 0.4
        cfg.histograms.check_every = 100

    elif mode != "fit":
        logging.warning(f"Unknown mode '{mode}' provided, only 'fit' and 'inspect' are supported.")

    logging.debug(f"Loaded config: {cfg}")

    return cfg
