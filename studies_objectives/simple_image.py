import time
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
from absl import logging

from dataset import path_from_name_idxs
from dataset.data_creation import get_dataset, load_data
from dataset.image_dataset import MEAN_STD_IMAGE_DATASETS, load_attributes
from fit_a_nef import RandomInit, SharedInit, SignalImageTrainer
from tasks.utils import find_seed_idx, get_num_nefs_list, get_signal_idx


def objective(
    trial,
    cfg,
    nef_cfg,
    experiment_folder,
    **kwargs,
):
    # setup the folders for storage
    trial_folder = Path(experiment_folder) / Path(f"trial_{trial.number}")
    trial_folder.mkdir(parents=True, exist_ok=True)

    nefs_folder = Path(trial_folder) / Path("nefs")
    nefs_folder.mkdir(parents=True, exist_ok=True)

    # SAMPLE THE PARAMS HERE
    study_params = {}
    cfg.train.fixed_init = trial.suggest_categorical("fixed_init", [True, False])
    study_params["fixed_init"] = cfg.train.fixed_init

    source_dataset = get_dataset(cfg.dataset)
    signals_in_dset = len(source_dataset)

    num_nefs_list = get_num_nefs_list(
        nef_start_idx=cfg.train.start_idx,
        nef_end_idx=cfg.train.end_idx,
        num_parallel_nefs=cfg.train.num_parallel_nefs,
        signals_in_dset=signals_in_dset,
    )

    rng = jax.random.PRNGKey(cfg.seeds[0])
    train_rng, init_rng = jax.random.split(rng, 2)

    avg_visual_metrics = defaultdict(dict)
    avg_visual_metrics["psnr"] = {
        "means": [],
        "square_means": [],
        "num_samples": [],
    }

    # setup the initializer
    if cfg.train.fixed_init:
        initializer = SharedInit(init_rng)
    else:
        initializer = RandomInit(init_rng)

    images_mean, images_std = MEAN_STD_IMAGE_DATASETS[cfg.dataset.name]

    for i, num_nefs in enumerate(num_nefs_list):
        nef_start_idx = cfg.train.start_idx + sum(num_nefs_list[:i])
        nef_end_idx = nef_start_idx + num_nefs

        assert find_seed_idx(nef_start_idx, signals_in_dset) == find_seed_idx(
            nef_end_idx - 1, signals_in_dset
        ), (
            f"The starting and ending indices of the nefs to train "
            f"({nef_start_idx} - {nef_end_idx}) should be in the same seed idx "
            f"({find_seed_idx(nef_start_idx, signals_in_dset)})."
        )

        start_idx = get_signal_idx(nef_start_idx, signals_in_dset)
        end_idx = get_signal_idx(nef_end_idx - 1, signals_in_dset) + 1

        coords, images, images_shape, _, _ = load_data(
            source_dataset, cfg=cfg, start_idx=start_idx, end_idx=end_idx
        )

        trainer = SignalImageTrainer(
            signals=images,
            coords=coords,
            nef_cfg=nef_cfg,
            scheduler_cfg=cfg.scheduler,
            optimizer_cfg=cfg.optimizer,
            log_cfg=cfg.log,
            initializer=initializer,
            num_steps=cfg.train.num_steps,
            train_rng=train_rng,
            masked_portion=cfg.train.masked_portion,
            verbose=cfg.train.verbose,
            images_mean=images_mean,
            images_std=images_std,
            images_shape=images_shape,
        )

        trainer.compile()

        start_time = time.time()
        trainer.train_model()
        end_time = time.time()

        mean_psnr, square_mean_psnr = trainer.psnr()

        avg_visual_metrics["psnr"]["means"].append(mean_psnr)
        avg_visual_metrics["psnr"]["square_means"].append(square_mean_psnr)
        avg_visual_metrics["psnr"]["num_samples"].append(end_idx - start_idx)

        time_per_image = (end_time - start_time) / (end_idx - start_idx)

        logging.info(f"Run done in {end_time - start_time:.2f} s")
        logging.info(f"Time per image: {1000*time_per_image:.2f} ms")

        attributes = load_attributes(
            source_dataset, start_idx=start_idx, end_idx=end_idx, attribute_name="labels"
        )

        trainer.save(
            nefs_folder / Path(path_from_name_idxs("nefs", nef_start_idx, nef_end_idx)),
            **attributes,
        )

        trainer.clean_up(clear_caches=True)

    # compute the mean and std of the metrics
    avg_psnr = np.average(
        avg_visual_metrics["psnr"]["means"], weights=avg_visual_metrics["psnr"]["num_samples"]
    )

    return avg_psnr
