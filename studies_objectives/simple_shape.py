import time
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
from absl import logging
from fast_fitting import TuningShapeTrainer

from dataset import path_from_name_idxs
from dataset.data_creation import get_dataset, load_data
from tasks.utils import find_seed_idx, get_num_nefs_list, get_signal_idx


def objective(
    trial,
    cfg,
    nef_cfg,
    classifier_cfg,
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
    init_rng = jax.random.PRNGKey(cfg.seeds[0])

    avg_visual_metrics = defaultdict(dict)
    avg_visual_metrics["iou"] = {
        "means": [],
        "square_means": [],
        "num_samples": [],
    }

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

        # select the seed from the list of available seeds
        seed = cfg.seeds[find_seed_idx(nef_start_idx, signals_in_dset)]

        start_idx = get_signal_idx(nef_start_idx, signals_in_dset)
        end_idx = get_signal_idx(nef_end_idx - 1, signals_in_dset) + 1

        # Create the loader
        loader = load_data(
            source_dataset=source_dataset,
            cfg=cfg,
            start_idx=start_idx,
            end_idx=end_idx,
            batch_size=end_idx - start_idx,
            num_workers=0,
            shuffle=True,  # Unsure how to shuffle this and keep consistent with labeling
        )

        if not cfg.train.fixed_init:
            cur_init_rng, init_rng = jax.random.split(init_rng, 2)
        else:
            cur_init_rng = init_rng

        trainer = TuningShapeTrainer(
            loader=loader,
            nef_cfg=nef_cfg,
            scheduler_cfg=cfg.scheduler,
            optimizer_cfg=cfg.optimizer,
            log_cfg=cfg.log_cfg,
            num_steps=cfg.train.num_steps,
            masked_portion=cfg.train.masked_portion,
            init_rng=cur_init_rng,
            fixed_init=cfg.train.fixed_init,
            seed=seed,
        )

        trainer.compile()

        start_time = time.time()
        trainer.train_model()
        end_time = time.time()

        mean_iou, square_mean_iou = trainer.iou()

        avg_visual_metrics["iou"]["means"].append(mean_iou)
        avg_visual_metrics["iou"]["square_means"].append(square_mean_iou)
        avg_visual_metrics["iou"]["num_samples"].append(end_idx - start_idx)

        time_per_shape = (end_time - start_time) / (end_idx - start_idx)

        logging.info(f"Run done in {end_time - start_time:.2f} s")
        logging.info(f"Time per shape: {1000*time_per_shape:.2f} ms")

        attributes = {"labels": trainer.labels}

        trainer.save(
            nefs_folder / Path(path_from_name_idxs("nefs", nef_start_idx, nef_end_idx)),
            **attributes,
        )
        trainer.clean_up(clear_caches=True)

    # compute the mean and std of the metrics
    avg_iou = np.average(
        avg_visual_metrics["iou"]["means"], weights=avg_visual_metrics["iou"]["num_samples"]
    )

    return avg_iou
