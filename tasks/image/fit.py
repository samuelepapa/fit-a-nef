import json
import math
import time
from pathlib import Path

import jax
from absl import app, logging
from ml_collections import config_flags

from config import load_cfgs
from dataset import path_from_name_idxs
from dataset.data_creation import get_dataset
from dataset.image_dataset import load_attributes, load_images
from dataset.image_dataset.utils import MEAN_STD_IMAGE_DATASETS
from fit_a_nef import MetaLearnedInit, RandomInit, SharedInit, SignalImageTrainer
from fit_a_nef.utils import get_meta_init, store_cfg
from tasks.utils import find_seed_idx, get_num_nefs_list, get_signal_idx

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/image.py")
_NEF_FILE = config_flags.DEFINE_config_file("nef", default="config/nef.py")
_SCHEDULER_FILE = config_flags.DEFINE_config_file("scheduler", default="config/scheduler.py")
_OPTIMIZER_FILE = config_flags.DEFINE_config_file("optimizer", default="config/optimizer.py")


def main(_):
    cfg, nef_cfg = load_cfgs(_TASK_FILE, _NEF_FILE, _SCHEDULER_FILE, _OPTIMIZER_FILE)

    seeds = cfg.seeds

    storage_folder = Path(cfg.nef_dir) / Path(cfg.dataset.name) / Path(f"{nef_cfg.name}")
    meta_storage_folder = Path(cfg.meta_nef_dir) / Path(cfg.dataset.name) / Path(f"{nef_cfg.name}")
    storage_folder.mkdir(parents=True, exist_ok=True)

    store_cfg(nef_cfg, storage_folder, "nef.json", overwrite=cfg.train.multi_gpu)
    store_cfg(cfg, storage_folder, "cfg.json", overwrite=cfg.train.multi_gpu)

    source_dataset = get_dataset(cfg.dataset)

    signals_in_dset = len(source_dataset)

    total_nefs = cfg.train.end_idx - cfg.train.start_idx

    assert total_nefs / signals_in_dset <= len(seeds), (
        f"Indices requested ({cfg.train.start_idx} - {cfg.train.end_idx}) "
        f"go over the number of seeds ({len(seeds)}) available with dataset "
        f"of size {signals_in_dset}."
    )

    num_nefs_list = get_num_nefs_list(
        nef_start_idx=cfg.train.start_idx,
        nef_end_idx=cfg.train.end_idx,
        num_parallel_nefs=cfg.train.num_parallel_nefs,
        signals_in_dset=signals_in_dset,
    )

    init_rngs_per_seed = [jax.random.PRNGKey(seed) for seed in seeds]

    # If we are using the meta learned initialization, load it.
    if cfg.train.from_meta_init:
        initializers = [
            MetaLearnedInit(get_meta_init(meta_storage_folder, cfg.train.meta_init_epoch))
            for i in range(len(seeds))
        ]
    else:
        if cfg.train.fixed_init:
            initializers = [SharedInit(init_rngs_per_seed[i]) for i in range(len(seeds))]
        else:
            initializers = [RandomInit(init_rngs_per_seed[i]) for i in range(len(seeds))]

    logging.info(f"Training {total_nefs} nefs in {len(seeds)} seeds.")

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
        seed_idx = find_seed_idx(nef_start_idx, signals_in_dset)
        seed = seeds[seed_idx]
        train_rng = jax.random.PRNGKey(seed)

        start_idx = get_signal_idx(nef_start_idx, signals_in_dset)
        end_idx = get_signal_idx(nef_end_idx - 1, signals_in_dset) + 1

        coords, images, images_shape, _, _ = load_images(source_dataset, start_idx, end_idx)

        dataset_mean, dataset_std = MEAN_STD_IMAGE_DATASETS[cfg.dataset.name]

        logging.info(f"Training nefs {nef_start_idx} - {nef_end_idx}.")

        if cfg.log.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name + "_" + str(nef_start_idx) + "-" + str(nef_end_idx),
                config={"cfg": dict(cfg), "nef_cfg": dict(nef_cfg)},
            )

        trainer = SignalImageTrainer(
            signals=images,
            coords=coords,
            nef_cfg=nef_cfg,
            scheduler_cfg=cfg.scheduler,
            optimizer_cfg=cfg.optimizer,
            log_cfg=cfg.log,
            train_rng=train_rng,
            initializer=initializers[seed_idx],
            num_steps=cfg.train.num_steps,
            masked_portion=cfg.train.masked_portion,
            images_mean=dataset_mean,
            images_std=dataset_std,
            images_shape=images_shape,
            verbose=cfg.train.verbose,
        )

        logging.info("Compiling training step.")

        start_time = time.time()
        trainer.compile()
        end_time = time.time()

        logging.info(f"Compiling done in {end_time - start_time:.5f}s.")

        logging.info("Training starting.")

        if cfg.train.train_to_target_psnr is not None:
            start_time = time.time()
            trainer.train_model_to_target_psnr(
                cfg.train.train_to_target_psnr, check_every=cfg.train.check_every
            )
        else:
            start_time = time.time()
            trainer.train_model()
        end_time = time.time()

        logging.info(f"Training done in {end_time - start_time:.5f}s.")

        if cfg.log.use_wandb and WANDB_AVAILABLE:
            wandb.summary["training_time"] = end_time - start_time
            wandb.finish()

        attributes = load_attributes(source_dataset, start_idx, end_idx)

        trainer.save(
            storage_folder / Path(path_from_name_idxs("nefs", nef_start_idx, nef_end_idx)),
            **attributes,
        )

        trainer.clean_up(clear_caches=True)

        logging.info(f"Saved nefs {nef_start_idx} - {nef_end_idx}.")


if __name__ == "__main__":
    app.run(main)
