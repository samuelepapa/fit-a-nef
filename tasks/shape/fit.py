import json
import time
from pathlib import Path

import jax
from absl import app, logging
from ml_collections import config_flags

from config import load_cfgs
from dataset import path_from_name_idxs
from dataset.data_creation import get_dataset
from dataset.image_dataset.image_data import load_attributes
from dataset.shape_dataset.shape_data import load_shapes
from fit_a_nef import MetaLearnedInit, RandomInit, SharedInit
from fit_a_nef.utils import get_meta_init, store_cfg
from tasks.shape.trainer import ShapeTrainer
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

        # Check whether the current model and seed combination has already been trained, if so we skip for now.
        if (
            storage_folder / Path(path_from_name_idxs("nefs", nef_start_idx, nef_end_idx))
        ).exists():
            logging.info(f"Skipping nefs {nef_start_idx} - {nef_end_idx} as they already exist.")
            continue

        loader = load_shapes(source_dataset, start_idx, end_idx, batch_size=num_nefs)

        if cfg.log.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name + "_" + str(nef_start_idx) + "-" + str(nef_end_idx),
                config={"cfg": dict(cfg), "nef_cfg": dict(nef_cfg)},
            )

        trainer = ShapeTrainer(
            loader=loader,
            nef_cfg=nef_cfg,
            scheduler_cfg=cfg.scheduler,
            optimizer_cfg=cfg.optimizer,
            log_cfg=cfg.log,
            train_rng=train_rng,
            initializer=initializers[seed_idx],
            num_steps=cfg.train.num_steps,
            verbose=cfg.train.verbose,
        )

        logging.info("Compiling training step.")

        start_time = time.time()
        trainer.compile()
        end_time = time.time()

        logging.info(f"Compiling done in {end_time - start_time:.2f}s.")

        logging.info("Training starting.")

        start_time = time.time()
        trainer.train_model()
        end_time = time.time()

        logging.info(f"Training done in {end_time - start_time:.2f}s.")

        attributes = {"labels": trainer.labels}

        trainer.save(
            storage_folder / Path(path_from_name_idxs("nefs", nef_start_idx, nef_end_idx)),
            **attributes,
        )
        trainer.clean_up(clear_caches=True)

        logging.info(f"Saved nefs {nef_start_idx} - {nef_end_idx}.")


if __name__ == "__main__":
    app.run(main)
