import json
import os
import time
from glob import glob
from pathlib import Path

import jax
import numpy as np
from absl import app, flags, logging
from ml_collections import ConfigDict, config_dict, config_flags

from config.utils import load_cfgs
from dataset.data_creation import get_dataset
from dataset.shape_dataset.shape_data import load_shapes
from dataset.utils import start_end_idx_from_path
from fit_a_nef import SharedInit, SignalShapeTrainer

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/shape.py:inspect")
_NEF_FILE = config_flags.DEFINE_config_file("nef", default="config/nef.py")
_SCHEDULER_FILE = config_flags.DEFINE_config_file("scheduler", default="config/scheduler.py")
_OPTIMIZER_FILE = config_flags.DEFINE_config_file("optimizer", default="config/optimizer.py")


def main(_):
    cfg, nef_cfg = load_cfgs(_TASK_FILE, _NEF_FILE, _SCHEDULER_FILE, _OPTIMIZER_FILE)

    vis_folder = Path(
        cfg.get("experiment_dir", f"experiment_dir/{time.strftime('%Y-%m-%d_%H-%M-%S')}")
    )
    vis_folder.mkdir(parents=True, exist_ok=True)

    dset_cfg = cfg.get("dataset", None)
    if dset_cfg is None:
        raise ValueError("No dataset config found in task config.")

    storage_folder = Path(
        cfg.get("nef_dir", ".")
        / Path(dset_cfg.get("name", "ShapeNet"))
        / Path(nef_cfg.get("name", "SIREN"))
    )

    nef_cfg = ConfigDict(json.load(open(storage_folder / "nef.json")))
    source_dataset = get_dataset(cfg.dataset)
    seed = cfg.get("seeds", [0])[0]
    init_rng, rng = jax.random.split(jax.random.PRNGKey(seed))
    train_rng, rng = jax.random.split(rng)

    log_cfg = cfg.get("log", None)

    initializer = SharedInit(init_rng=jax.random.split(init_rng))

    nef_paths = sorted(
        glob(os.path.join(storage_folder, "*.hdf5")),
        key=lambda x: int(x.split("_")[-1].split("-")[0]),
    )

    for i, nef_path in enumerate(nef_paths):
        start_idx, end_idx = start_end_idx_from_path(nef_path)

        loader = load_shapes(source_dataset, start_idx, end_idx, batch_size=end_idx - start_idx)
        coords, occupancies, _ = next(iter(loader))

        trainer = SignalShapeTrainer(
            coords=coords,
            occupancies=occupancies,
            train_rng=train_rng,
            nef_cfg=nef_cfg,
            scheduler_cfg=cfg.scheduler,
            optimizer_cfg=cfg.optimizer,
            log_cfg=log_cfg,
            num_steps=cfg.train.num_steps,
            initializer=initializer,
            verbose=cfg.train.verbose,
            num_points=cfg.train.num_points,
        )

        trainer.load(nef_path)
        (vis_folder / Path("meshes")).mkdir(parents=True, exist_ok=True)
        trainer.extract_and_save_meshes(vis_folder / Path("meshes"))

        ious = trainer.iou()
        mean_iou = np.mean(ious)
        max_iou = np.max(ious)
        min_iou = np.min(ious)

        print(
            f"Evaluating NeFs {start_idx} to {end_idx} - mean IOU: {mean_iou} - max IOU: {max_iou} - min IOU: {min_iou}"
        )


if __name__ == "__main__":
    app.run(main)
