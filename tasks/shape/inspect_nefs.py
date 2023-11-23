import json
import os
from glob import glob
from pathlib import Path

import h5py
import jax
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from absl import app, flags, logging
from fast_fitting.trainers.fit_trainer import ShapeTrainer
from fast_fitting.trainers.tuning_trainer import TuningShapeTrainerClassifier
from ml_collections import ConfigDict, config_dict, config_flags

from config.utils import load_cfgs
from dataset.data_creation import get_dataset
from dataset.nef_dataset.nef_normalization import compute_mean_std_for_nef_dataset
from dataset.shape_dataset.shape_data import load_shapes
from dataset.utils import start_end_idx_from_path

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/shape.py:inspect")
_NEF_FILE = config_flags.DEFINE_config_file("nef", default="config/nef.py")
_SCHEDULER_FILE = config_flags.DEFINE_config_file("scheduler", default="config/scheduler.py")
_OPTIMIZER_FILE = config_flags.DEFINE_config_file("optimizer", default="config/optimizer.py")


def main(_):
    cfg, nef_cfg = load_cfgs(_TASK_FILE, _NEF_FILE, _SCHEDULER_FILE, _OPTIMIZER_FILE)

    vis_folder = Path(cfg.experiment_dir) / Path(f"{cfg.dataset.name}") / Path(f"{nef_cfg.name}")
    vis_folder.mkdir(parents=True, exist_ok=True)

    dset_cfg = cfg.get("dataset", None)
    if dset_cfg is None:
        raise ValueError("No dataset config found in task config.")

    storage_folder = Path(
        "/media/davidknigge/hard-disk1/in-depth-studies/rffnet_number_of_steps_shape"
    )
    # (
    #     Path(cfg.get("nef_dir", "."))
    #     # / Path(dset_cfg.get("name", "ShapeNet"))
    #     # / Path(nef_cfg.get("name", "SIREN"))
    # )
    nef_cfg.params.num_layers = 3
    nef_cfg.params.hidden_dim = 20
    nef_cfg.params.std = 0.28521153152999573
    #    nef_cfg = ConfigDict(json.load(open(storage_folder / "nef.json")))

    source_dataset = get_dataset(cfg.dataset)

    # metadata = compute_mean_std_for_nef_dataset(
    #     storage_folder,
    #     data_split=(0.8, 0.1, 0.1),
    #     num_workers=0,
    #     batch_size=64,
    #     norm_type="per_layer",
    # )

    nef_paths = glob(os.path.join(storage_folder, "trial_45/*/steps_20/*.hdf5"))
    init_rng = jax.random.PRNGKey(42)
    for i, nef_path in enumerate(nef_paths):
        start_idx, end_idx = start_end_idx_from_path(nef_path)

        loader = load_shapes(source_dataset, start_idx, end_idx, batch_size=end_idx - start_idx)

        trainer = TuningShapeTrainerClassifier(
            loader=loader,
            nef_cfg=nef_cfg,
            scheduler_cfg=cfg.scheduler,
            optimizer_cfg=cfg.optimizer,
            init_rng=init_rng,
            num_steps_checkpoints=[0, cfg.train.num_steps],
            nef_dir=storage_folder,
            start_idx=start_idx,
            end_idx=end_idx,
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
