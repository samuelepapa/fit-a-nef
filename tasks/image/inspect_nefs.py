import json
import os
import time
from functools import partial
from glob import glob
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as T
from absl import app, flags, logging
from ml_collections import ConfigDict, config_dict, config_flags
from PIL import Image

from config.utils import load_cfgs
from dataset.data_creation import get_dataset
from dataset.image_dataset import load_images
from dataset.image_dataset.utils import MEAN_STD_IMAGE_DATASETS, unnormalize_image
from dataset.utils import start_end_idx_from_path
from fit_a_nef import SharedInit, SignalImageTrainer

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/image.py:inspect")
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

    storage_folder = (
        Path(cfg.get("nef_dir", "."))
        / Path(dset_cfg.get("name", "MNIST"))
        / Path(nef_cfg.get("name", "SIREN"))
    )

    seed = cfg.get("seeds", [0])[0]

    nef_cfg = ConfigDict(json.load(open(storage_folder / "nef.json")))

    source_dataset = get_dataset(cfg.dataset)

    nef_paths = sorted(
        glob(os.path.join(storage_folder, "*.hdf5")),
        key=lambda x: int(x.split("_")[-1].split("-")[0]),
    )
    num_paths = len(nef_paths)

    new_rng, rng = jax.random.split(jax.random.PRNGKey(seed))
    idxs_to_plot = jax.random.randint(new_rng, shape=(1,), minval=0, maxval=len(nef_paths))

    unnormalize = partial(unnormalize_image, dataset_name=dset_cfg.get("name", None))

    acc_metrics = {
        "psnr": {
            "mean": jnp.zeros(num_paths),
            "square_mean": jnp.zeros(num_paths),
            "max": 0,
            "min": 100,
        },
        "mse": {
            "mean": jnp.zeros(num_paths),
            "square_mean": jnp.zeros(num_paths),
            "max": 0,
            "min": 100,
        },
        "mae": {
            "mean": jnp.zeros(num_paths),
            "square_mean": jnp.zeros(num_paths),
            "max": 0,
            "min": 100,
        },
        "ssim": {
            "mean": jnp.zeros(num_paths),
            "square_mean": jnp.zeros(num_paths),
            "max": 0,
            "min": 100,
        },
        "simse": {
            "mean": jnp.zeros(num_paths),
            "square_mean": jnp.zeros(num_paths),
            "max": 0,
            "min": 100,
        },
    }

    log_cfg = cfg.get("log", None)

    initializer = SharedInit(
        init_rng=jax.random.PRNGKey(seed),
    )
    train_rng, rng = jax.random.split(rng)

    num_signals_in_dset = len(source_dataset)

    for i, nef_path in enumerate(nef_paths):
        start_idx, end_idx = start_end_idx_from_path(nef_path)

        start_idx = (start_idx) % num_signals_in_dset
        end_idx = (end_idx - 1) % num_signals_in_dset + 1

        logging.info(f"Loading images from {start_idx} to {end_idx}.")

        coords, images, image_shape, _, _ = load_images(source_dataset, start_idx, end_idx)

        mean, std = MEAN_STD_IMAGE_DATASETS[dset_cfg.get("name", "MNIST")]

        trainer = SignalImageTrainer(
            signals=images,
            images_shape=image_shape,
            log_cfg=log_cfg,
            coords=coords,
            nef_cfg=nef_cfg,
            scheduler_cfg=cfg.scheduler,
            optimizer_cfg=cfg.optimizer,
            num_steps=cfg.train.num_steps,
            masked_portion=cfg.train.masked_portion,
            initializer=initializer,
            images_mean=mean,
            images_std=std,
            train_rng=train_rng,
        )

        trainer.load(nef_path)

        mean_psnr, mean_square_psnr = trainer.psnr()
        acc_metrics["psnr"]["mean"] = acc_metrics["psnr"]["mean"].at[i].set(mean_psnr)
        acc_metrics["psnr"]["square_mean"] = (
            acc_metrics["psnr"]["square_mean"].at[i].set(mean_square_psnr)
        )
        acc_metrics["psnr"]["max"] = jnp.maximum(acc_metrics["psnr"]["max"], mean_psnr)
        acc_metrics["psnr"]["min"] = jnp.minimum(acc_metrics["psnr"]["min"], mean_psnr)

        mean_mse, mean_square_mse = trainer.mse()
        acc_metrics["mse"]["mean"] = acc_metrics["mse"]["mean"].at[i].set(mean_mse)
        acc_metrics["mse"]["square_mean"] = (
            acc_metrics["mse"]["square_mean"].at[i].set(mean_square_mse)
        )
        acc_metrics["mse"]["max"] = jnp.maximum(acc_metrics["mse"]["max"], mean_mse)
        acc_metrics["mse"]["min"] = jnp.minimum(acc_metrics["mse"]["min"], mean_mse)

        mean_mae, mean_square_mae = trainer.mae()
        acc_metrics["mae"]["mean"] = acc_metrics["mae"]["mean"].at[i].set(mean_mae)
        acc_metrics["mae"]["square_mean"] = (
            acc_metrics["mae"]["square_mean"].at[i].set(mean_square_mae)
        )
        acc_metrics["mae"]["max"] = jnp.maximum(acc_metrics["mae"]["max"], mean_mae)
        acc_metrics["mae"]["min"] = jnp.minimum(acc_metrics["mae"]["min"], mean_mae)

        mean_ssim, mean_square_ssim = trainer.ssim()
        acc_metrics["ssim"]["mean"] = acc_metrics["ssim"]["mean"].at[i].set(mean_ssim)
        acc_metrics["ssim"]["square_mean"] = (
            acc_metrics["ssim"]["square_mean"].at[i].set(mean_square_ssim)
        )
        acc_metrics["ssim"]["max"] = jnp.maximum(acc_metrics["ssim"]["max"], mean_ssim)
        acc_metrics["ssim"]["min"] = jnp.minimum(acc_metrics["ssim"]["min"], mean_ssim)

        mean_simse, mean_square_simse = trainer.simse()
        acc_metrics["simse"]["mean"] = acc_metrics["simse"]["mean"].at[i].set(mean_simse)
        acc_metrics["simse"]["square_mean"] = (
            acc_metrics["simse"]["square_mean"].at[i].set(mean_square_simse)
        )
        acc_metrics["simse"]["max"] = jnp.maximum(acc_metrics["simse"]["max"], mean_simse)
        acc_metrics["simse"]["min"] = jnp.minimum(acc_metrics["simse"]["min"], mean_simse)

        if i in idxs_to_plot:
            images_to_store = 50

            for j in range(images_to_store):
                new_rng, rng = jax.random.split(rng)
                idx = jax.random.randint(new_rng, shape=(1,), minval=0, maxval=len(images))

                ground_truth = images[idx].reshape(image_shape)
                ground_truth = Image.fromarray(
                    np.array(unnormalize(ground_truth) * 255, dtype=np.uint8).squeeze()
                )
                ground_truth.save(vis_folder / Path(f"{i}_ground_truth_{j}.png"))

                result = trainer.apply_model(coords)[idx].reshape(image_shape)
                result = Image.fromarray(
                    np.array(jnp.clip(unnormalize(result), 0, 1) * 255, dtype=np.uint8).squeeze()
                )
                result.save(vis_folder / Path(f"{i}_result_{j}.png"))

                # rotate coords by random angle

                rot_rng = jax.random.PRNGKey(0)
                angle = jax.random.uniform(rot_rng, minval=0, maxval=2 * jnp.pi)
                R = jnp.array(
                    [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
                )
                rotated_coords = jnp.dot(coords, R)

                result = trainer.apply_model(rotated_coords)[idx].reshape(image_shape)
                result = Image.fromarray(
                    np.array(jnp.clip(unnormalize(result), 0, 1) * 255, dtype=np.uint8).squeeze()
                )
                result.save(vis_folder / Path(f"{i}_rotated_result_{j}.png"))

                x = jnp.linspace(0.5, image_shape[0] - 0.5, image_shape[0] - 1)
                y = jnp.linspace(0.5, image_shape[1] - 0.5, image_shape[1] - 1)
                x, y = jnp.meshgrid(x, y)
                interpolated_coords = jnp.stack([x, y], axis=-1)

                def get_channel(interpolated_coords, image):
                    return jax.scipy.ndimage.map_coordinates(
                        image[:, :, i], interpolated_coords, order=1
                    )

                resampled_channels = []
                for i in range(image_shape[2]):
                    resampled_channels.append(
                        get_channel(
                            interpolated_coords.transpose(), images[idx].reshape(*image_shape)
                        )
                    )
                interpolated_signals = jax.numpy.stack(resampled_channels, axis=-1)

                x = jnp.linspace(
                    -1 + 1 / image_shape[0], 1 - 1 / image_shape[0], image_shape[0] - 1
                )
                y = jnp.linspace(
                    -1 + 1 / image_shape[1], 1 - 1 / image_shape[1], image_shape[1] - 1
                )
                x, y = jnp.meshgrid(x, y)
                interpolated_coords = jnp.stack([x, y], axis=-1)

                interpolated_recon = trainer.apply_model(interpolated_coords.reshape(-1, 2))[
                    idx
                ].reshape(*interpolated_coords.shape[:-1], image_shape[-1])
                result = Image.fromarray(
                    np.array(
                        jnp.clip(unnormalize(interpolated_recon), 0, 1) * 255, dtype=np.uint8
                    ).squeeze()
                )
                result.save(vis_folder / Path(f"{i}_interpolated_result_{j}.png"))

                result = Image.fromarray(
                    np.array(
                        jnp.clip(unnormalize(interpolated_signals), 0, 1) * 255, dtype=np.uint8
                    ).squeeze()
                )
                result.save(vis_folder / Path(f"{i}_interpolated_gt_{j}.png"))

    final_results = {}
    results_log = "metric_name, mean, std, max, min\n"

    for metric_name, metric_dict in acc_metrics.items():
        mean = metric_dict["mean"].mean()
        std = jnp.sqrt(metric_dict["square_mean"].mean() - mean**2)
        final_results[metric_name] = {
            "mean": mean,
            "std": std,
            "max": metric_dict["max"],
            "min": metric_dict["min"],
        }
        results_log += (
            f"{metric_name}, {mean:.2f}, {std:.2f}, {metric_dict['max']}, {metric_dict['min']}\n"
        )
        print(
            f"{metric_name}: {mean:.2f} +/- {std:.2f}. Max: {metric_dict['max']}. Min: {metric_dict['min']}"
        )

    with open(vis_folder / "results.txt", "w") as f:
        f.write(results_log)


if __name__ == "__main__":
    app.run(main)
