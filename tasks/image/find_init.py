import json
import time
from pathlib import Path

import jax
import matplotlib.pyplot as plt
from absl import app, logging
from fast_fitting import MetaInitImageTrainer
from ml_collections import config_flags

from config import load_cfgs, store_cfg
from config.optimizer import get_config as get_optimizer_config
from config.scheduler import get_config as get_scheduler_config
from dataset.data_creation import get_dataset
from dataset.image_dataset import load_images

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/image.py")
_NEF_FILE = config_flags.DEFINE_config_file("nef", default="config/nef.py")
_SCHEDULER_FILE = config_flags.DEFINE_config_file("scheduler", default="config/scheduler.py")
_OPTIMIZER_FILE = config_flags.DEFINE_config_file("optimizer", default="config/optimizer.py")


def main(_):
    # Get configuration files
    cfg, nef_cfg = load_cfgs(_TASK_FILE, _NEF_FILE, _SCHEDULER_FILE, _OPTIMIZER_FILE)

    # Store the configs used for the meta initialization.
    storage_folder = Path(cfg.meta_nef_dir) / Path(cfg.dataset.name) / Path(f"{nef_cfg.name}")
    storage_folder.mkdir(parents=True, exist_ok=True)

    image_storage_folder = (
        Path(cfg.meta_nef_dir) / Path(cfg.dataset.name) / Path(f"{nef_cfg.name}/reconstructions")
    )
    image_storage_folder.mkdir(parents=True, exist_ok=True)

    store_cfg(nef_cfg, storage_folder, "meta_init_nef_cfg.json")
    store_cfg(cfg, storage_folder, "meta_init_cfg.json")

    source_dataset = get_dataset(cfg.dataset)

    # Total number of images to find the initialization for
    total_images = cfg.train.end_idx - cfg.train.start_idx

    # Split the images into chunks of size cfg.train.num_parallel_nefs
    num_images_list = [cfg.train.num_parallel_nefs] * (total_images // cfg.train.num_parallel_nefs)

    # Add the remainder images if any are left
    if total_images % cfg.train.num_parallel_nefs != 0:
        num_images_list.append(total_images % cfg.train.num_parallel_nefs)

    # We only train on a single seed here, since we are optimizing the initialization. Hence we start from 0.
    assert isinstance(cfg.seeds[0], int), "Seed must be an int."
    global_start_idx = 0

    # Obtain first example batch of images to be used in inner loop, used in compilation
    start_idx = global_start_idx
    end_idx = start_idx + num_images_list[0]
    logging.info(
        f"Example batch used for compilation with start idx: {start_idx}, end idx: {end_idx}"
    )
    coords, images, image_shape, _ = load_images(source_dataset, start_idx, end_idx)

    # Initialize the trainer using the example batch.
    trainer = MetaInitImageTrainer(
        coords=coords,
        nef_cfg=nef_cfg,
        outer_scheduler_cfg=cfg.scheduler,
        outer_optimizer_cfg=cfg.optimizer,
        inner_optimizer_name=cfg.meta.inner_optimizer_name,
        inner_learning_rate=cfg.meta.inner_learning_rate,
        num_inner_steps=cfg.meta.num_inner_steps,
        seed=cfg.seeds[0],
        masked_portion=cfg.train.masked_portion,
    )

    # Compile the training step for the outer loop.
    logging.info("Compiling training step.")
    start_time = time.time()
    trainer.compile(coords, images)
    end_time = time.time()
    logging.info(f"Compiling done in {end_time - start_time:.2f}s.")

    # Obtain set of validation images.
    val_coords, val_images, _, _ = load_images(source_dataset, 0, sum(num_images_list[:50]))
    _, val_loss = trainer.train_inner_models(
        trainer.inner_optimizer,
        trainer.state.params,
        val_coords,
        val_images,
        num_steps=cfg.meta.num_inner_steps,
    )
    logging.info(f"Validation loss for {cfg.meta.num_inner_steps} inner steps: {val_loss:.4f}.")

    # Repeat the outer training step for cfg.meta.num_outer_steps times, each outer step consists of a loop over the
    # entire dataset (or subset in this case).
    logging.info("Training starting.")
    for i in range(cfg.meta.num_outer_steps):
        for j, num_images in enumerate(num_images_list):
            start_idx = global_start_idx + sum(num_images_list[:j])
            end_idx = start_idx + num_images

            coords, images, _, _ = load_images(source_dataset, start_idx, end_idx)

            # Perform outer training step with loaded images
            trainer.state = trainer.outer_train_step(trainer.state, coords, images)

        # Perform validation step.
        params, val_loss = trainer.train_inner_models(
            trainer.inner_optimizer,
            trainer.state.params,
            val_coords,
            val_images,
            num_steps=cfg.meta.num_inner_steps,
        )
        psnr = trainer.psnr(params, val_coords, val_images)
        recon = trainer.recon(params, val_coords)
        logging.info(f"Finished outer step {i}. - val_loss: {val_loss:.4f} val_psnr: {psnr:.4f}.")

        # Every 10 outer steps, store the validation images and reconstructions.
        if i % 10 == 0:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs[0, 0].imshow(jax.numpy.reshape(val_images[0], image_shape))
            axs[0, 1].imshow(jax.numpy.reshape(recon[0], image_shape))
            axs[1, 0].imshow(jax.numpy.reshape(val_images[1], image_shape))
            axs[1, 1].imshow(jax.numpy.reshape(recon[1], image_shape))
            axs[0, 0].set_title("Original")
            axs[1, 0].set_title("Original")
            axs[0, 1].set_title("Reconstruction")
            axs[1, 1].set_title("Reconstruction")
            plt.savefig(image_storage_folder / Path(f"val_recon_{i}.png"))
            plt.close()

            # Store the meta learned initialization of the model.
            if cfg.store_model:
                trainer.save(storage_folder / Path(f"meta_init_epoch_{i}.h5py"))

    # Store the meta learned initialization of the model.
    if cfg.store_model:
        trainer.save(storage_folder / Path(f"meta_init_epoch_{i}.h5py"))


if __name__ == "__main__":
    app.run(main)
