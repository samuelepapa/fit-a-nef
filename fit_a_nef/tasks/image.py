from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import optax
from absl import logging

from ..initializers import InitModel
from ..metrics import mae, mse, psnr, simse, ssim
from ..trainer import SignalTrainer

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SignalImageTrainer(SignalTrainer):
    """Fit a set of neural fields to a set of images, given a certain initialization method.

    :param signals: The images to fit to.
    :type signals: jnp.ndarray
    :param coords: The coordinates of the images.
    :type coords: jnp.ndarray
    :param train_rng: The random number generator to use.
    :type train_rng: jnp.ndarray
    :param nef_cfg: The config for the neural fields.
    :type nef_cfg: Dict[str, Any]
    :param scheduler_cfg: The config for the scheduler.
    :type scheduler_cfg: Dict[str, Any]
    :param optimizer_cfg: The config for the optimizer.
    :type optimizer_cfg: Dict[str, Any]
    :param initializer: The initializer to use.
    :type initializer: InitModel
    :param log_cfg: The config for the logger. Defaults to None.
    :type log_cfg: Optional[Dict[str, Any]], optional
    :param num_steps: The number of steps to train for. Defaults to 500.
    :type num_steps: int, optional
    :param verbose: Whether to log the training. Defaults to False.
    :type verbose: bool, optional
    :param masked_portion: The portion of the image to mask. Defaults to 0.5.
    :type masked_portion: float, optional
    :param images_shape: The shape of the images. Defaults to None.
    :type images_shape: Optional[Tuple[int, int, int]], optional
    :param images_mean: The mean of the images. Defaults to None.
    :type images_mean: Optional[jnp.ndarray], optional
    :param images_std: The std of the images. Defaults to None.
    :type images_std: Optional[jnp.ndarray], optional
    """

    def __init__(
        self,
        signals: jnp.ndarray,
        coords: jnp.ndarray,
        train_rng: jnp.ndarray,
        nef_cfg: Dict[str, Any],
        scheduler_cfg: Dict[str, Any],
        optimizer_cfg: Dict[str, Any],
        initializer: InitModel,
        log_cfg: Optional[Dict[str, Any]] = None,
        num_steps: int = 500,
        verbose: bool = False,
        masked_portion: float = 0.5,
        images_shape: Optional[Tuple[int, int, int]] = None,
        images_mean: Optional[jnp.ndarray] = None,
        images_std: Optional[jnp.ndarray] = None,
    ):
        """Constructor for."""

        self.masked_portion = masked_portion

        self.log_cfg = log_cfg

        self.max_images_logged = 5
        self.max_recons_metrics = 10

        if verbose:
            assert images_shape is not None
            assert images_mean is not None
            assert images_std is not None

        self.images_shape = images_shape
        self.images_mean = images_mean
        self.images_std = images_std

        num_signals = signals.shape[0]

        super().__init__(
            signals=signals,
            coords=coords,
            initializer=initializer,
            nef_cfg=nef_cfg,
            scheduler_cfg=scheduler_cfg,
            optimizer_cfg=optimizer_cfg,
            num_steps=num_steps,
            train_rng=train_rng,
            verbose=verbose,
            num_signals=num_signals,
        )

    def create_loss(self):
        def loss_fn(params, coords, images):
            y = self.model.apply({"params": params}, coords)
            loss = (images - y) ** 2
            recon_loss = loss.mean()
            return recon_loss

        self.loss_fn = jax.vmap(loss_fn, in_axes=(0, None, 0), out_axes=0)

    def init_model(
        self,
        example_input: jnp.ndarray,
    ):
        super().init_model(
            example_input[: int(self.masked_portion * example_input.shape[0])],
        )

    def process_batch(self, state, coords, images):
        rng, step_rng = jax.random.split(state.rng)
        mask = jax.random.permutation(step_rng, coords.shape[0])
        mask = mask[: int(self.masked_portion * coords.shape[0])]
        # apply_mask
        coords = coords[mask]
        images = images[:, mask]

        return coords, images, rng

    def clean_up(self, clear_caches=True):
        super().clean_up(clear_caches)
        # clear all the points
        del self.signals
        del self.coords

    def verbose_train_model(self):
        for step_num in range(1, self.num_steps + 1):
            # Train model for one epoch, and log avg loss
            self.state, losses = self.train_step(self.state, self.coords, self.signals)

            if self.log_cfg is not None:
                if step_num % self.log_cfg.loss == 0 or (step_num == self.num_steps):
                    learning_rate = self.get_lr()
                    if WANDB_AVAILABLE and self.log_cfg.use_wandb:
                        wandb.log(
                            {
                                "loss": losses.mean(),
                                "lr": learning_rate,
                            },
                            step=step_num,
                        )
                    logging.info(f"Step: {step_num}. Loss: {losses.mean()}. LR {learning_rate}")
                if step_num % self.log_cfg.images == 0 or (step_num == self.num_steps):
                    recons = self.apply_model(self.coords)
                    if WANDB_AVAILABLE and self.log_cfg.use_wandb:
                        wandb.log(
                            {
                                "images": [
                                    wandb.Image(
                                        jax.device_get(self.signals[i]).reshape(self.images_shape)
                                    )
                                    for i in range(min(self.max_images_logged, self.num_signals))
                                ],
                                "recons": [
                                    wandb.Image(
                                        jax.device_get(recons[i]).reshape(self.images_shape)
                                    )
                                    for i in range(min(self.max_images_logged, self.num_signals))
                                ],
                            },
                            step=step_num,
                        )
                    else:
                        logging.info("Wandb not available. Skipping logging images.")

                if step_num % self.log_cfg.metrics == 0 or (step_num == self.num_steps):
                    psnr_mean, psnr_squared_mean = self.psnr()
                    if WANDB_AVAILABLE and self.log_cfg.use_wandb:
                        wandb.log(
                            {
                                "PSNR": psnr_mean,
                            },
                            step=step_num,
                        )
                    logging.info(f"Step: {step_num}. PSNR: {psnr_mean}")

    def train_to_target_psnr(self, target_psnr, check_every, mean, std):
        """
        Args:
            target_psnr (int): The target psnr to reach.
            check_every (int): How often to check the psnr.
            mean (float): The mean of the dataset. Used in the psnr calculation.
            std (float): The std of the dataset. Used in the psnr calculation.

        Returns:
            num_steps (int): The number of steps it took to reach the target psnr.
        """
        num_steps = 0
        average_psnr = 0

        while average_psnr < target_psnr:
            self.state, losses = self.train_step(self.state, self.coords, self.signals)
            if num_steps % check_every == 0:
                average_psnr, _ = self.psnr(mean, std)
                average_val_psnr, _ = self.validation_psnr(mean, std)
                logging.info(
                    f"Step: {num_steps}. PSNR: {average_psnr}, Validation PSNR: {average_val_psnr}"
                )
            num_steps += 1

        return num_steps

    def psnr(self):
        """Calculate the Peak Signal-to-Noise Ratio (PSNR) between the reconstructed images and the
        original images.

        Returns:
            Tuple[float, float]: A tuple containing the mean PSNR and the mean squared PSNR.
        """
        recon = self.apply_model(self.coords)
        metric = psnr(recon, self.signals, self.images_mean, self.images_std)
        return jnp.mean(metric), jnp.mean(jnp.square(metric))

    def mae(self):
        """Calculate the mean absolute error (MAE) between the reconstructed signals and the
        original signals.

        Returns:
            Tuple[float, float]: A tuple containing the mean of the MAE metric and the mean squared MAE metric.
        """
        recon = self.apply_model(self.coords)
        metric = mae(recon, self.signals)
        return jnp.mean(metric), jnp.mean(jnp.square(metric))

    def mse(self):
        """Calculates the mean squared error (MSE) between the reconstructed signals and the
        original signals.

        Returns:
            Tuple[float, float]: A tuple containing the mean MSE and the mean squared MSE.
        """
        recon = self.apply_model(self.coords)
        metric = mse(recon, self.signals)
        return jnp.mean(metric), jnp.mean(jnp.square(metric))

    def ssim(self):
        """Calculates the Structural Similarity Index (SSIM) between the reconstructed images and
        the original signals.

        Returns:
            Tuple[float, float]: A tuple containing the mean SSIM and the mean squared SSIM.
        """
        # Your code here
        recon = self.apply_model(self.coords)
        metric = ssim(
            recon.reshape(-1, *self.images_shape), self.signals.reshape(-1, *self.images_shape)
        )
        return jnp.mean(metric), jnp.mean(jnp.square(metric))

    def simse(self):
        """Calculate the Structural Similarity Index (SIMSE) between the reconstructed image and
        the original signal.

        Returns:
            Tuple[float, float]: A tuple containing the mean SIMSE and the mean squared SIMSE.
        """
        recon = self.apply_model(self.coords)
        metric = simse(recon, self.signals)
        return jnp.mean(metric), jnp.mean(jnp.square(metric))

    def validation_psnr(self):
        """Calculate the Peak Signal-to-Noise Ratio (PSNR) for the validation images.

        Returns:
            Tuple[float, float]: The mean PSNR and the mean squared PSNR.
        """

        # TODO move this in the init, no need to calculate every time.
        x = jnp.linspace(0.5, self.images_shape[0] - 0.5, self.images_shape[0] - 1)
        y = jnp.linspace(0.5, self.images_shape[1] - 0.5, self.images_shape[1] - 1)
        x, y = jnp.meshgrid(x, y)
        coords = jnp.stack([x, y], axis=-1)

        @partial(jax.vmap, in_axes=(None, 0))
        def get_channel(coords, image):
            return jax.scipy.ndimage.map_coordinates(image[:, :, i], coords, order=1)

        resampled_channels = []
        for i in range(self.images_shape[2]):
            resampled_channels.append(
                get_channel(coords.transpose(), self.signals.reshape(-1, *self.images_shape))
            )
        interpolated_signals = jax.numpy.stack(resampled_channels, axis=-1).reshape(
            -1, (coords.shape[0]) * (coords.shape[1]), self.images_shape[-1]
        )

        # TODO move this in the init, no need to calculate every time.
        x = jnp.linspace(
            -1 + 1 / self.images_shape[0], 1 - 1 / self.images_shape[0], self.images_shape[0] - 1
        )
        y = jnp.linspace(
            -1 + 1 / self.images_shape[1], 1 - 1 / self.images_shape[1], self.images_shape[1] - 1
        )
        x, y = jnp.meshgrid(x, y)
        coords = jnp.stack([x, y], axis=-1)

        interpolated_recon = self.apply_model(coords.reshape(-1, 2))
        recon = self.apply_model(self.coords)

        metric = psnr(interpolated_recon, interpolated_signals, self.images_mean, self.images_std)
        return jnp.mean(metric), jnp.mean(jnp.square(metric))
