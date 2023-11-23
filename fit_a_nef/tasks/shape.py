import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging

from fit_a_nef.initializers import InitModel
from fit_a_nef.metrics import iou
from fit_a_nef.trainer import SignalTrainer

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def stack_cut_tensor(
    tensor_list: Sequence[jnp.ndarray], repeat_upto: int = 100000, shuffle: bool = False
):
    """Repeat each tensor by repeat and stack them over batch dimension. Tensor is trimmed to
    minimum size of the tensor list.

    Args:
        tensor_list (Sequence[jnp.ndarray]): The list of tensors to concatenate and pad.
        repeat (int, optional): Number of times to repeat each tensor. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the tensors in the repeating step. Defaults to False.

    Returns:
        jnp.ndarray: The concatenated and cut tensor.
    """
    shapes = np.array([tensor.shape[0] for tensor in tensor_list])
    repeat_nums = np.ceil(repeat_upto / shapes).astype(np.int32)

    if shuffle:
        tensor_list = [
            np.concatenate(
                [np.random.permutation(tensor) for _ in range(repeat_nums[tensor_idx])], axis=0
            )
            for tensor_idx, tensor in enumerate(tensor_list)
        ]
    else:
        tensor_list = [
            np.tile(tensor, (repeat_nums[tensor_idx],) + (1,) * (tensor.ndim - 1))
            for tensor_idx, tensor in enumerate(tensor_list)
        ]
    return np.stack([tensor[:repeat_upto] for tensor in tensor_list])


class SignalShapeTrainer(SignalTrainer):
    def __init__(
        self,
        coords: jnp.ndarray,
        occupancies: jnp.ndarray,
        train_rng: jnp.ndarray,
        nef_cfg: Dict[str, Any],
        scheduler_cfg: Dict[str, Any],
        optimizer_cfg: Dict[str, Any],
        log_cfg: Dict[str, Any],
        initializer: InitModel,
        num_steps: int,
        verbose: bool = False,
        num_points: Tuple[int, int] = (2048, 2048),
    ):
        """
        Args:
            occupancy (jnp.ndarray): The occupancy values to train on.
            coords (jnp.ndarray): The coordinates to train on.
            nef_cfg (Dict[str, Any]): The config for the neural network.
            scheduler_cfg (Dict[str, Any]): The config for the scheduler.
            out_channels (int, optional): The number of output channels. Defaults to 1.
            seed (int, optional): The seed to use. Defaults to 42.
            num_steps (int, optional): The number of steps to train for. Defaults to 20000.

        Raises:
            NotImplementedError: If the model is not implemented.

        Returns:
            None
        """
        self.num_points = num_points
        self.log_cfg = log_cfg
        num_signals = coords.shape[0]

        # Preprocess data by splitting into positive and negative points
        # Repeat and shuffle points to get easier iterator over data (especially for positive points)
        start_time = time.time()
        pin = [coords[i, occupancies[i] >= 0.5] for i in range(occupancies.shape[0])]
        self.points_in = stack_cut_tensor(pin, repeat_upto=100_000, shuffle=True)

        pout = [coords[i, occupancies[i] < 0.5] for i in range(occupancies.shape[0])]
        self.points_out = stack_cut_tensor(pout, repeat_upto=150_000, shuffle=True)

        # Prepare batch data
        self.counter = 0
        Nt_in, Nt_out = self.num_points
        occ0 = jnp.zeros((Nt_out,), dtype=np.float32)
        occ1 = jnp.ones((Nt_in,), dtype=np.float32)
        self.batch_occ = jnp.concatenate([occ0, occ1], axis=0)
        self.batch_range_out = np.array([0, Nt_out - 1], dtype=np.int32)
        self.batch_range_in = np.array([0, Nt_in - 1], dtype=np.int32)
        self.batch_coords = np.zeros((num_signals, Nt_out + Nt_in, 3), dtype=np.float32)

        super().__init__(
            coords=self.batch_coords,
            signals=self.batch_occ,
            num_signals=num_signals,
            nef_cfg=nef_cfg,
            scheduler_cfg=scheduler_cfg,
            optimizer_cfg=optimizer_cfg,
            initializer=initializer,
            train_rng=train_rng,
            num_steps=num_steps,
            verbose=verbose,
        )

    def create_loss(self):
        def loss_fn(params, coords, occupancy):
            y = self.model.apply({"params": params}, coords)

            # Calculate binary cross entropy loss
            recon_loss = optax.sigmoid_binary_cross_entropy(y, jnp.expand_dims(occupancy, -1))
            return recon_loss

        self.loss_fn = jax.vmap(loss_fn, in_axes=(0, 0, None), out_axes=0)

    def init_model(self, example_input: jnp.ndarray):
        example_input = self.batch_coords[0]
        return super().init_model(example_input)

    def compile(self):
        coords = self.ram_process_batch()
        _, _ = self.train_step(self.state, coords, self.batch_occ)

    def ram_process_batch(self):
        Nt_out, Nt_in = self.num_points
        # Determine positive points. For efficiency, we use numpy's slicing when possible
        if self.batch_range_out[-1] > self.batch_range_out[0]:
            self.batch_coords[:, :Nt_out, :] = self.points_out[
                :, self.batch_range_out[0] : self.batch_range_out[-1] + 1
            ]
        else:
            s = self.batch_range_out[-1] + 1
            self.batch_coords[:, : Nt_out - s, :] = self.points_out[:, self.batch_range_out[0] :]
            self.batch_coords[:, Nt_out - s : Nt_out, :] = self.points_out[
                :, : self.batch_range_out[-1] + 1
            ]
        # Determine negative points. For efficiency, we use numpy's slicing when possible
        if self.batch_range_in[-1] > self.batch_range_in[0]:
            self.batch_coords[:, Nt_out:, :] = self.points_in[
                :, self.batch_range_in[0] : self.batch_range_in[-1] + 1
            ]
        else:
            s = self.batch_range_in[-1] + 1
            self.batch_coords[:, Nt_out : Nt_in + Nt_out - s, :] = self.points_in[
                :, self.batch_range_in[0] :
            ]
            self.batch_coords[:, Nt_in + Nt_out - s :, :] = self.points_in[
                :, : self.batch_range_in[-1] + 1
            ]
        # Move batch range for next batch
        self.batch_range_in = np.mod(self.batch_range_in + Nt_in, self.points_in.shape[1])
        self.batch_range_out = np.mod(self.batch_range_out + Nt_out, self.points_out.shape[1])

        return self.batch_coords

    def fast_train_model(self):
        for i in range(1, self.num_steps):
            coords = self.ram_process_batch()
            self.state, _ = self.train_step(self.state, coords, self.batch_occ)

    def verbose_train_model(self):
        for step_num in range(1, self.num_steps + 1):
            # Train model for one epoch, and log avg loss
            coords = self.ram_process_batch()
            self.state, losses = self.train_step(self.state, coords, self.batch_occ)

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

                if step_num % self.log_cfg.metrics == 0 or (step_num == self.num_steps):
                    mean_iou, mean_iou_squared = self.iou()
                    if WANDB_AVAILABLE and self.log_cfg.use_wandb:
                        wandb.log(
                            {
                                "iou": mean_iou,
                            },
                            step=step_num,
                        )
                    logging.info(f"Step: {step_num}. IOU: {mean_iou}")

    def iou(self):
        # Apply model to all coordinates in the dataset for each shape.
        occ_hats = self.apply_model_all_coords()

        metric = iou(self.signals, occ_hats)
        return jnp.mean(metric), jnp.mean(jnp.square(metric))

    def apply_model_all_coords(self):
        return jax.vmap(
            fun=lambda params, coords: self.model.apply({"params": params}, coords), in_axes=(0, 0)
        )(self.state.params, self.coords)

    def train_to_target_iou(self, target_iou, check_every):
        num_steps = 0
        average_iou = 0

        while average_iou < target_iou:
            coords = self.ram_process_batch()
            self.state, losses = self.train_step(self.state, coords, self.batch_occ)
            if num_steps % check_every == 0:
                average_iou, _ = self.iou()
                logging.info(f"Step: {num_steps}. IOU: {average_iou}")
            num_steps += 1

        return num_steps
