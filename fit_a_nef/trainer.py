# Documentation
import json
import pickle
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging

# JAX
from flax.training import train_state

from fit_a_nef.initializers import InitModel, MetaLearnedInit, RandomInit, SharedInit
from fit_a_nef.nef import param_key_dict

# Misc
from fit_a_nef.utils import (
    TrainState,
    flatten_params,
    get_nef,
    get_optimizer,
    get_scheduler,
    unflatten_params,
)


class SignalTrainer:
    def __init__(
        self,
        coords: jnp.ndarray,
        signals: jnp.ndarray,
        nef_cfg: Dict[str, Any],
        scheduler_cfg: Dict[str, Any],
        optimizer_cfg: Dict[str, Any],
        initializer: InitModel,
        train_rng: jnp.ndarray,
        num_signals: int,
        num_steps: int = 20000,
        verbose: bool = False,
    ):
        """
        Base class for training neural networks.
        Args:
            signals (jnp.ndarray): The signal values to train on, i.e. images (pixel values) objects (occupancy).
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
        super().__init__()

        self.coords = coords
        self.signals = signals
        self.train_rng = train_rng
        self.num_signals = num_signals

        self.num_steps = num_steps
        self.scheduler_cfg = scheduler_cfg
        self.optimizer_cfg = optimizer_cfg
        self.initializer = initializer
        self.verbose = verbose

        self.model = get_nef(nef_cfg=nef_cfg)
        self.init_model(example_input=coords)
        self.param_key = partial(
            param_key_dict[nef_cfg.get("name", None)], nef_cfg=nef_cfg.get("params", None)
        )

        self.create_functions()

    def create_functions(self):
        if not hasattr(self, "loss_fn"):
            self.create_loss()

        if not hasattr(self, "train_step"):
            self.create_train_step()

        if not hasattr(self, "train_model"):
            self.create_train_model()

    def create_train_step(self):
        def train_step(state, coords, signals):
            coords, signals, rng = self.process_batch(state, coords, signals)
            # pass the coordinates and the signals all at once
            my_loss = lambda params: self.loss_fn(params, coords, signals).sum()
            # compute gradients wrt state.params
            loss, grads = jax.value_and_grad(my_loss, has_aux=False)(state.params)
            state = state.apply_gradients(grads=grads, rng=rng)
            return state, loss

        self.train_step = jax.jit(train_step)

    def process_batch(self, state, coords, signals):
        rng, cur_rng = jax.random.split(state.rng)
        return coords, signals, rng

    def init_model(
        self,
        example_input: jnp.ndarray,
    ):
        # Initialize model parameters
        params = self.initializer(self.model, example_input, self.num_signals)

        # Initialize optimizer and learning rate schedule
        self.lr_schedule = get_scheduler(self.scheduler_cfg)
        optimizer = get_optimizer(self.optimizer_cfg, self.lr_schedule)

        # Create the train state
        self.state = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer, rng=self.train_rng
        )

    def compile(self):
        """Executes the training function ones to compile the train_step.

        Args:
            None

        Returns:
            None
        """
        _ = self.train_step(self.state, self.coords, self.signals)

    def create_train_model(self):
        """Trains the model for the given number of epochs.

        Args:
            num_epochs (int): The number of epochs to train for.

        Returns:
            None
        """
        if self.verbose:
            self.train_model = self.verbose_train_model
        else:
            self.train_model = self.fast_train_model

    def fast_train_model(
        self,
    ):
        for _ in range(1, self.num_steps + 1):
            self.state, _ = self.train_step(self.state, self.coords, self.signals)

    def verbose_train_model(
        self,
    ):
        for step in range(1, self.num_steps + 1):
            self.state, loss = self.train_step(self.state, self.coords, self.signals)
            if step % 100 == 0:
                logging.info(f"Step {step}, loss {loss}")

    def get_params(self, model_id: Optional[int] = None):
        """Returns the params for a given model ID.

        Args:
            model_id (int): The model ID to get the params for.

        Returns:
            jax.tree_util.PartiallyMutableTree: The params for the given model ID.
        """
        if model_id is None:
            return self.state.params
        else:
            return jax.tree_map(lambda x: x[model_id], self.state.params)

    def get_flat_params(self, model_id: Optional[int] = None) -> Tuple[np.ndarray, str]:
        """Returns the params for a given model ID.

        Args:
            model_id (int): The model ID to get the params for.

        Returns:
            jax.tree_util.PartiallyMutableTree: The params for the given model ID.
        """
        if model_id is None:
            params = self.state.params
        else:
            params = jax.tree_map(lambda x: x[model_id], self.state.params)

        param_config, comb_params = flatten_params(
            params, num_batch_dims=1, param_key=self.param_key
        )
        comb_params = jax.device_get(comb_params)

        return comb_params, param_config

    def apply_model(self, coords: jnp.ndarray, model_id: Optional[int] = None):
        """Applies the model to a given set of coordinates.

        Args:
            model_id (int): The model ID to apply.
            coords (jnp.ndarray): The coordinates to apply the model to.

        Returns:
            jnp.ndarray: The output of the model.
        """
        if model_id is None:
            return jax.vmap(lambda params: self.model.apply({"params": params}, coords))(
                self.state.params
            )
        else:
            return self.model.apply({"params": self.get_params(model_id)}, coords)

    def save(self, path: Path, **kwargs):
        """Save the parameters to a pickle file."""
        param_config, comb_params = flatten_params(
            self.state.params, num_batch_dims=1, param_key=self.param_key
        )
        comb_params = jax.device_get(comb_params)
        param_config = json.dumps(param_config)

        with h5py.File(path, "w") as f:
            f.create_dataset("params", data=comb_params)
            dt = h5py.special_dtype(vlen=str)
            data = f.create_dataset("param_config", (1,), dtype=dt)
            data[0] = param_config
            for key, value in kwargs.items():
                if isinstance(value, jnp.ndarray):
                    value = jax.device_get(value)
                f.create_dataset(key, data=value)

    def load(self, path: Path):
        """Used to load the parameters from a pickle file that can be created using the `save`
        function."""
        with h5py.File(path, "r") as f:
            param_config = json.loads(f["param_config"][0].decode("utf-8"))
            comb_params = f["params"][:]
        params = unflatten_params(param_config, comb_params)
        self.state = self.state.replace(params=params)

    def clean_up(self, clear_caches=True):
        del self.state
        if hasattr(self, "train_step"):
            del self.train_step
        if clear_caches:
            jax.clear_caches()

    def get_lr(self):
        schedule = self.lr_schedule
        if schedule is None:
            logging.warning("No learning rate schedule found.")
            return
        opt_state = self.state.opt_state
        opt_state = [s for s in opt_state if isinstance(s, optax.ScaleByScheduleState)]

        if len(opt_state) == 0:
            logging.warning("No state of a learning rate schedule found.")
            return
        if len(opt_state) > 1:
            logging.warning(
                "Found multiple states of a learning rate schedule. Using the last one."
            )
        step = opt_state[-1].count
        lr = schedule(step)
        return lr


# TODO Properly implement this.
# class MetaTrainer:
#     def __init__(
#         self,
#         coords: jnp.ndarray,
#         nef_cfg: Dict[str, Any],
#         outer_scheduler_cfg: Dict[str, Any],
#         outer_optimizer_cfg: Dict[str, Any],
#         inner_optimizer_name: Literal["sgd", "adam"],
#         inner_learning_rate: float,
#         seed: int = 42,
#         num_inner_steps: int = 3,
#         masked_portion: float = 0.5,
#         weight_decay: float = 0.0,
#         **kwargs,
#     ):
#         """
#         Args:
#             exmp_image (jnp.ndarray): An example image used for model initialization.
#             coords (jnp.ndarray): The coordinates to train on.
#             nef_cfg (Dict[str, Any]): The config for the neural network.
#             scheduler_cfg (Dict[str, Any]): The config for the scheduler.
#             out_channels (int, optional): The number of output channels. Defaults to 1.
#             seed (int, optional): The seed to use. Defaults to 42.
#             num_images (int, optional): The number of images to train on. Defaults to 5.
#             num_steps (int, optional): The number of steps to train for. Defaults to 20000.

#         Raises:
#             NotImplementedError: If the model is not implemented.

#         Returns:
#             None
#         """
#         self.masked_portion = masked_portion
#         self.weight_decay = weight_decay

#         super().__init__(
#             nef_cfg=nef_cfg,
#             scheduler_cfg=outer_scheduler_cfg,
#             optimizer_cfg=outer_optimizer_cfg,
#             seed=seed,
#             num_steps=num_inner_steps,
#         )

#         # Store inner learning rate
#         self.inner_optimizer_name = inner_optimizer_name
#         self.inner_learning_rate = inner_learning_rate

#         # Initialize the model with the example image, creates weights, outer and inner optimizer and scheduler.
#         self.init_model(exmp_points=coords)

#     def init_model(self, exmp_points: jnp.ndarray):
#         # Initialize model parameters
#         rng = jax.random.PRNGKey(self.seed)
#         rng, init_rng = jax.random.split(rng)
#         vmap_init = jax.vmap(self.model.init, in_axes=(0, None))

#         # Initialize model parameters for a single neural field (hence the 1)
#         params = vmap_init(jax.random.split(init_rng, 1), exmp_points)["params"]

#         # Initialize optimizer and learning rate schedule
#         self.outer_lr_schedule = get_scheduler(scheduler_cfg=self.scheduler_cfg)
#         outer_optimizer = get_optimizer(self.optimizer_cfg, self.outer_lr_schedule)

#         # Create the train state for the outer loop
#         self.state = train_state.TrainState.create(
#             apply_fn=None, params=params, tx=outer_optimizer
#         )

#         # We always use a constant schedule with SGD for the inner loop.
#         self.inner_lr_schedule = optax.constant_schedule(self.inner_learning_rate)
#         self.inner_optimizer = getattr(optax, self.inner_optimizer_name)(self.inner_lr_schedule)

#     def create_loss(self):
#         def loss_fn(params, coords, images):
#             y = self.model.apply({"params": params}, coords)
#             loss = (images - y) ** 2
#             recon_loss = loss.mean()
#             return recon_loss

#         self.loss_fn = jax.vmap(loss_fn, in_axes=(0, None, 0), out_axes=0)

#     def train_model(
#         self,
#     ):
#         """The model training loop happens outside of the trainer, since we need to load the data
#         in the outer loop."""
#         raise NotImplementedError("Meta learning is done outside of the trainer.")

#     def train_inner_models(self, inner_optimizer, params, coords, images, num_steps=None):
#         """Train the inner models for a given number of steps. This is mainly used for debugging
#         purposes and to obtain loss values for the inner loop.

#         Args:
#             inner_optimizer (optax.GradientTransformation): The inner optimizer to use.
#             params (jax.tree_util.PartiallyMutableMapping): The outer parameters (initialization) to train on.
#             coords (jnp.ndarray): The coordinates to train on.
#             images (jnp.ndarray): The images to train on.
#             num_steps (int, optional): The number of steps to train for. Defaults to None in which case self.num_steps
#                 is used.
#         """

#         # Broadcast params over batch dimension
#         params = jax.tree_map(
#             lambda x: jnp.broadcast_to(x, (images.shape[0], *x.shape[1:])), params
#         )

#         # Initialize inner optimizer
#         inner_optimizer_state = inner_optimizer.init(params)

#         for inner_step in range(self.num_steps if num_steps is None else num_steps):
#             params, inner_optimizer_state, loss = self.inner_train_step(
#                 params, inner_optimizer_state, coords, images
#             )
#         return params, loss

#     def create_inner_train_step(self):
#         def inner_train_step(inner_params, inner_optimizer_state, coords, images):
#             """Meta learning inner training step.

#             Args:
#                 inner_params (jax.tree_util.PartiallyMutableMapping): The inner parameters to train on.
#                 coords (jnp.ndarray): The coordinates to train on.
#                 images (jnp.ndarray): The images to train on.
#             """
#             # Pass the coordinates and the images all at once
#             my_loss = lambda params: self.loss_fn(params, coords, images).sum()

#             # Compute gradients wrt state.params
#             grad = jax.grad(my_loss, has_aux=False)(inner_params)

#             # Get weight updates
#             updates, inner_optimizer_state = self.inner_optimizer.update(
#                 grad, inner_optimizer_state
#             )

#             # Apply weight updates
#             params = optax.apply_updates(inner_params, updates)
#             return params, inner_optimizer_state, my_loss(params)

#         self.inner_train_step = jax.jit(inner_train_step)

#     def create_outer_train_step(self):
#         def outer_train_step(state, coords, images):
#             """Meta learning outer training step.

#             Args:
#                 state (train_state.TrainState): The train state containing meta-learned init parameters.
#                 coords (jnp.ndarray): The coordinates to train on.
#                 images (jnp.ndarray): The images to train on.
#             """
#             # Pass the coordinates and the images all at once
#             my_loss = lambda params: self.loss_fn(params, coords, images).sum()

#             def inner_train_loop_loss(outer_params):
#                 # Broadcast params over batch dimension
#                 inner_params = jax.tree_map(
#                     lambda x: jnp.broadcast_to(x, (images.shape[0], *x.shape[1:])), outer_params
#                 )

#                 # Initialize inner optimizer
#                 inner_optimizer_state = self.inner_optimizer.init(inner_params)

#                 # Compute gradients wrt state.params
#                 inner_grad_fn = jax.grad(my_loss, has_aux=False)

#                 # Optimize inner loop for m steps
#                 for inner_step in range(self.num_steps):
#                     inner_grad = inner_grad_fn(inner_params)
#                     inner_updates, inner_optimizer_state = self.inner_optimizer.update(
#                         inner_grad, inner_optimizer_state
#                     )
#                     inner_params = optax.apply_updates(inner_params, inner_updates)

#                 # Calculate loss with resulting params
#                 return my_loss(inner_params)

#             # Create outer grad function, backpropagating through the inner loop.
#             outer_grad_fn = jax.grad(inner_train_loop_loss, has_aux=False)

#             # Obtain gradients for outer params
#             grads = outer_grad_fn(state.params)

#             # Update state parameters with unrolled gradients
#             state = state.apply_gradients(
#                 # grads=jax.tree_map(lambda x: jnp.mean(x, axis=0, keepdims=True), grads)
#                 grads=grads
#             )

#             return state

#         self.outer_train_step = jax.jit(outer_train_step)
#         # self.outer_train_step = outer_train_step

#     def compile(self, exmp_coords, exmp_images):
#         """Compiles outer and inner training steps."""
#         _ = self.outer_train_step(self.state, exmp_coords, exmp_images)

#     def create_functions(self):
#         if not hasattr(self, "loss_fn"):
#             self.create_loss()

#         if not hasattr(self, "outer_train_step"):
#             self.create_outer_train_step()

#         if not hasattr(self, "inner_train_step"):
#             self.create_inner_train_step()

#     def recon(self, params, coords):
#         # vmap model application over batch dimension.
#         recon = jax.vmap(lambda params: self.model.apply({"params": params}, coords))(params)
#         return recon

#     def psnr(self, params, coords, images):
#         """Obtain the PSNR for a given set of parameters and images.

#         Args:
#             params (jax.tree_util.PartiallyMutableMapping): The parameters to use.
#             coords (jnp.ndarray): The coordinates to use.
#             images (jnp.ndarray): The images to use.
#         """
#         # Calculate PSNR
#         metric = psnr(self.recon(params, coords), images)
#         return jnp.mean(metric)
