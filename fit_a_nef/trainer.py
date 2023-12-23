# Documentation
import json
import pickle
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging

# JAX
from flax.training import train_state

from .initializers import InitModel, MetaLearnedInit, RandomInit, SharedInit
from .nef import param_key_dict

# Misc
from .utils import (
    TrainState,
    flatten_params,
    get_nef,
    get_optimizer,
    get_scheduler,
    unflatten_params,
)


class SignalTrainer:
    """Base class for training neural networks.

    :param coords: the coordinates to train on
    :type coords: jnp.ndarray
    :param signals: the signal values to train on, i.e. images (pixel values) objects (occupancy)
    :type signals: jnp.ndarray
    :param nef_cfg: the config for the neural network
    :type nef_cfg: Dict[str, Any]
    :param scheduler_cfg: the config for the scheduler
    :type scheduler_cfg: Dict[str, Any]
    :param optimizer_cfg: the config for the optimizer
    :type optimizer_cfg: Dict[str, Any]
    :param initializer: the initializer for the model, see initializers.py for more info
    :type initializer: InitModel
    :param train_rng: the random number generator to use for training
    :type train_rng: jnp.ndarray
    :param num_signals: the number of signals being fit
    :type num_signals: int
    :param num_steps: the number of steps to train for, defaults to 20000
    :type num_steps: int, optional
    :param verbose: whether to have verbose training or not, defaults to False. Overwrite the :func:`verbose_train_model` function to change the logging behavior.
    :type verbose: bool, optional
    """

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
        """Constructor."""
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
        """Creates the functions needed for training the model. This includes the loss function,
        the train step and the train model functions.

        This is needed to allow for proper handling of the Jax JIT compilation of the train_step
        function.

        For the train_model function, this allows to switch between verbose and fast training
        without any computational overhead during the actual fitting.
        """
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

    def process_batch(
        self, state: TrainState, coords: jnp.ndarray, signals: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Used to process the batch before passing it to the loss function. This is useful for
        selecting specific coordinates or changing the shapes of the signals.

        :param state: the current state of the training, used for functional programming.
        :type state: TrainState
        :param coords: the coordinates to process.
        :type coords: jnp.ndarray
        :param signals: the signals to process.
        :type signals: jnp.ndarray
        :return: the processed coordinates, signals and the random number generator.
        :rtype: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
        """
        rng, cur_rng = jax.random.split(state.rng)
        return coords, signals, rng

    def init_model(
        self,
        example_input: jnp.ndarray,
    ) -> None:
        """Initializes the model parameters using the initializer defined in the constructor.

        :param example_input: An example input to the model. Used by Jax to initialize the model
            correctly.
        :type example_input: jnp.ndarray
        :return: None
        :rtype: None
        """
        # Initialize model parameters
        params = self.initializer(self.model, example_input, self.num_signals)

        # Initialize optimizer and learning rate schedule
        self.lr_schedule = get_scheduler(self.scheduler_cfg)
        optimizer = get_optimizer(self.optimizer_cfg, self.lr_schedule)

        # Create the train state
        self.state = TrainState.create(
            apply_fn=self.model.apply, params=params, tx=optimizer, rng=self.train_rng
        )

    def compile(self) -> None:
        """Executes the training function ones to compile the train_step.

        :return: None
        :rtype: None
        """

        _ = self.train_step(self.state, self.coords, self.signals)

    def create_train_model(self) -> None:
        """Creates the train_model function. This is used to switch between verbose and fast
        training without any computational overhead during the actual fitting.

        :return: None
        :rtype: None
        """
        if self.verbose:
            self.train_model = self.verbose_train_model
        else:
            self.train_model = self.fast_train_model

    def fast_train_model(self) -> None:
        """Quickly trains the model for the number of steps specified in the init function."""
        for _ in range(1, self.num_steps + 1):
            self.state, _ = self.train_step(self.state, self.coords, self.signals)

    def verbose_train_model(self) -> None:
        """Trains the model for the number of steps specified in the init function and logs the
        loss every 100 steps."""
        for step in range(1, self.num_steps + 1):
            self.state, loss = self.train_step(self.state, self.coords, self.signals)
            if step % 100 == 0:
                logging.info(f"Step {step}, loss {loss}")

    def get_params(self, model_id: Optional[int] = None) -> jnp.ndarray:
        """Returns the params for a given model ID or all params if no model ID is specified.

        :param model_id: The model ID to get the params for or None to get all params.
        :type model_id: int, optional
        :return: The params for the given model ID or all params if no model ID is specified.
        :rtype: jnp.ndarray
        """

        if model_id is None:
            return self.state.params
        else:
            return jax.tree_map(lambda x: x[model_id], self.state.params)

    def get_flat_params(
        self, model_id: Optional[int] = None
    ) -> Tuple[jnp.ndarray, List[Tuple[str, List[int]]]]:
        """Returns the *flattened* params for a given model ID or all params if no model ID is
        specified.

        :param model_id: The model ID to get the params for or None to get all params.
        :type model_id: int, optional
        :return: A tuple with the *flattened* params for the given model ID or all params if no
            model ID is specified, and the param configuration.
        :rtype: Tuple[jnp.ndarray, List[Tuple[str, List[int]]]]
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

    def apply_model(self, coords: jnp.ndarray, model_id: Optional[int] = None) -> jnp.ndarray:
        """Applies the model to a given set of coordinates.

        :param coords: The coordinates to apply the model to.
        :type coords: jnp.ndarray
        :param model_id: The model ID to apply. Defaults to None in which case all models are used.
        :type model_id: int, optional
        :return: The output of the model.
        :rtype: jnp.ndarray
        """

        if model_id is None:
            return jax.vmap(lambda params: self.model.apply({"params": params}, coords))(
                self.state.params
            )
        else:
            return self.model.apply({"params": self.get_params(model_id)}, coords)

    def save(self, path: Path, **kwargs) -> None:
        """Save the parameters to a hdf5 file.

        :param path: The path to save the parameters to.
        :type path: Path
        :param kwargs: Additional data to save.
        :type kwargs: Dict[str, Any]
        :return: None
        :rtype: None
        """

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

    def load(self, path: Path) -> None:
        """Used to load the parameters from a hdf5 file that can be created using the `save`
        function.

        :param path: The path to load the parameters from.
        :type path: Path
        :return: None
        :rtype: None
        """
        with h5py.File(path, "r") as f:
            param_config = json.loads(f["param_config"][0].decode("utf-8"))
            comb_params = f["params"][:]
        params = unflatten_params(param_config, comb_params)
        self.state = self.state.replace(params=params)

    def clean_up(self, clear_caches=True):
        """Cleans up the trainer by deleting the state and train_step attributes. This is useful to
        free up memory.

        :param clear_caches: Whether to clear the Jax caches or not. Defaults to True.
        :type clear_caches: bool, optional
        :return: None
        :rtype: None
        """
        del self.state
        if hasattr(self, "train_step"):
            del self.train_step
        if clear_caches:
            jax.clear_caches()

    def get_lr(self):
        """Returns the current learning rate.

        :return: The current learning rate.
        :rtype: float
        """
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
