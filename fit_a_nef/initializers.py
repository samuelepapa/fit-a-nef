from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from flax import linen as nn


class InitModel(ABC):
    """Abstract class for initializing the model."""

    @abstractmethod
    def __call__(
        self, model: nn.Module, example_input: jnp.ndarray, num_signals: int
    ) -> jnp.ndarray:
        """Computes the parameters to initialize the model. This method is not supposed to be
        jitted.

        Args:
            model (nn.Module): The model to initialize.
            example_input (jnp.ndarray): An example input to the model.
            num_signals (int): The number of signals being fit.
        """
        pass


class SharedInit(InitModel):
    def __init__(self, init_rng: jnp.ndarray):
        """Initialize the initializer object.

        Args:
            init_rng (jnp.ndarray): The initial random number generator.
        """
        self.init_rng = init_rng

    def __call__(
        self, model: nn.Module, example_input: jnp.ndarray, num_signals: int
    ) -> jnp.ndarray:
        vmap_init = jax.vmap(model.init, in_axes=(0, None))
        return vmap_init(
            jnp.broadcast_to(self.init_rng, (num_signals, *self.init_rng.shape)), example_input
        )["params"]


class RandomInit(InitModel):
    def __init__(self, init_rng: jnp.ndarray):
        """Initialize the initializer object.

        Args:
            init_rng (jnp.ndarray): The initial random number generator.
        """
        self.init_rng = init_rng

    def __call__(
        self, model: nn.Module, example_input: jnp.ndarray, num_signals: int
    ) -> jnp.ndarray:
        self.init_rng, init_rng = jax.random.split(self.init_rng)
        vmap_init = jax.vmap(model.init, in_axes=(0, None))
        return vmap_init(jax.random.split(init_rng, num_signals), example_input)["params"]


class MetaLearnedInit(InitModel):
    def __init__(self, meta_learned_init: jnp.ndarray):
        """Initialize the initializer object.

        Args:
            meta_learned_init (jnp.ndarray): The meta-learned initialization.
        """
        self.meta_learned_init = meta_learned_init

    def __call__(
        self, model: nn.Module, example_input: jnp.ndarray, num_signals: int
    ) -> jnp.ndarray:
        return jax.tree_map(
            lambda p: jnp.broadcast_to(p, (num_signals, *p.shape[1:])), self.meta_learned_init
        )
