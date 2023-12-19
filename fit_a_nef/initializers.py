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

        :param model: The model to initialize.
        :type model: nn.Module
        :param example_input: An example input to the model.
        :type example_input: jnp.ndarray
        :param num_signals: The number of signals being fit.
        :type num_signals: int
        :return: The parameters to initialize the model.
        """
        pass


class SharedInit(InitModel):
    """Initializes the model with the same parameters for each signal. The parameters are sampled
    using the same random number generator.

    :param init_rng: The initial random number generator.
    :type init_rng: jnp.ndarray
    """

    def __init__(self, init_rng: jnp.ndarray):
        """Constructor method."""
        self.init_rng = init_rng

    def __call__(
        self, model: nn.Module, example_input: jnp.ndarray, num_signals: int
    ) -> jnp.ndarray:
        vmap_init = jax.vmap(model.init, in_axes=(0, None))
        return vmap_init(
            jnp.broadcast_to(self.init_rng, (num_signals, *self.init_rng.shape)), example_input
        )["params"]


class RandomInit(InitModel):
    """Initializes the model with different parameters for each signal. The parameters are sampled
    using different random number generators. The rng is split for each signal starting from the
    init_rng.

    :param init_rng: The initial random number generator.
    :type init_rng: jnp.ndarray
    """

    def __init__(self, init_rng: jnp.ndarray):
        """Constructor method."""
        self.init_rng = init_rng

    def __call__(
        self, model: nn.Module, example_input: jnp.ndarray, num_signals: int
    ) -> jnp.ndarray:
        self.init_rng, init_rng = jax.random.split(self.init_rng)
        vmap_init = jax.vmap(model.init, in_axes=(0, None))
        return vmap_init(jax.random.split(init_rng, num_signals), example_input)["params"]


class MetaLearnedInit(InitModel):
    """Initializes all models using the parameters passed in the init.

    :param meta_learned_init: The meta-learned initialization.
    :type meta_learned_init: jnp.ndarray
    """

    def __init__(self, meta_learned_init: jnp.ndarray):
        """Constructor method."""
        self.meta_learned_init = meta_learned_init

    def __call__(
        self, model: nn.Module, example_input: jnp.ndarray, num_signals: int
    ) -> jnp.ndarray:
        return jax.tree_map(
            lambda p: jnp.broadcast_to(p, (num_signals, *p.shape[1:])), self.meta_learned_init
        )
