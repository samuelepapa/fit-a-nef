from typing import Any, Sequence

import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from jax import core, dtypes, random
from jax.nn.initializers import Initializer
from jax.random import KeyArray

#############################################
#       Base MFN
##############################################


def FourierNet_key(param_name, nef_cfg):
    # bias before kernel, ordered based on layer number starting from 0
    if param_name.startswith("output_linear."):
        index = 4 * nef_cfg.get("num_filters") - 2
    else:
        index = 2 * int(param_name.split(".")[0].split("_")[-1])

    if param_name.startswith("linears_"):
        index = index + 2 * nef_cfg.get("num_filters")

    if param_name.endswith(".bias"):
        return index
    elif param_name.endswith(".kernel"):
        return index + 1


def GaborNet_key(param_name, num_layers):
    raise NotImplementedError(
        "GaborNet_key not implemented yet, cannot sort the parameters correctly"
    )


def gamma_initialization(
    alpha: float = 6, beta: float = 1, dtype: Any = jnp.float_
) -> Initializer:
    def init(key: KeyArray, shape: core.Shape, dtype: Any = dtype) -> Any:
        dtype = dtypes.canonicalize_dtype(dtype)

        return random.gamma(key, alpha, shape, dtype) / beta

    return init


def simple_uniform(maxval=1, minval=None, dtype: Any = jnp.float_) -> Initializer:
    def init(key: KeyArray, shape: core.Shape, dtype: Any = dtype) -> Any:
        dtype = dtypes.canonicalize_dtype(dtype)

        if minval is None:
            return random.uniform(key, shape, dtype, minval=-maxval, maxval=maxval)
        else:
            return random.uniform(key, shape, dtype, minval=minval, maxval=maxval)

    return init


class MFNBase(nn.Module):
    """Multiplicative filter network base class.

    Expects the child class to define the 'filters' attribute, which should be a nn.Module of
    num_filters+1 filters with output equal to hidden_dim.
    """

    output_dim: int
    hidden_dim: int
    num_filters: int
    weight_scale: float = 6.0

    def setup(self):
        assert self.num_filters >= 1
        self.linears = [
            nn.Dense(
                features=self.hidden_dim,
                use_bias=True,
                kernel_init=simple_uniform(maxval=jnp.sqrt(self.weight_scale / (self.hidden_dim))),
                bias_init=nn.initializers.zeros,
            )
            for _ in range(self.num_filters - 1)
        ]
        self.output_linear = nn.Dense(
            features=self.output_dim, use_bias=True, kernel_init=nn.initializers.he_uniform()
        )

    def __call__(self, x):
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.linears[i - 1](out)
        out = self.output_linear(out)
        return out


#############################################
#       FourierNet
##############################################
def fourier_filter_linear_init(weight_scale: float, dtype: Any = jnp.float_) -> Initializer:
    def init(key: KeyArray, shape: core.Shape, dtype: Any = dtype) -> Any:
        dtype = dtypes.canonicalize_dtype(dtype)

        return jnp.ones(shape, dtype) * weight_scale

    return init


class FourierFilter(nn.Module):
    hidden_dim: int
    num_filters: int
    input_scale: float

    def setup(self):
        self.linear = nn.Dense(
            features=self.hidden_dim,
            use_bias=True,
            kernel_init=nn.initializers.variance_scaling(
                self.input_scale / np.sqrt(self.num_filters),
                mode="fan_in",
                distribution="uniform",
            ),
            bias_init=simple_uniform(jnp.pi),
        )

    def __call__(self, x):
        return jnp.sin(self.linear(x))


class FourierNet(MFNBase):
    input_scale: float = 256.0

    def setup(self):
        super().setup()
        self.filters = [
            FourierFilter(
                hidden_dim=self.hidden_dim,
                input_scale=self.input_scale,
                num_filters=self.num_filters,
            )
            for _ in range(self.num_filters)
        ]


#############################################
#       GaborNet
##############################################


def gaussian_window(x, gamma, mu):
    D = (x**2).sum(1, keepdims=True) + (mu**2).sum(0, keepdims=True) - 2 * jnp.matmul(x, mu)
    return jnp.exp(-0.5 * D * gamma[None, ...])


class GaborFilter(nn.Module):
    hidden_dim: int
    num_filters: int
    input_scale: float
    alpha: float
    beta: float

    def setup(self):
        weight_scale = self.input_scale / jnp.sqrt(self.num_filters)

        alpha = self.alpha / (self.num_filters + 1)
        beta = self.beta
        self.mu = nn.Dense(
            features=self.hidden_dim,
            use_bias=False,
        )
        self.gamma = self.param(
            "gamma", gamma_initialization(alpha=alpha, beta=beta), (self.hidden_dim,)
        )

        self.linear = nn.Dense(
            features=self.hidden_dim,
            use_bias=True,
            kernel_init=nn.initializers.variance_scaling(
                jnp.sqrt(self.gamma[None, :]) * weight_scale, mode="fan_in", distribution="uniform"
            ),
            bias_init=simple_uniform(jnp.pi),
        )

    def __call__(self, x):
        gauss_window = gaussian_window(
            x,
            self.gamma,
            self.mu(jnp.identity(x.shape[-1])),
        )
        out = gauss_window * jnp.sin(self.linear(x))

        return out


class GaborNet(MFNBase):
    input_scale: float = 256.0
    alpha: float = 1 / 6
    beta: float = 1.0

    def setup(self):
        super().setup()
        self.filters = [
            GaborFilter(
                hidden_dim=self.hidden_dim,
                num_filters=self.num_filters,
                input_scale=self.input_scale,
                alpha=self.alpha,
                beta=self.beta,
            )
            for _ in range(self.num_filters)
        ]
