import jax.numpy as jnp
from flax import linen as nn
from jax import lax


def RFFNet_key(param_name, nef_cfg):
    # bias before kernel, ordered based on layer number
    if param_name.startswith("linear_final."):
        index = 2 * (nef_cfg.get("num_layers") - 1)
    else:
        index = 2 * int(param_name.split(".")[0].split("_")[-1])

    if param_name.endswith(".bias"):
        return index
    elif param_name.endswith(".kernel"):
        return index + 1
    else:
        raise ValueError(f"param_name (`{param_name}`) must end with either `.bias` or `.kernel`.")


class RFFNet(nn.Module):
    output_dim: int
    hidden_dim: int
    num_layers: int
    learnable_coefficients: bool
    std: float
    numerator: float = 2.0

    def setup(self):
        assert (
            self.num_layers >= 2
        ), "At least two layers (the hidden plus the output one) are required."

        # Encoding
        self.encoding = RFF(
            hidden_dim=self.hidden_dim,
            learnable_coefficients=self.learnable_coefficients,
            std=self.std,
        )

        # Hidden layers
        self.layers = [
            Layer(hidden_dim=self.hidden_dim, numerator=self.numerator)
            for _ in range(self.num_layers - 1)
        ]

        # Output layer
        self.linear_final = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=nn.initializers.variance_scaling(self.numerator, "fan_in", "uniform"),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )

    def __call__(self, x):
        x = self.encoding(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.linear_final(x)
        return x


class Layer(nn.Module):
    hidden_dim: int
    numerator: float = 2.0

    def setup(self):
        self.linear = nn.Dense(
            features=self.hidden_dim,
            use_bias=True,
            kernel_init=nn.initializers.variance_scaling(self.numerator, "fan_in", "normal"),
            bias_init=nn.initializers.normal(stddev=1e-6),
        )
        self.activation = nn.relu

    def __call__(self, x):
        return self.activation(self.linear(x))


class RFF(nn.Module):
    hidden_dim: int
    learnable_coefficients: bool
    std: float

    def setup(self):
        # Make sure we have an even number of hidden features.
        assert (
            not self.hidden_dim % 2.0
        ), "For the Fourier Features hidden_dim should be even to calculate them correctly."

        # Store pi
        self.pi = 2 * jnp.pi

        # Embedding layer
        self.coefficients = nn.Dense(
            self.hidden_dim // 2,
            use_bias=False,
            kernel_init=nn.initializers.normal(stddev=1),
        )
        self.concat = lambda x: jnp.concatenate([jnp.sin(x), jnp.cos(x)], axis=-1)

        if self.learnable_coefficients:
            self.parsed_coefficients = lambda x: self.coefficients(self.pi * x)
        else:
            self.parsed_coefficients = lambda x: lax.stop_gradient(self.coefficients(self.pi * x))

    def __call__(self, x):
        x_proj = self.std * self.parsed_coefficients(x)
        return self.concat(x_proj)
