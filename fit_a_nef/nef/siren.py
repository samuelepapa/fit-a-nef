import jax.numpy as jnp
from flax import linen as nn

from ..nef.utils import custom_uniform


def SIREN_key(param_name, nef_cfg):
    # bias before kernel, ordered based on layer number
    if param_name.startswith("output_linear."):
        index = 2 * nef_cfg.get("num_layers") - 2
    else:
        index = 2 * int(param_name.split(".")[0].split("_")[-1])

    if param_name.endswith(".bias"):
        return index
    elif param_name.endswith(".kernel"):
        return index + 1
    else:
        raise ValueError(f"param_name (`{param_name}`) must end with either `.bias` or `.kernel`.")


class SIREN(nn.Module):
    output_dim: int
    hidden_dim: int
    num_layers: int
    omega_0: float

    def setup(self):
        self.kernel_net = [
            SirenLayer(
                output_dim=self.hidden_dim,
                omega_0=self.omega_0,
                is_first_layer=True,
            )
        ] + [
            SirenLayer(
                output_dim=self.hidden_dim,
                omega_0=self.omega_0,
            )
            for _ in range(self.num_layers - 2)
        ]

        self.output_linear = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=custom_uniform(numerator=1, mode="fan_in", distribution="normal"),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        for layer in self.kernel_net:
            x = layer(x)

        out = self.output_linear(x)

        return out


class SirenLayer(nn.Module):
    output_dim: int
    omega_0: float
    is_first_layer: bool = False

    def setup(self):
        c = 1 if self.is_first_layer else 6 / self.omega_0**2
        distrib = "uniform_squared" if self.is_first_layer else "uniform"
        self.linear = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=custom_uniform(numerator=c, mode="fan_in", distribution=distrib),
            bias_init=nn.initializers.zeros,
        )

    def __call__(self, x):
        after_linear = self.omega_0 * self.linear(x)
        return jnp.sin(after_linear)
