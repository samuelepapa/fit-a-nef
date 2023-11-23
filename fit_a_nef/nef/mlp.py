# Flax
from flax import linen as nn


def MLP_key(param_name, nef_cfg):
    # bias before kernel, ordered based on layer number
    index = 2 * int(param_name.split(".")[0].split("_")[-1])

    if param_name.endswith(".bias"):
        return index
    elif param_name.endswith(".kernel"):
        return index + 1
    else:
        raise ValueError(f"param_name (`{param_name}`) must end with either `.bias` or `.kernel`.")


class MLP(nn.Module):
    hidden_dim: int
    output_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = nn.Dense(
                features=self.hidden_dim,
                use_bias=True,
                kernel_init=nn.initializers.glorot_normal(),
            )(x)
            x = nn.relu(x)
        x = nn.Dense(
            features=self.output_dim,
            use_bias=True,
            kernel_init=nn.initializers.glorot_normal(),
        )(x)
        return x
