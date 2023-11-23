import torch
from jax import numpy as jnp


def make_3d_grid(min_val, max_val, resolution):
    """Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    """
    # Create 3D meshgrid of shape (resolution, resolution, resolution, 3)
    x_ = jnp.linspace(min_val, max_val, resolution)
    y_ = x_.copy()
    z_ = x_.copy()

    # Create meshgrid
    x, y, z = jnp.meshgrid(x_, y_, z_, indexing="ij")

    # Reshape to (resolution**3, 3)
    x = x.reshape(-1)
    y = y.reshape(-1)
    z = z.reshape(-1)

    # Stack to (resolution**3, 3)
    p = jnp.stack([x, y, z], axis=1)
    return p
