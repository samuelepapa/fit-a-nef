from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import torch.utils.data as data


def load_attributes(
    source_dataset: data.Dataset, start_idx: int, end_idx: int, attribute_name: str = "labels"
) -> Dict[str, Any]:
    """Get the attributes of the images in the dataset. This is the second element of the tuple
    that the dataset returns. If the dataset does not have attributes, an empty dictionary is
    returned. The attribute name is assumed to be "labels" by default.

    Args:
        source_dataset (data.Dataset): The dataset.
        start_idx (int): The index of the first attribute to load.
        end_idx (int): The index of the last attribute loaded will be end_idx - 1.
        attribute_name (str): The name of the attribute to load. Defaults to "labels".

    Returns:
        A dictionary with the attributes of the images.
    """
    # Create a subset of the dataset
    dset = data.Subset(source_dataset, range(start_idx, end_idx))

    # Create a loader with a single worker to get the data from disk to RAM
    loader = data.DataLoader(
        dset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x
    )

    if next(iter(loader))[0][1] is None:
        return {}
    else:
        return {attribute_name: [x[0][1] for x in iter(loader)]}


def load_images(
    source_dataset: data.Dataset,
    start_idx: int,
    end_idx: int,
    rng: Optional[jax.random.PRNGKey] = None,
    force_shuffle: bool = False,
) -> Tuple[jnp.array, jnp.array, Tuple[int, int, int], Optional[jax.random.PRNGKey]]:
    """Load images from the dataset and create the coordinates. The returned images will have shape
    (num_images, num_pixels, num_channels) while the coordinates will have shape (num_pixels,
    num_channels).

    When shuffling is required, a Jax key is used to get the randomness.
    A new key is returned which should be further split before the next use.

    Args:
        source_dataset (data.Dataset): The dataset.
        start_idx (int): The index of the first image to load.
        end_idx (int): The last image loaded will be the one with index end_idx - 1.
        rng (jax.random.PRNGKey): a random key used in the pseudo-random number generation. Default: None.
        force_shuffle (bool): whether to shuffle the dataset. Default: False.

    Returns:
        coords: The coordinates of the pixels with shape
                (num_pixels, num_channels).
        images: The images with shape
                (num_images, num_pixels, num_channels).
        image_shape: The original shape of the images.
        new_rng: A new PRNG key if one was provided and
                shuffling was performed, otherwise returns the
                provided key in `rng`.
    """
    # Create a subset of the dataset
    dset = data.Subset(source_dataset, range(start_idx, end_idx))

    # Create a loader with a single worker to get the data from disk to RAM
    loader = data.DataLoader(
        dset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x
    )
    # make them into a jnp array with the correct shape, assumes the data is in [0, 1]
    images = jnp.stack([x[0][0] for x in iter(loader)], axis=0)
    # remember the original shape of the images for plotting and coordinate making
    images_shape = list(images.shape[-3:])
    # reshape to (num_images, num_pixels, num_channels)
    images = images.reshape(-1, images_shape[0] * images_shape[1], images_shape[2])

    if force_shuffle:
        if rng is None:
            raise RuntimeError(
                "force_shuffle is set to True, but the rng key has not been provided."
            )
        else:
            rng, new_rng = jax.random.split(rng)
            index_perm = jax.random.permutation(new_rng, jnp.arange(0, images.shape[0]), axis=0)
            images = images[index_perm]
    else:
        index_perm = jnp.arange(0, images.shape[0])

    # coordinates
    x = jnp.linspace(-1, 1, images_shape[0])
    y = jnp.linspace(-1, 1, images_shape[1])
    x, y = jnp.meshgrid(x, y)
    coords = jnp.stack([x, y], axis=-1)
    # reshape to (1, num_pixels, num_channels)
    coords = coords.reshape(images_shape[0] * images_shape[1], 2)

    return coords, images, images_shape, rng, index_perm
