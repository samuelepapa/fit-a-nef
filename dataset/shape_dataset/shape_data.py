from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from torch.utils import data


def numpy_collate(batch: Union[np.ndarray, Sequence[Any], Any]):
    """
    TODO: this might be a repeat, maybe it's ok to make it special for shapes, but needs a check
    Collate function for numpy arrays.

    This function acts as replacement to the standard PyTorch-tensor collate function in PyTorch DataLoader.

    Args:
        batch: Batch of data. Can be a numpy array, a list of numpy arrays, or nested lists of numpy arrays.

    Returns:
        Batch of data as (potential list or tuple of) numpy array(s).
    """
    if isinstance(batch, np.ndarray):
        return batch
    elif isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


def load_attributes(loader: data.DataLoader, attribute_name: str = "labels") -> Dict[str, Any]:
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
    if next(iter(loader))[0][1] is None:
        return {}
    else:
        return {attribute_name: [x[0][1] for x in iter(loader)]}


def load_shapes(
    source_dataset: data.Dataset,
    start_idx: int,
    end_idx: int,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
) -> Tuple[jnp.array, jnp.array, Tuple[int, int, int]]:
    """
    TODO: missing docs
    """
    # Create a subset of the dataset
    dset = data.Subset(source_dataset, range(start_idx, end_idx))

    # Create a loader with a single worker to get the data from disk to RAM
    loader = data.DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        persistent_workers=False,
    )

    return loader
