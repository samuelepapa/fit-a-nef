import math
from pathlib import Path
from typing import Dict, List, Optional

import jax.numpy as jnp
import numpy as np


def find_seed_idx(idx: int, signals_in_dset: int) -> int:
    return math.floor(idx / signals_in_dset)


def get_signal_idx(idx: int, signals_in_dset: int) -> int:
    return idx % signals_in_dset


def get_num_nefs_list(
    nef_start_idx: int, nef_end_idx: int, num_parallel_nefs: int, signals_in_dset: int
) -> List[int]:
    """Split the nefs into chunks of size num_parallel_nefs. Keeps into account the seeding of the
    nefs. Assumes that the nefs are ordered by seed and that nef_end_idx is not too large (i.e. it
    is not larger than signals_in_dset*number_of_seeds).

    Args:
        nef_start_idx (int): The starting index of the nefs to train.
        nef_end_idx (int): The ending index of the nefs to train.
        num_parallel_nefs (int): The number of nefs to train in parallel.
        signals_in_dset (int): The number of signals in the dataset.

    Returns:
        List[int]: The number of nefs to train in each chunk.
    """

    assert num_parallel_nefs <= signals_in_dset, (
        f"Number of parallel nef ({num_parallel_nefs}) "
        f"cannot be larger than the dataset size ({signals_in_dset})."
    )
    # Split the nefs into chunks of size num_parallel_nefs
    num_nefs_list = []
    start_idx = nef_start_idx
    # keep adding chunks until we reach the desired end
    while start_idx < nef_end_idx:
        # By default this is the number of nefs to train
        nefs_to_train = num_parallel_nefs

        # find the seeds that the first nef and last nef in this parallel
        # chunk belong to
        start_seed_idx = find_seed_idx(start_idx, signals_in_dset)
        end_seed_idx = find_seed_idx(start_idx + num_parallel_nefs - 1, signals_in_dset)

        # if they are not the same, we need to reduce the number of nefs to
        # train in order to have each chunk share the same seed
        if start_seed_idx != end_seed_idx:
            nefs_to_train = signals_in_dset - get_signal_idx(start_idx, signals_in_dset)

        # the last chunk might be smaller than the parallel chunk size
        if start_idx + nefs_to_train > nef_end_idx:
            nefs_to_train = nef_end_idx - start_idx

        num_nefs_list.append(nefs_to_train)
        start_idx += nefs_to_train

    total_nefs = nef_end_idx - nef_start_idx
    assert sum(num_nefs_list) == total_nefs, (
        f"Something went wrong when splitting the nefs into chunks of size "
        f"{num_parallel_nefs}."
    )

    return num_nefs_list
