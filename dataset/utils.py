from pathlib import Path
from typing import Any, Callable, Sequence, Tuple, Union

from absl import logging


def start_end_idx_from_path(path: str) -> Tuple[int, int]:
    """Get start and end index from path.

    Args:
        path: Path from which to extract the start and end index.

    Returns:
        Tuple with start and end index.
    """
    start_idx = int(Path(path).stem.split("_")[1].split("-")[0])
    end_idx = int(Path(path).stem.split("_")[1].split("-")[1])
    return start_idx, end_idx


def path_from_name_idxs(name: str, start_idx: int, end_idx: int) -> str:
    return f"{name}_{start_idx}-{end_idx}.hdf5"
