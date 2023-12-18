import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import flax
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state

from fit_a_nef import nef


class TrainState(train_state.TrainState):
    # Adding rng key for masking
    rng: Any = None


def get_scheduler(scheduler_cfg: Dict[str, Any]) -> optax.Schedule:
    """Returns the scheduler for the given config. All schedulers from optax are supported.

    Args:
        scheduler_cfg (ConfigDict): The config for the scheduler.

    Raises:
        NotImplementedError: If the scheduler is not implemented.

    Returns:
        optax.Schedule: The scheduler.
    """
    # raise exception if scheduler is not implemented
    if scheduler_cfg["name"] not in dir(optax):
        raise NotImplementedError(
            f"Scheduler {scheduler_cfg['name']} not implemented in optax. Available are: {dir(optax)}"
        )
    else:
        # use getattr to get the scheduler
        scheduler = getattr(optax, scheduler_cfg["name"])

        return scheduler(**scheduler_cfg["params"])


def get_optimizer(
    optimizer_cfg: Dict[str, Any], scheduler: optax.Schedule
) -> optax.GradientTransformation:
    """Get the optimizer based on the provided configuration and scheduler.

    Args:
        optimizer_cfg (Dict[str, Any]): Configuration for the optimizer.
        scheduler (optax.Schedule): Learning rate schedule.

    Returns:
        optax.GradientTransformation: Optimizer instance.

    Raises:
        NotImplementedError: If the specified optimizer is not implemented in optax.
    """
    # raise exception if scheduler is not implemented
    if optimizer_cfg["name"] not in dir(optax):
        raise NotImplementedError(
            f"Optimizer {optimizer_cfg['name']} not implemented in optax. Available are: {dir(optax)}"
        )
    else:
        # use getattr to get the scheduler
        optimizer = getattr(optax, optimizer_cfg["name"])

        return optimizer(scheduler, **optimizer_cfg["params"])


def get_nef(nef_cfg: Dict[str, Any]) -> flax.linen.Module:
    """Returns the model for the given config.

    Args:
        nef_cfg (ConfigDict): The config for the model.

    Raises:
        NotImplementedError: If the model is not implemented.

    Returns:
        flax.linen.Module: The model.
    """
    if nef_cfg["name"] not in dir(nef):
        raise NotImplementedError(
            f"Model {nef_cfg['name']} not implemented. Available are: {dir(nef)}"
        )
    else:
        model = getattr(nef, nef_cfg["name"])
        return model(**nef_cfg["params"])


def flatten_dict(d: Dict, separation: str = "."):
    """Flattens a dictionary.

    Args:
        d (Dict): The dictionary to flatten.

    Returns:
        Dict: The flattened dictionary.
    """
    flat_d = {}
    for key, value in d.items():
        if isinstance(value, (dict, FrozenDict)):
            sub_dict = flatten_dict(value)
            for sub_key, sub_value in sub_dict.items():
                flat_d[key + separation + sub_key] = sub_value
        else:
            flat_d[key] = value
    return flat_d


def unflatten_dict(d: Dict, separation: str = "."):
    """Unflattens a dictionary, inverse to flatten_dict.

    Args:
        d (Dict): The dictionary to unflatten.
        separation (str, optional): The separation character. Defaults to ".".

    Returns:
        Dict: The unflattened dictionary.
    """
    unflat_d = {}
    for key, value in d.items():
        if separation in key:
            sub_keys = key.split(separation)
            sub_dict = unflat_d
            for sub_key in sub_keys[:-1]:
                if sub_key not in sub_dict:
                    sub_dict[sub_key] = {}
                sub_dict = sub_dict[sub_key]
            sub_dict[sub_keys[-1]] = value
        else:
            unflat_d[key] = value
    return unflat_d


def flatten_params(params: Any, num_batch_dims: int = 0, param_key: callable = None):
    """Flattens the parameters of the model.

    Args:
        params (jax.PyTree): The parameters of the model.
        num_batch_dims (int, optional): The number of batch dimensions. Tensors will not be flattened over these dimensions. Defaults to 0.

    Returns:
        List[Tuple[str, List[int]]]: Structure of the flattened parameters.
        jnp.ndarray: The flattened parameters.
    """
    flat_params = flatten_dict(params)
    keys = sorted(list(flat_params.keys()), key=param_key)
    param_config = [(k, flat_params[k].shape[num_batch_dims:]) for k in keys]
    comb_params = jnp.concatenate(
        [flat_params[k].reshape(*flat_params[k].shape[:num_batch_dims], -1) for k in keys], axis=-1
    )
    return param_config, comb_params


def unflatten_params(
    param_config: List[Tuple[str, List[int]]],
    comb_params: jnp.ndarray,
):
    """Unflattens the parameters of the model.

    Args:
        param_config (List[Tuple[str, List[int]]]): Structure of the flattened parameters.
        comb_params (jnp.ndarray): The flattened parameters.

    Returns:
        jax.PyTree: The parameters of the model.
    """
    params = []
    key_dict = {}
    idx = 0
    for key, shape in param_config:
        params.append(
            comb_params[..., idx : idx + np.prod(shape)].reshape(*comb_params.shape[:-1], *shape)
        )
        key_dict[key] = 0
        idx += np.prod(shape)
    # reorder the params based on lexicographical order of the key_dict.keys()
    # this is done because, internally, jax always sorts the keys of a dict lexygraphically
    sort_idx = np.argsort(list(key_dict.keys()))
    params = [params[i] for i in sort_idx]
    key_dict = unflatten_dict(key_dict)

    return FrozenDict(jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(key_dict), params))


# TODO properly implement this.
def get_meta_init(storage_folder, epoch_index: int):
    """Load meta-learned initialization for current configuration.

    Args:
        nef_cfg (Dict[str]): The config for the neural network.
        cfg (Dict[str]): The config for the experiment.
        epoch_index (int): The epoch index to load.
    """
    # Load meta-learned initialization
    meta_init_path = storage_folder / Path(f"meta_init_epoch_{epoch_index}.h5py")
    if meta_init_path.exists():
        with h5py.File(meta_init_path, "r") as f:
            param_config = json.loads(f["param_config"][0].decode("utf-8"))
            comb_params = f["params"][:]
            return unflatten_params(param_config, comb_params)

    print(f"Meta init not found at {meta_init_path}.")
    print(
        "Please run `python tasks/image/find_init.py --task=config/image.py:find_init ...` to create it."
    )
    return None
