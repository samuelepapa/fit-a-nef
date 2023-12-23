import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import flax
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state

from fit_a_nef import nef


class TrainState(train_state.TrainState):
    # Adding rng key for masking
    rng: Any = None


def get_scheduler(scheduler_cfg: Dict[str, Any]) -> optax.Schedule:
    """Returns the scheduler for the given config. All schedulers from optax are supported.

    :param scheduler_cfg: The config for the scheduler.
    :type scheduler_cfg: ConfigDict
    :raises NotImplementedError: If the scheduler is not implemented.
    :return: The scheduler.
    :rtype: optax.Schedule
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

    :param optimizer_cfg: Configuration for the optimizer.
    :type optimizer_cfg: Dict[str, Any]
    :param scheduler: Learning rate schedule.
    :type scheduler: optax.Schedule
    :raises NotImplementedError: If the specified optimizer is not implemented in optax.
    :return: Optimizer instance.
    :rtype: optax.GradientTransformation
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

    :param nef_cfg: The config for the model.
    :type nef_cfg: ConfigDict
    :raises NotImplementedError: If the model is not implemented.
    :return: The model.
    :rtype: flax.linen.Module
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

    :param d: The dictionary to flatten.
    :type d: Dict
    :param separation: The separation character, defaults to ".".
    :type separation: str, optional
    :return: The flattened dictionary.
    :rtype: Dict
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

    :param d: The dictionary to unflatten.
    :type d: Dict
    :param separation: The separation character, defaults to ".".
    :type separation: str, optional
    :return: The unflattened dictionary.
    :rtype: Dict
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


def flatten_params(
    params: Any, num_batch_dims: int = 0, param_key: callable = None
) -> Tuple[List[Tuple[str, List[int]]], jnp.ndarray]:
    """Flattens the parameters of the model.

    :param params: The parameters of the model.
    :type params: jax.PyTree
    :param num_batch_dims: The number of batch dimensions. Tensors will not be flattened over these
        dimensions, defaults to 0.
    :type num_batch_dims: int, optional
    :param param_key: The key to sort the parameters, defaults to None.
    :type param_key: callable, optional
    :return: Structure of the flattened parameters, the flattened parameters.
    :rtype: List[Tuple[str, List[int]]], jnp.ndarray
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

    :param param_config: Structure of the flattened parameters.
    :type param_config: List[Tuple[str, List[int]]]
    :param comb_params: The flattened parameters.
    :type comb_params: jnp.ndarray
    :return: The parameters of the model.
    :rtype: jax.PyTree
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


def get_meta_init(storage_folder: Path, epoch_index: int) -> Union[jnp.ndarray, None]:
    """Load meta-learned initialization for current configuration. TODO this is not implemented
    properly yet.

    :param storage_folder: The folder to load the meta-learned initialization from.
    :type storage_folder: Path
    :param epoch_index: The epoch index to load.
    :type epoch_index: int
    :return: The meta-learned initialization.
    :rtype: Union[jnp.ndarray, None]
    """
    # Load meta-learned initialization
    meta_init_path = storage_folder / Path(f"meta_init_epoch_{epoch_index}.h5py")
    if meta_init_path.exists():
        with h5py.File(meta_init_path, "r") as f:
            param_config = json.loads(f["param_config"][0].decode("utf-8"))
            comb_params = f["params"][:]
            return unflatten_params(param_config, comb_params)

    logging.warn(f"Meta init not found at {meta_init_path}.")
    logging.warn(
        "Please run `python tasks/image/find_init.py --task=config/image.py:find_init ...` to create it."
    )
    return None
