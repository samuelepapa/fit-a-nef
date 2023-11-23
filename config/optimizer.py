import pathlib

from absl import logging
from ml_collections import ConfigDict, config_dict

available_optimizers = {
    "adam": ConfigDict(
        {
            "name": "adam",
            "params": {
                "b1": 0.9,
                "b2": 0.999,
                "eps": 1e-8,
                "eps_root": 0.0,
            },
        }
    ),
    "sgd": ConfigDict(
        {
            "name": "sgd",
            "params": {
                "momentum": 0.9,
                "nesterov": False,
            },
        }
    ),
    "rmsprop": ConfigDict(
        {
            "name": "rmsprop",
            "params": {
                "decay": 0.9,
                "eps": 1e-08,
                "initial_scale": 0.0,
                "centered": False,
                "momentum": None,
                "nesterov": False,
            },
        }
    ),
    "adamw": ConfigDict(
        {
            "name": "adamw",
            "params": {
                "b1": 0.9,
                "b2": 0.999,
                "eps": 1e-8,
                "eps_root": 0.0,
                "weight_decay": 1e-3,
            },
        }
    ),
}


def get_config(optimizer_name: str = None):
    if optimizer_name is None:
        optimizer_name = "adamw"
        logging.info(f"No optimizer name provided, using {optimizer_name} as default")

    if optimizer_name not in available_optimizers:
        raise NotImplementedError(
            f"Scheduler {optimizer_name} not implemented. Implemented models: {available_optimizers.keys()}."
        )

    model_cfg = available_optimizers.get(optimizer_name)

    return model_cfg
