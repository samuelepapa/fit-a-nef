import pathlib

from absl import logging
from ml_collections import ConfigDict

available_nefs = {
    "MLP": ConfigDict(
        {
            "name": "MLP",
            "params": {
                "hidden_dim": 15,
                "num_layers": 3,
            },
        }
    ),
    "FourierNet": ConfigDict(
        {
            "name": "FourierNet",
            "params": {
                "num_filters": 8,
                "hidden_dim": 8,
                "input_scale": 16,
                "weight_scale": 6.0,
            },
        }
    ),
    "SIREN": ConfigDict(
        {
            "name": "SIREN",
            "params": {
                "hidden_dim": 8,
                "num_layers": 3,
                "omega_0": 8.0,
            },
        }
    ),
    "GaborNet": ConfigDict(
        {
            "name": "GaborNet",
            "params": {
                "num_filters": 10,
                "hidden_dim": 15,
                "input_scale": 256.0,
                "alpha": 0.16666666666666666,
                "beta": 1.0,
                "weight_scale": 6.0,
            },
        }
    ),
    "RFFNet": ConfigDict(
        {
            "name": "RFFNet",
            "params": {
                "num_layers": 3,
                "hidden_dim": 32,
                "std": 3.5,
                "learnable_coefficients": True,
            },
        }
    ),
}


def get_config(model_name: str = None):
    if model_name is None:
        model_name = "SIREN"
        logging.info(f"No model name provided, using {model_name} as default")

    if model_name not in available_nefs:
        raise NotImplementedError(
            f"Model {model_name} not implemented. Implemented models: {available_nefs.keys()}."
        )

    model_cfg = available_nefs.get(model_name)

    return model_cfg
