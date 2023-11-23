from absl import logging
from ml_collections import ConfigDict

available_schedulers = {
    "warmup_cosine_decay_schedule": ConfigDict(
        {
            "name": "warmup_cosine_decay_schedule",
            "params": {
                "init_value": 1e-6,
                "peak_value": 1e-3,
                "warmup_steps": 10,
                "decay_steps": 1000,
                "end_value": 1e-6,
            },
        }
    ),
    "constant_schedule": ConfigDict(
        {
            "name": "constant_schedule",
            "params": {
                "value": 5e-4,
            },
        }
    ),
}


def get_config(scheduler_name: str = None):
    if scheduler_name is None:
        scheduler_name = "constant_schedule"
        logging.info(f"No scheduler name provided, using {scheduler_name} as default")

    if scheduler_name not in available_schedulers:
        raise NotImplementedError(
            f"Scheduler {scheduler_name} not implemented. Implemented models: {available_schedulers.keys()}."
        )

    model_cfg = available_schedulers.get(scheduler_name)

    return model_cfg
