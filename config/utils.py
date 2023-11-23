import os
from pathlib import Path
from typing import Optional

from absl import flags, logging
from ml_collections import ConfigDict

from config.nef import available_nefs


def find_env_path(env_var: str = "NEF_PATH", default_path: str = "saved_models"):
    if env_var not in os.environ:
        env_path = Path(default_path).absolute()
        env_path.mkdir(parents=True, exist_ok=True)
        logging.warning(f"{env_var} environment variable not set, using default value {env_path}")
        return str(env_path)
    else:
        return str(Path(os.environ[env_var]).absolute())


def load_cfgs(
    _TASK_FILE,
    _NEF_FILE,
    _SCHEDULER_FILE: Optional[flags.FlagHolder] = None,
    _OPTIMIZER_FILE: Optional[flags.FlagHolder] = None,
):
    cfg = _TASK_FILE.value
    nef_cfg = _NEF_FILE.value

    # TODO find a way to not have to do this
    nef_cfg.unlock()
    nef_cfg.params.output_dim = cfg.dataset.get("out_channels", 1)
    nef_cfg.lock()

    if _SCHEDULER_FILE is not None:
        scheduler_cfg = _SCHEDULER_FILE.value
        cfg.unlock()
        cfg["scheduler"] = scheduler_cfg
        cfg.lock()

    if _OPTIMIZER_FILE is not None:
        optimizer_cfg = _OPTIMIZER_FILE.value
        cfg.unlock()
        cfg["optimizer"] = optimizer_cfg
        cfg.lock()

    return cfg, nef_cfg


def load_default_nef_cfg(name: str, custom_nefs: Optional[ConfigDict] = None):
    if custom_nefs is None:
        nef_cfgs = available_nefs
    else:
        nef_cfgs = custom_nefs

    if name not in nef_cfgs:
        raise NotImplementedError(
            f"Model {name} not implemented. Implemented models: {nef_cfgs.keys()}."
        )

    model_cfg = ConfigDict(nef_cfgs[name])

    return model_cfg
