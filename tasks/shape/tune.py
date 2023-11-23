import time
from functools import partial
from pathlib import Path

import numpy as np
import optuna
from absl import app, logging
from ml_collections import ConfigDict, config_flags

from config import load_cfgs
from studies_objectives.utils import load_module

_TASK_FILE = config_flags.DEFINE_config_file("task", default="config/image.py:tune")
_NEF_FILE = config_flags.DEFINE_config_file("nef", default="config/nef.py")
_SCHEDULER_FILE = config_flags.DEFINE_config_file("scheduler", default="config/scheduler.py")
_OPTIMIZER_FILE = config_flags.DEFINE_config_file("optimizer", default="config/optimizer.py")
_CLASSIFIER_FILE = config_flags.DEFINE_config_file(
    "classifier", default="config/classification.py"
)


def main(_):
    # TODO IMPORTANT: PARALLELISM CAN ONLY HAPPEN ACROSS DISJOINT PARAMETERS,
    # OTHERWISE THE SAMPLER WON'T REMEMBER THE STATE AND SAMPLE THE SAME PARAMS
    # AND THE BAYESIN OPTIMIZATION WON'T WORK
    # for example, run the first job with SIREN MNIST and the second with SIREN CIFAR10
    # then the parameters explored are the same, but with different datasets.
    # if we want to have different parameters we just change the --task.optuna.seed to
    # something different from 43.
    cfg, nef_cfg = load_cfgs(_TASK_FILE, _NEF_FILE, _SCHEDULER_FILE, _OPTIMIZER_FILE)
    classifier_cfg = _CLASSIFIER_FILE.value

    # Unlock to allow changing during tuning
    cfg.unlock()
    nef_cfg.unlock()

    # load the study with the desired objective function
    study_path = Path(f"studies_objectives/{cfg.study_objective}.py")
    study_module = load_module(study_path)

    # if a search space is present in the study_module, use it
    if hasattr(study_module, "search_space"):
        sampler = optuna.samplers.BruteForceSampler()
    else:
        # random search
        sampler = optuna.samplers.TPESampler(multivariate=True)

    # create a unique experiment folder based on optuna study_name and wandb project
    experiment_folder = Path(cfg.nef_dir) / Path(cfg.wandb.project) / Path(cfg.optuna.study_name)
    experiment_folder.mkdir(parents=True, exist_ok=True)

    tuning_db_folder = experiment_folder / Path("optunadb")
    tuning_db_folder.mkdir(parents=True, exist_ok=True)

    # load the study with optuna
    # study = optuna.load_study(
    #     study_name=cfg.optuna.study_name,
    #     storage=f"sqlite:///{tuning_db_folder}/{cfg.optuna.study_name}.db",
    # )

    study = optuna.create_study(
        directions=cfg.optuna.direction,
        study_name=cfg.optuna.study_name,
        storage=f"sqlite:///{tuning_db_folder}/{cfg.optuna.study_name}.db",
        load_if_exists=True,
        sampler=sampler,
    )

    # pass the cfg and nef_cfg to the objective function
    part_objective = partial(
        study_module.objective,
        cfg=cfg,
        nef_cfg=nef_cfg,
        classifier_cfg=classifier_cfg,
        experiment_folder=experiment_folder,
    )

    # run the study
    study.optimize(part_objective, n_trials=cfg.optuna.num_trials, n_jobs=cfg.optuna.n_jobs)


if __name__ == "__main__":
    app.run(main)
