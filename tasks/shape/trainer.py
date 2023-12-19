from typing import Any, Dict, Tuple

import jax.numpy as jnp
from torch.utils.data import DataLoader

from fit_a_nef import InitModel, SignalShapeTrainer


class ShapeTrainer(SignalShapeTrainer):
    def __init__(
        self,
        loader: DataLoader,
        train_rng: jnp.ndarray,
        nef_cfg: Dict[str, Any],
        scheduler_cfg: Dict[str, Any],
        optimizer_cfg: Dict[str, Any],
        log_cfg: Dict[str, Any],
        initializer: InitModel,
        num_steps: int,
        verbose: bool = False,
        num_points: Tuple[int, int] = (2048, 2048),
    ):
        coords, occupancies, labels = next(iter(loader))
        self.labels = labels
        super().__init__(
            coords=coords,
            occupancies=occupancies,
            train_rng=train_rng,
            nef_cfg=nef_cfg,
            scheduler_cfg=scheduler_cfg,
            optimizer_cfg=optimizer_cfg,
            log_cfg=log_cfg,
            initializer=initializer,
            num_steps=num_steps,
            verbose=verbose,
            num_points=num_points,
        )

    def verbose_train_model(self):
        # also plot the 3D shapes
        return super().verbose_train_model()

    def marching_cubes(self):
        pass
