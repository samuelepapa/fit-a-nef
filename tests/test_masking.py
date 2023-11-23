import jax
import numpy as np
from absl import logging
from absl.testing import absltest
from ml_collections import ConfigDict

from config.utils import load_default_nef_cfg
from fit_a_nef import RandomInit, SignalImageTrainer

MIN_INIT_STD = 0.25
MAX_INIT_STD = 2.00
BATCH_SIZE = 4096


class MaskingTest(absltest.TestCase):
    def test_mask(self):
        np.random.seed(42)
        init_rng = jax.random.PRNGKey(42)
        train_rng = jax.random.PRNGKey(43)
        num_images = 4
        output_dim = np.random.randint(1, 16)
        image_shape = (2, 3, 1)
        images = np.random.normal(size=(num_images, image_shape[0] * image_shape[1], output_dim))
        coords = np.random.uniform(size=(image_shape[0] * image_shape[1], 2))
        for model_name in ["RFFNet", "SIREN", "GaborNet", "FourierNet"]:
            logging.info(f"Testing {model_name}")
            nef_cfg = load_default_nef_cfg(model_name)
            nef_cfg.params.output_dim = output_dim
            scheduler_cfg = ConfigDict({"name": "constant_schedule", "params": {"value": 1e-3}})
            optimizer_cfg = ConfigDict({"name": "adam", "params": {}})
            initializer = RandomInit(init_rng=init_rng)
            trainer = SignalImageTrainer(
                signals=images,
                coords=coords,
                train_rng=train_rng,
                nef_cfg=nef_cfg,
                scheduler_cfg=scheduler_cfg,
                optimizer_cfg=optimizer_cfg,
                initializer=initializer,
                log_cfg=None,
                num_steps=100,
                verbose=False,
                masked_portion=0.8,
            )

            trainer.compile()
