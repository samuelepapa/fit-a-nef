import jax
import numpy as np
from absl.testing import absltest

from fit_a_nef.initializers import RandomInit, SharedInit
from fit_a_nef.utils import get_nef
from tests.utils import dict_equal


class RandomInitTest(absltest.TestCase):
    NUM_SIGNALS = 5
    INPUT_SHAPE = (10, 2)

    def test_random_init(self):
        init_rng = jax.random.PRNGKey(42)
        nef_cfg = {
            "name": "SIREN",
            "params": {
                "num_layers": 3,
                "hidden_dim": 16,
                "omega_0": 30,
                "output_dim": 3,
            },
        }
        example_input = jax.random.normal(init_rng, RandomInitTest.INPUT_SHAPE)
        initializer = RandomInit(init_rng=init_rng)
        params = initializer(get_nef(nef_cfg), example_input, RandomInitTest.NUM_SIGNALS)
        # any pair of models should have different parameters
        for model_id in range(RandomInitTest.NUM_SIGNALS):
            for other_model_id in range(model_id + 1, RandomInitTest.NUM_SIGNALS):
                self.assertFalse(
                    dict_equal(
                        jax.tree_map(lambda x: x[model_id], params),
                        jax.tree_map(lambda x: x[other_model_id], params),
                        lambda x, y: (x == y).all(),
                    )
                )

        new_params = initializer(get_nef(nef_cfg), example_input, RandomInitTest.NUM_SIGNALS)
        # any pair of models should have different parameters
        for model_id in range(RandomInitTest.NUM_SIGNALS):
            for other_model_id in range(model_id + 1, RandomInitTest.NUM_SIGNALS):
                self.assertFalse(
                    dict_equal(
                        jax.tree_map(lambda x: x[model_id], new_params),
                        jax.tree_map(lambda x: x[other_model_id], new_params),
                        lambda x, y: (x == y).all(),
                    )
                )
                self.assertFalse(
                    dict_equal(
                        jax.tree_map(lambda x: x[model_id], params),
                        jax.tree_map(lambda x: x[other_model_id], new_params),
                        lambda x, y: (x == y).all(),
                    )
                )

    def test_shared_init(self):
        init_rng = jax.random.PRNGKey(42)
        nef_cfg = {
            "name": "SIREN",
            "params": {
                "num_layers": 3,
                "hidden_dim": 16,
                "omega_0": 30,
                "output_dim": 3,
            },
        }
        example_input = jax.random.normal(init_rng, RandomInitTest.INPUT_SHAPE)
        initializer = SharedInit(init_rng=init_rng)
        params = initializer(get_nef(nef_cfg), example_input, RandomInitTest.NUM_SIGNALS)
        # any pair of models should have the same parameters
        for model_id in range(RandomInitTest.NUM_SIGNALS):
            for other_model_id in range(model_id + 1, RandomInitTest.NUM_SIGNALS):
                self.assertTrue(
                    dict_equal(
                        jax.tree_map(lambda x: x[model_id], params),
                        jax.tree_map(lambda x: x[other_model_id], params),
                        lambda x, y: (x == y).all(),
                    )
                )

        new_params = initializer(get_nef(nef_cfg), example_input, RandomInitTest.NUM_SIGNALS)
        # any pair of models should have the same parameters
        for model_id in range(RandomInitTest.NUM_SIGNALS):
            for other_model_id in range(model_id + 1, RandomInitTest.NUM_SIGNALS):
                self.assertTrue(
                    dict_equal(
                        jax.tree_map(lambda x: x[model_id], new_params),
                        jax.tree_map(lambda x: x[other_model_id], new_params),
                        lambda x, y: (x == y).all(),
                    )
                )
                self.assertTrue(
                    dict_equal(
                        jax.tree_map(lambda x: x[model_id], params),
                        jax.tree_map(lambda x: x[other_model_id], new_params),
                        lambda x, y: (x == y).all(),
                    )
                )

        new_init_rng = jax.random.PRNGKey(43)
        new_initializer = SharedInit(init_rng=new_init_rng)
        new_params = new_initializer(get_nef(nef_cfg), example_input, RandomInitTest.NUM_SIGNALS)
        # any pair of models should have different parameters
        for model_id in range(RandomInitTest.NUM_SIGNALS):
            self.assertFalse(
                dict_equal(
                    jax.tree_map(lambda x: x[model_id], new_params),
                    jax.tree_map(lambda x: x[model_id], params),
                    lambda x, y: (x == y).all(),
                )
            )


if __name__ == "__main__":
    absltest.main()
