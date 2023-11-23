import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from flax.core.frozen_dict import FrozenDict
from ml_collections import ConfigDict

from fit_a_nef import RandomInit, SignalImageTrainer
from fit_a_nef.utils import (
    flatten_dict,
    flatten_params,
    unflatten_dict,
    unflatten_params,
)
from tests.utils import dict_equal


class TestFlattenAndUnflattenDict(absltest.TestCase):
    def test_flatten_dict(self):
        d = {
            "a": 1,
            "b": 2,
            "c": {"d": 3, "e": 4},
            "f": {"g": {"h": 5, "i": 6}},
            "j": 7,
            "k": {"l": {"m": 8}},
        }
        flat_d = flatten_dict(d)
        self.assertEqual(flat_d["a"], 1)
        self.assertEqual(flat_d["b"], 2)
        self.assertEqual(flat_d["c.d"], 3)
        self.assertEqual(flat_d["c.e"], 4)
        self.assertEqual(flat_d["f.g.h"], 5)
        self.assertEqual(flat_d["f.g.i"], 6)
        self.assertEqual(flat_d["j"], 7)
        self.assertEqual(flat_d["k.l.m"], 8)

    def test_unflatten_dict(self):
        d = {"a": 1, "b": 2, "c.d": 3, "c.e": 4, "f.g.h": 5, "f.g.i": 6, "j": 7, "k.l.m": 8}
        unflat_d = unflatten_dict(d)
        self.assertEqual(unflat_d["a"], 1)
        self.assertEqual(unflat_d["b"], 2)
        self.assertEqual(unflat_d["c"]["d"], 3)
        self.assertEqual(unflat_d["c"]["e"], 4)
        self.assertEqual(unflat_d["f"]["g"]["h"], 5)
        self.assertEqual(unflat_d["f"]["g"]["i"], 6)
        self.assertEqual(unflat_d["j"], 7)
        self.assertEqual(unflat_d["k"]["l"]["m"], 8)

    def test_flatten_and_unflatten(self):
        d = {
            "a": 1,
            "b": 2,
            "c": {"d": 3, "e": 4},
            "f": {"g": {"h": 5, "i": 6}},
            "j": 7,
            "k": {"l": {"m": 8}},
        }
        flat_d = flatten_dict(d)
        unflat_d = unflatten_dict(flat_d)
        self.assertEqual(d, unflat_d)


class TestFlattenAndUnflattenParams(absltest.TestCase):
    def test_flatten_params(self):
        params = FrozenDict(
            {
                "a": jnp.array([1, 2, 3]),
                "b": jnp.array(
                    [
                        4,
                    ]
                ),
                "c": {"d": jnp.array([5, 6]), "e": jnp.array([7, 8, 9, 10])},
                "f": {"g": {"h": jnp.array([[11, 12], [13, 14]]), "i": jnp.array([[[15, 16]]])}},
                "j": jnp.array([17, 18, 19]),
            }
        )
        param_config, flat_params = flatten_params(params)
        self.assertEqual(
            param_config,
            [
                ("a", (3,)),
                ("b", (1,)),
                ("c.d", (2,)),
                ("c.e", (4,)),
                ("f.g.h", (2, 2)),
                ("f.g.i", (1, 1, 2)),
                ("j", (3,)),
            ],
        )
        self.assertEqual(flat_params.shape, (3 + 1 + 2 + 4 + 2 * 2 + 1 * 1 * 2 + 3,))
        self.assertTrue((flat_params == jnp.arange(1, 20, dtype=params["a"].dtype)).all())

        params = FrozenDict(
            {
                "c_2.bias": jnp.array([9, 10]),
                "c_2.kernel": jnp.array([11, 12]),
                "a_0.bias": jnp.array([1, 2, 3]),
                "a_0.kernel": jnp.array([4, 5, 6]),
                "z_1.kernel": jnp.array(
                    [
                        8,
                    ]
                ),
                "z_1.bias": jnp.array(
                    [
                        7,
                    ]
                ),
                "last_j.kernel": jnp.array([16, 17, 18]),
                "last_j.bias": jnp.array([13, 14, 15]),
            }
        )
        nef_cfg = ConfigDict({"num_layers": 4})

        def param_key(param_name):
            if param_name.startswith("last_"):
                index = 2 * (nef_cfg.get("num_layers") - 1)
            else:
                index = 2 * int(param_name.split(".")[0].split("_")[-1])

            if param_name.endswith(".bias"):
                return index
            elif param_name.endswith(".kernel"):
                return index + 1
            else:
                raise ValueError(
                    f"param_name (`{param_name}`) must end with either `.bias` or `.kernel`."
                )

        param_config, flat_params = flatten_params(params, param_key=param_key)
        self.assertEqual(
            param_config,
            [
                ("a_0.bias", (3,)),
                ("a_0.kernel", (3,)),
                ("z_1.bias", (1,)),
                ("z_1.kernel", (1,)),
                ("c_2.bias", (2,)),
                ("c_2.kernel", (2,)),
                ("last_j.bias", (3,)),
                ("last_j.kernel", (3,)),
            ],
        )

        self.assertEqual(flat_params.shape, (3 + 3 + 1 + 1 + 2 + 2 + 3 + 3,))
        self.assertTrue((flat_params == jnp.arange(1, 19, dtype=params["a_0.bias"].dtype)).all())

    def test_unflatten_params(self):
        param_config = [
            ("a", (3,)),
            ("b", (1,)),
            ("c.d", (2,)),
            ("c.e", (4,)),
            ("f.g.h", (2, 2)),
            ("f.g.i", (1, 1, 2)),
            ("j", (3,)),
        ]
        flat_params = jnp.arange(1, 20, dtype=jnp.float32)
        params = unflatten_params(param_config, flat_params)
        self.assertEqual(params["a"].shape, (3,))
        self.assertEqual(params["b"].shape, (1,))
        self.assertEqual(params["c"]["d"].shape, (2,))
        self.assertEqual(params["c"]["e"].shape, (4,))
        self.assertEqual(params["f"]["g"]["h"].shape, (2, 2))
        self.assertEqual(params["f"]["g"]["i"].shape, (1, 1, 2))
        self.assertEqual(params["j"].shape, (3,))
        self.assertTrue((params["a"] == jnp.array([1, 2, 3], dtype=params["a"].dtype)).all())
        self.assertTrue(
            (
                params["b"]
                == jnp.array(
                    [
                        4,
                    ],
                    dtype=params["b"].dtype,
                )
            ).all()
        )
        self.assertTrue(
            (params["c"]["d"] == jnp.array([5, 6], dtype=params["c"]["d"].dtype)).all()
        )
        self.assertTrue(
            (params["c"]["e"] == jnp.array([7, 8, 9, 10], dtype=params["c"]["e"].dtype)).all()
        )
        self.assertTrue(
            (
                params["f"]["g"]["h"]
                == jnp.array([[11, 12], [13, 14]], dtype=params["f"]["g"]["h"].dtype)
            ).all()
        )
        self.assertTrue(
            (
                params["f"]["g"]["i"] == jnp.array([[[15, 16]]], dtype=params["f"]["g"]["i"].dtype)
            ).all()
        )
        self.assertTrue((params["j"] == jnp.array([17, 18, 19], dtype=params["j"].dtype)).all())

    def test_flatten_and_unflatten_params(self):
        params = FrozenDict(
            {
                "a": jnp.array([1, 2, 3]),
                "b": jnp.array(
                    [
                        4,
                    ]
                ),
                "c": {"d": jnp.array([5, 6]), "e": jnp.array([7, 8, 9, 10])},
                "f": {"g": {"h": jnp.array([[11, 12], [13, 14]]), "i": jnp.array([[[15, 16]]])}},
                "j": jnp.array([17, 18, 19]),
            }
        )
        param_config, flat_params = flatten_params(params, param_key=lambda x: x)
        recon_params = unflatten_params(param_config, flat_params)
        self.assertListEqual(list(params.keys()), list(recon_params.keys()))
        self.assertTrue(dict_equal(params, recon_params, lambda x, y: (x == y).all()))


class TestModelSaveAndLoad(absltest.TestCase):
    SAVE_PATH = "test_model.hdf5"

    def setUp(self):
        NUM_MODELS = 5
        self.init_rng = jax.random.PRNGKey(0)
        self.train_rng = jax.random.PRNGKey(1)
        initializer = RandomInit(init_rng=self.init_rng)
        images_shape = (32, 32, 3)
        self.trainer = SignalImageTrainer(
            signals=jnp.ones((NUM_MODELS, images_shape[0] * images_shape[1], 3)),
            coords=jnp.ones((images_shape[0] * images_shape[1], 2)),
            nef_cfg={
                "name": "FourierNet",
                "params": {"num_filters": 23, "hidden_dim": 4, "output_dim": 3},
            },
            optimizer_cfg={"name": "adam", "params": {}},
            scheduler_cfg={"name": "constant_schedule", "params": {"value": 1e-3}},
            initializer=initializer,
            train_rng=self.train_rng,
            num_steps=100,
            images_shape=images_shape,
        )

    def test1_save_model(self):
        self.trainer.save(TestModelSaveAndLoad.SAVE_PATH, test_data=jnp.array([1, 2, 3, 4, 5]))

    def test2_check_saved_hdf5(self):
        self.assertTrue(os.path.exists(TestModelSaveAndLoad.SAVE_PATH))
        with h5py.File(TestModelSaveAndLoad.SAVE_PATH) as f:
            data = f["params"][:]
            test_data = f["test_data"][:]
        self.assertEqual(len(data.shape), 2)
        self.assertEqual(data.shape[0], 5)
        params = jax.tree_util.tree_leaves(self.trainer.state.params)
        num_params = sum([np.prod(p.shape[1:]) for p in params])
        self.assertEqual(data.shape[1], num_params)
        self.assertTrue((test_data == jnp.array([1, 2, 3, 4, 5])).all())

    def test3_load_model(self):
        self.assertTrue(os.path.exists(TestModelSaveAndLoad.SAVE_PATH))
        prev_params = self.trainer.state.params
        new_rng = jax.random.split(self.init_rng, 1)[0]
        self.trainer.initializer = initializer = RandomInit(init_rng=new_rng)
        self.trainer.init_model(self.trainer.coords)
        self.assertFalse(
            dict_equal(prev_params, self.trainer.state.params, lambda x, y: (x == y).all())
        )
        self.trainer.load(TestModelSaveAndLoad.SAVE_PATH)
        self.assertTrue(
            dict_equal(prev_params, self.trainer.state.params, lambda x, y: (x == y).all())
        )

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.SAVE_PATH):
            os.remove(cls.SAVE_PATH)
