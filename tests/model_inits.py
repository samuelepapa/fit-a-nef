import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest

from fit_a_nef import MLP, SIREN, FourierNet, GaborNet, RFFNet

MIN_INIT_STD = 0.25
MAX_INIT_STD = 2.00
BATCH_SIZE = 4096


class TestGabor(absltest.TestCase):
    def test_init(self):
        np.random.seed(42)
        for _ in range(5):
            input_dim = 2
            output_dim = np.random.randint(1, 16)
            hidden_dim = np.random.randint(32, 128)
            num_filters = np.random.randint(2, 16)
            seed = np.random.randint(0, 2**32)
            model = GaborNet(output_dim=output_dim, hidden_dim=hidden_dim, num_filters=num_filters)
            x = jax.random.normal(jax.random.PRNGKey(seed), (BATCH_SIZE, input_dim))
            out, _ = model.init_with_output(jax.random.PRNGKey(seed + 1), x)
            out_std = jnp.std(out, axis=0).mean()
            self.assertEqual(out.shape, (BATCH_SIZE, output_dim))
            self.assertGreaterEqual(out_std, MIN_INIT_STD)
            self.assertGreaterEqual(MAX_INIT_STD, out_std)


class TestFourier(absltest.TestCase):
    def test_init(self):
        np.random.seed(42)
        for _ in range(5):
            input_dim = 2
            output_dim = np.random.randint(1, 16)
            hidden_dim = np.random.randint(32, 128)
            num_filters = np.random.randint(2, 16)
            seed = np.random.randint(0, 2**32)
            model = FourierNet(
                output_dim=output_dim, hidden_dim=hidden_dim, num_filters=num_filters
            )
            x = jax.random.normal(jax.random.PRNGKey(seed), (BATCH_SIZE, input_dim))
            out, _ = model.init_with_output(jax.random.PRNGKey(seed + 1), x)
            out_std = jnp.std(out, axis=0).mean()
            self.assertEqual(out.shape, (BATCH_SIZE, output_dim))
            self.assertGreaterEqual(out_std, MIN_INIT_STD)
            self.assertGreaterEqual(MAX_INIT_STD, out_std)


class TestSIREN(absltest.TestCase):
    def test_init(self):
        np.random.seed(42)
        for _ in range(5):
            input_dim = 2
            output_dim = np.random.randint(1, 16)
            hidden_dim = np.random.randint(32, 128)
            num_layers = np.random.randint(2, 16)
            omega_0 = np.random.uniform(1.0, 30.0)
            seed = np.random.randint(0, 2**32)
            model = SIREN(
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                omega_0=omega_0,
            )
            x = jax.random.normal(jax.random.PRNGKey(seed), (BATCH_SIZE, input_dim))
            out, _ = model.init_with_output(jax.random.PRNGKey(seed + 1), x)
            out_std = jnp.std(out, axis=0).mean()
            self.assertEqual(out.shape, (BATCH_SIZE, output_dim))
            self.assertGreaterEqual(out_std, MIN_INIT_STD)
            self.assertGreaterEqual(MAX_INIT_STD, out_std)


class TestMLP(absltest.TestCase):
    def test_init(self):
        np.random.seed(42)
        for _ in range(5):
            input_dim = 2
            output_dim = np.random.randint(1, 16)
            hidden_dim = np.random.randint(32, 128)
            num_layers = np.random.randint(2, 16)
            seed = np.random.randint(0, 2**32)
            model = MLP(output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers)
            x = jax.random.normal(jax.random.PRNGKey(seed), (BATCH_SIZE, input_dim))
            out, _ = model.init_with_output(jax.random.PRNGKey(seed + 1), x)
            out_std = jnp.std(out, axis=0).mean()
            self.assertEqual(out.shape, (BATCH_SIZE, output_dim))
            self.assertGreaterEqual(out_std, MIN_INIT_STD)
            self.assertGreaterEqual(MAX_INIT_STD, out_std)


class TestRFFNet(absltest.TestCase):
    def test_init(self):
        np.random.seed(42)
        for _ in range(5):
            input_dim = 2
            output_dim = np.random.randint(1, 16)
            hidden_dim = 2 * np.random.randint(16, 64)
            num_layers = np.random.randint(2, 16)
            std = np.random.uniform(0.1, 1.0)
            learnable_coefficients = np.random.choice([True, False])
            seed = np.random.randint(0, 2**32)
            model = RFFNet(
                output_dim=output_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                std=std,
                learnable_coefficients=learnable_coefficients,
            )
            x = jax.random.normal(jax.random.PRNGKey(seed), (BATCH_SIZE, input_dim))
            out, _ = model.init_with_output(jax.random.PRNGKey(seed + 1), x)
            out_std = jnp.std(out, axis=0).mean()
            self.assertEqual(out.shape, (BATCH_SIZE, output_dim))
            self.assertGreaterEqual(out_std, MIN_INIT_STD)
            self.assertGreaterEqual(MAX_INIT_STD, out_std)
