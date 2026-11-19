"""Numerical agreement with the original default fitter, not its alternate gradient."""

import unittest

import autograd.numpy as anp
from autograd import grad
from autograd.scipy.special import logsumexp
import numpy as np
import pandas as pd

from sliced_normals.features import Features, get_F_and_Random_Samples
from sliced_normals.legacy_optimization import estimate_optimal_B as original_fit
from sliced_normals.optimization import CachedObjective, estimate_optimal_B


def original_cost(Z, grid, volume, penalty, threshold):
    """The original default cost, including Python max and its tie convention."""
    def cost(B):
        empirical = anp.mean(anp.einsum("ij,jk,ik->i", Z, B, Z))
        energy = anp.einsum("ij,jk,ik->i", grid, B, grid)
        raw = logsumexp(-energy) - anp.log(len(grid)) + anp.log(volume)
        floor = anp.log(threshold) if threshold > 0 else -anp.inf
        return empirical + max(raw, floor) + penalty * anp.sqrt(anp.sum(B ** 2))
    return cost


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(72)
        self.Z = Features(pd.DataFrame(rng.normal(size=(80, 2))), 2).to_numpy()
        self.grid = Features(pd.DataFrame(rng.uniform(-2, 2, (150, 2))), 2).to_numpy()
        A = rng.normal(size=(6, 6))
        self.B = 0.02 * A.T @ A + np.eye(6) * 0.1
        self.volume = 16.0  # Unit volume would hide the old alternate-gradient bug.

    def test_cost_and_gradient_in_both_floor_branches(self):
        for penalty in [0.0, 0.07]:
            for threshold in [0.0, 1e-10, 1e6]:
                with self.subTest(penalty=penalty, threshold=threshold):
                    reference = original_cost(self.Z, self.grid, self.volume, penalty, threshold)
                    cached = CachedObjective(self.Z, self.grid, self.volume, penalty, threshold)
                    self.assertAlmostEqual(cached.cost(self.B), reference(self.B), places=12)
                    np.testing.assert_allclose(cached.gradient(self.B), grad(reference)(self.B),
                                               atol=2e-12, rtol=2e-12)

    def test_derivative_at_exact_floor_tie(self):
        Z = np.array([[1., 0.], [1., 1.]])
        grid = np.array([[1., 0.]])
        B = np.diag([2., 1.])
        threshold = np.exp(-2.)
        reference = original_cost(Z, grid, 1., 0., threshold)
        cached = CachedObjective(Z, grid, 1., 0., threshold)
        np.testing.assert_allclose(cached.gradient(B), grad(reference)(B), atol=1e-14)
        self.assertTrue(cached._partition_active)

    def test_symmetric_finite_differences(self):
        rng = np.random.default_rng(8)
        H = rng.normal(size=self.B.shape)
        H = (H + H.T) / 2
        H /= np.linalg.norm(H)
        for threshold in [1e-10, 1e6]:
            cached = CachedObjective(self.Z, self.grid, self.volume, 0.07, threshold)
            analytic = np.sum(cached.gradient(self.B) * H)
            eps = 1e-6
            numerical = (cached.cost(self.B + eps * H) - cached.cost(self.B - eps * H)) / (2 * eps)
            np.testing.assert_allclose(analytic, numerical, atol=2e-8, rtol=2e-8)

    def test_cost_gradient_share_one_energy_evaluation(self):
        cached = CachedObjective(self.Z, self.grid, self.volume)
        cached.cost(self.B)
        first = cached.gradient(self.B.copy())
        first[:] = 123  # Returned gradient must not mutate the cache.
        self.assertFalse(np.all(cached.gradient(self.B) == 123))
        self.assertEqual(cached.energy_evaluations, 1)
        changed = self.B.copy()
        cached.cost(changed)
        changed[0, 0] += 0.1
        cached.cost(changed)
        self.assertEqual(cached.energy_evaluations, 2)

    def test_input_array_mutation_does_not_stale_cache(self):
        cached = CachedObjective(self.Z, self.grid, self.volume)
        before = cached.cost(self.B)
        self.Z[:] = 100
        self.grid[:] = 100
        other_B = self.B + np.eye(6) * 0.01
        cached.cost(other_B)
        self.assertEqual(cached.cost(self.B), before)

    def test_constant_energy_shift_above_floor(self):
        cached = CachedObjective(self.Z, self.grid, self.volume, winsor_threshold=0)
        shifted = self.B.copy()
        shifted[0, 0] += 5
        self.assertAlmostEqual(cached.cost(self.B), cached.cost(shifted), places=12)
        np.testing.assert_allclose(cached.gradient(self.B), cached.gradient(shifted), atol=1e-12)

    def test_invalid_shapes_and_volume(self):
        with self.assertRaises(ValueError):
            CachedObjective(self.Z, self.grid[:, :-1], self.volume)
        with self.assertRaises(ValueError):
            CachedObjective(self.Z, self.grid, 0)

    def test_actual_fit_api_matches_original_on_shared_grid(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=120)
        data = pd.DataFrame({'x': x, 'y': 0.4*x*x + rng.normal(size=120)})
        settings = dict(degree=2, n_grid=300, verbosity=0, max_iterations=15,
                        min_gradient_norm=1e-4, penalty_lambda=0.01)
        np.random.seed(99)
        old = original_fit(data, **settings)
        np.random.seed(99)
        new = estimate_optimal_B(data, **settings)
        np.random.seed(99)
        Z, grid, volume = get_F_and_Random_Samples(data, 2, 300)
        check = original_cost(Z.to_numpy(), grid.to_numpy(), volume, 0.01, 1e-10)
        np.testing.assert_allclose(check(new), check(old), atol=1e-7, rtol=1e-7)
        self.assertGreater(np.linalg.eigvalsh(new).min(), 0)


if __name__ == "__main__":
    unittest.main()
