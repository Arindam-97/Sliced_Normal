"""The original clipped likelihood, evaluated with two computational changes.

1. Cache S = Z.T @ Z / n instead of scanning training rows at each evaluation.
2. Cache grid energies at the last B and reuse them for an analytic gradient.

The grid, feature ordering, FMLE, SPD geometry, steepest-descent steps,
partition floor, and Frobenius penalty match the original default fitter.
Stopping is checked at the point that is returned, before another step is taken.
"""

import time
from copy import deepcopy

import numpy as np
from scipy.special import logsumexp
from pymanopt import Problem
from pymanopt.function import numpy as pymanopt_numpy
from pymanopt.manifolds import SymmetricPositiveDefinite
from pymanopt.optimizers import SteepestDescent

from .features import get_F_and_Random_Samples
from .fmle import FMLE


class _SamePointSteepestDescent(SteepestDescent):
    """Pymanopt steepest descent with consistent stopping and result metadata.

    Pymanopt 2.2.1 steps before checking the previous gradient. This local
    override checks the current point first, using the same manifold, line
    search, and stopping thresholds. Iterations count completed search steps;
    the final cost and gradient norm always describe the returned point.
    """

    def run(self, problem, *, initial_point=None, reuse_line_searcher=False):
        manifold = problem.manifold
        gradient = problem.riemannian_gradient
        cost_evaluations = 0

        def objective(point):
            nonlocal cost_evaluations
            cost_evaluations += 1
            return problem.cost(point)

        if not reuse_line_searcher or self.line_searcher is None:
            self.line_searcher = deepcopy(self._line_searcher)
        point = manifold.random_point() if initial_point is None else initial_point
        self._initialize_log(
            optimizer_parameters={"line_searcher": self.line_searcher}
        )
        iteration = 0
        step_size = np.inf
        start_time = time.time()
        if self._verbosity >= 1:
            print("Optimizing...")
        if self._verbosity >= 2:
            print("Iteration    Cost                       Gradient norm")

        while True:
            cost = float(objective(point))
            grad = gradient(point)
            gradient_norm = manifold.norm(point, grad)
            if self._verbosity >= 2:
                print(f"{iteration:5d}        {cost:+.16e}    {gradient_norm:.8e}")
            self._add_log_entry(
                iteration=iteration, point=point, cost=cost,
                gradient_norm=gradient_norm,
            )

            # All stopping tests and the returned metadata concern this point.
            # As in Pymanopt's SteepestDescent, the active limits are time,
            # iterations, gradient norm, and step size; count actual cost calls
            # for reporting rather than its iteration-based estimate.
            if not np.isfinite(cost) or not np.isfinite(gradient_norm):
                stopping_criterion = "Terminated - non-finite cost or gradient norm."
            else:
                stopping_criterion = self._check_stopping_criterion(
                    start_time=start_time, iteration=iteration,
                    gradient_norm=gradient_norm, step_size=step_size,
                )
                if stopping_criterion is None and gradient_norm == 0:
                    stopping_criterion = "Terminated - zero gradient norm reached."
            if stopping_criterion:
                break

            step_size, point = self.line_searcher.search(
                objective, manifold, point, -grad, cost, -(gradient_norm**2)
            )
            iteration += 1

        if self._verbosity >= 1:
            print(stopping_criterion)
            print("")
        return self._return_result(
            start_time=start_time, point=point, cost=cost, iterations=iteration,
            stopping_criterion=stopping_criterion,
            cost_evaluations=cost_evaluations, step_size=step_size,
            gradient_norm=gradient_norm,
        )


class CachedObjective:
    """Cost and Euclidean gradient on a fixed integration grid.

    Training features are reduced to their second moment during construction.
    The grid is copied so that external changes cannot invalidate cached values.
    A private copy of the last B also detects in-place changes by a caller.
    This one-point cache is intended for one sequential optimizer, not threads.
    """

    def __init__(self, Z, Z_grid, volume, penalty_lambda=0.0,
                 winsor_threshold=1e-10):
        Z = np.asarray(Z, dtype=np.float64)
        grid = np.asarray(Z_grid, dtype=np.float64)
        if (Z.ndim != 2 or grid.ndim != 2 or not len(Z) or not len(grid)
                or Z.shape[1] == 0 or Z.shape[1] != grid.shape[1]):
            raise ValueError("Z and Z_grid must be nonempty matrices with matching columns")
        if not np.isfinite(Z).all() or not np.isfinite(grid).all():
            raise ValueError("Features must be finite")
        if not np.isfinite(volume) or volume <= 0:
            raise ValueError("Integration volume must be finite and positive")
        if not np.isfinite(winsor_threshold) or winsor_threshold < 0:
            raise ValueError("winsor_threshold must be finite and nonnegative")
        if not np.isfinite(penalty_lambda):
            raise ValueError("penalty_lambda must be finite")

        self.second_moment = (Z.T @ Z) / len(Z)
        self.second_moment.setflags(write=False)
        self.grid = np.array(grid, copy=True)
        self.grid.setflags(write=False)
        self.penalty_lambda = float(penalty_lambda)
        self.log_floor = (np.log(winsor_threshold)
                          if winsor_threshold > 0 else -np.inf)
        self.log_volume = np.log(volume)
        self.log_grid_count = np.log(len(grid))
        self._B = None
        self._gradient = None
        self.energy_evaluations = 0
        self.cost_requests = 0
        self.gradient_requests = 0

    def _evaluate(self, B):
        B = np.asarray(B, dtype=np.float64)
        if B.shape != self.second_moment.shape or not np.isfinite(B).all():
            raise ValueError("B must be a finite matrix matching the feature dimension")
        if self._B is not None and np.array_equal(B, self._B):
            return

        # Retain the original contraction; feature/kernel changes are out of scope.
        energy = np.einsum("ij,jk,ik->i", self.grid, B, self.grid)
        log_sum = logsumexp(-energy)
        raw_log_integral = log_sum - self.log_grid_count + self.log_volume
        norm = np.sqrt(np.sum(B ** 2))
        self._cost = float(
            np.sum(B * self.second_moment)
            + max(raw_log_integral, self.log_floor)
            + self.penalty_lambda * norm
        )
        # Python max(raw, floor) chooses its first argument at equality, as in
        # the original autograd objective. Below the floor its derivative is 0.
        self._partition_active = raw_log_integral >= self.log_floor
        self._weights = (np.exp(-energy - log_sum)
                         if self._partition_active else None)
        self._norm = norm
        self._B = B.copy()
        self._gradient = None
        self.energy_evaluations += 1

    def cost(self, B):
        self.cost_requests += 1
        self._evaluate(B)
        return self._cost

    def gradient(self, B):
        """Euclidean derivative, including the original partition floor."""
        self.gradient_requests += 1
        self._evaluate(B)
        if self._gradient is None:
            gradient = self.second_moment.copy()
            if self._partition_active:
                # The normalized weights already include the denominator;
                # there is no additional division by integration volume.
                gradient -= self.grid.T @ (self._weights[:, None] * self.grid)
            if self.penalty_lambda != 0 and self._norm > 0:
                gradient += self.penalty_lambda * self._B / self._norm
            self._gradient = gradient
        # Optimizer/caller mutation must not corrupt the cached derivative.
        return self._gradient.copy()


def estimate_optimal_B(data, degree=2, n_grid=10000, verbosity=2,
                       max_iterations=1000, min_gradient_norm=1e-4,
                       penalty_lambda=0.0, winsor_threshold=1e-10):
    """Drop-in replacement for the original default estimate_optimal_B.

    Seed NumPy identically before old/new calls for the same integration points.
    Returns the same kind of B matrix, usable with the original sampler and
    likelihood evaluator. Floating-point summation may change solver trajectories.
    """
    Z_df, Z_grid_df, volume = get_F_and_Random_Samples(data, degree, n_grid)
    Z = np.asarray(Z_df.values, dtype=np.float64)
    Z_grid = np.asarray(Z_grid_df.values, dtype=np.float64)
    B_fmle = FMLE(Z_df.iloc[:, 1:].to_numpy(dtype=float))
    objective = CachedObjective(Z, Z_grid, volume, penalty_lambda, winsor_threshold)
    manifold = SymmetricPositiveDefinite(Z.shape[1])

    @pymanopt_numpy(manifold)
    def cost(B):
        return objective.cost(B)

    @pymanopt_numpy(manifold)
    def gradient(B):
        return objective.gradient(B)

    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=gradient)
    optimizer = _SamePointSteepestDescent(
        verbosity=verbosity, max_iterations=max_iterations,
        min_gradient_norm=min_gradient_norm,
    )
    return optimizer.run(problem, initial_point=B_fmle).point


def estimate_optimal_B_with_grad(data, degree=2, n_grid=10000, verbosity=2,
                                 max_iterations=1000, min_gradient_norm=1e-6,
                                 penalty_lambda=0.0):
    """Cached analytic-gradient fit without the default partition floor.

    Preserves this helper's historical unclipped objective and signature.
    The shared derivative uses normalized weights without an extra volume factor.
    """
    return estimate_optimal_B(
        data, degree=degree, n_grid=n_grid, verbosity=verbosity,
        max_iterations=max_iterations, min_gradient_norm=min_gradient_norm,
        penalty_lambda=max(penalty_lambda, 0.0), winsor_threshold=0.0,
    )


def compute_integral(data, B, d, n_grid=10000):
    """
    Efficiently approximate the integral:
        ∫ exp(-zᵀ B z) dz
    over a bounding box enclosing the data, using Monte Carlo with uniform samples.

    Inputs:
        data : pd.DataFrame or np.ndarray
        B : np.ndarray, symmetric positive definite matrix (m × m)
        d : int, polynomial degree for feature expansion
        n_grid : int, number of random samples

    Returns:
        integral : float, Monte Carlo estimate of the integral
    """
    _, Z_grid, volume = get_F_and_Random_Samples(data, d, n_grid)

    # Compute quadratic form zᵀ B z for each z in the grid
    quad_vals = np.einsum("ij,jk,ik->i", Z_grid, B, Z_grid)

    # Use log-sum-exp trick to compute log(mean(exp(-quad_vals)))
    log_mean_exp = logsumexp(-quad_vals) - np.log(len(Z_grid))

    # Final estimate: integral ≈ mean × volume
    integral = np.exp(log_mean_exp) * volume
    return integral
