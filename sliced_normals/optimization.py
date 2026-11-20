"""Cached matrix fitting and coefficient-only primal Sliced Normal fitting.

1. Cache S = Z.T @ Z / n instead of scanning training rows at each evaluation.
2. Cache grid energies at the last B and reuse them for an analytic gradient.

The grid, feature ordering, FMLE, SPD geometry, steepest-descent steps,
partition floor, and Frobenius penalty match the original default fitter.
Stopping is checked at the point that is returned, before another step is taken.
The optional primal fitter holds an FMLE-derived SOS basis fixed and optimizes
its positive coefficients using a convex, unclipped likelihood on a fixed grid.
"""

import time
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import OptimizeResult, minimize
from scipy.special import logsumexp
from pymanopt import Problem
from pymanopt.function import numpy as pymanopt_numpy
from pymanopt.manifolds import SymmetricPositiveDefinite
from pymanopt.optimizers import SteepestDescent

from .features import Features, get_F_and_Random_Samples, _normalize_sample_weight
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
                 winsor_threshold=1e-10, *, sample_weight=None):
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

        weights = _normalize_sample_weight(sample_weight, len(Z))
        self.second_moment = ((Z.T @ Z) / len(Z) if weights is None
                              else Z.T @ (weights[:, None] * Z))
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
                       penalty_lambda=0.0, winsor_threshold=1e-10, *,
                       sample_weight=None):
    """Drop-in replacement for the original default estimate_optimal_B.

    Seed NumPy identically before old/new calls for the same integration points.
    Returns the same kind of B matrix, usable with the original sampler and
    likelihood evaluator. Floating-point summation may change solver trajectories.

    sample_weight provides nonnegative finite observation weights with positive
    total. They are normalized and used in both FMLE and empirical moments.
    None preserves the legacy unweighted initializer and objective exactly.
    Explicit weights use feature-space ML covariance in FMLE (including uniform
    weights), rather than the legacy n-1 convention. The integration box still
    uses every input row, including zero-weight observations.
    """
    _normalize_sample_weight(sample_weight, len(data))
    Z_df, Z_grid_df, volume = get_F_and_Random_Samples(data, degree, n_grid)
    Z = np.asarray(Z_df.values, dtype=np.float64)
    Z_grid = np.asarray(Z_grid_df.values, dtype=np.float64)
    B_fmle = FMLE(Z_df.iloc[:, 1:].to_numpy(dtype=float), sample_weight=sample_weight)
    objective = CachedObjective(Z, Z_grid, volume, penalty_lambda, winsor_threshold,
                                sample_weight=sample_weight)
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
                                 penalty_lambda=0.0, *, sample_weight=None):
    """Cached analytic-gradient fit without the default partition floor.

    Preserves this helper's historical unclipped objective and signature.
    The shared derivative uses normalized weights without an extra volume factor.
    """
    return estimate_optimal_B(
        data, degree=degree, n_grid=n_grid, verbosity=verbosity,
        max_iterations=max_iterations, min_gradient_norm=min_gradient_norm,
        penalty_lambda=max(penalty_lambda, 0.0), winsor_threshold=0.0,
        sample_weight=sample_weight,
    )


def _fmle_sos_basis(features, weights, method, regularization):
    """Return A such that z(x) @ A gives centered, whitened polynomials.

    Covariance uses normalized sample weights (the feature-space ML moment).
    Regularization is added after scaling each feature to unit variance, so
    the ridge is dimensionless. Eigen and Cholesky factors reproduce the same
    initial energy but generally define different coefficient families.
    """
    mean = weights @ features
    centered = features - mean
    scale = np.sqrt(weights @ (centered ** 2))
    scale = np.where(scale > 0, scale, 1.0)
    standardized = centered / scale
    covariance = standardized.T @ (weights[:, None] * standardized)
    covariance = (covariance + covariance.T) / 2
    covariance += regularization * np.eye(covariance.shape[0])
    if not np.isfinite(covariance).all():
        raise ValueError("Feature covariance is not finite; rescale the input data")
    try:
        if method == "eigen":
            values, vectors = np.linalg.eigh(covariance)
            if np.any(values <= 0):
                raise np.linalg.LinAlgError("Nonpositive covariance eigenvalue")
            factor = vectors / np.sqrt(values)
        else:
            lower = np.linalg.cholesky(covariance)
            factor = solve_triangular(lower.T, np.eye(len(lower)), lower=False)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Feature covariance is singular; increase covariance_regularization"
        ) from error
    factor = factor / (np.sqrt(2.0) * scale[:, None])
    return np.vstack((-mean @ factor, factor))


class _PrimalObjective:
    """Weighted data moments and cached squared polynomials on a fixed grid."""

    def __init__(self, data_squares, grid_squares, volume, weights, *,
                 log_grid_weights=None):
        self.data_moment = weights @ data_squares
        self.grid = grid_squares
        if log_grid_weights is None:
            self.log_grid_weights = 0.0
            self.log_measure = np.log(volume) - np.log(len(grid_squares))
        else:
            self.log_grid_weights = np.asarray(log_grid_weights, dtype=float)
            if (self.log_grid_weights.shape != (len(grid_squares),)
                    or not np.isfinite(self.log_grid_weights).all()):
                raise ValueError("Integration log weights must be finite and match the grid")
            self.log_measure = 0.0
        self._coefficients = None
        self.evaluations = 0

    def __call__(self, coefficients):
        if (self._coefficients is None
                or not np.array_equal(coefficients, self._coefficients)):
            energy = self.grid @ coefficients
            data_energy = float(coefficients @ self.data_moment)
            minimum = float(energy.min())
            tolerance = 1e-12 * max(1.0, abs(data_energy), abs(minimum))
            # Along t*coefficients, t -> infinity, the cost has asymptotic
            # slope data_energy - min(grid_energy). A negative slope proves
            # that this quadrature likelihood has no finite minimum. This is
            # a sufficient diagnostic, not a certificate that other grids are
            # accurate; sparse grids can miss narrow high-density regions.
            if np.all(coefficients >= 0) and data_energy < minimum - tolerance:
                raise ValueError(
                    "The fixed integration grid makes the primal likelihood "
                    "unbounded along a positive coefficient direction. It misses "
                    "the data's high-density region. Increase n_grid or reduce "
                    "degree, and independently check normalization."
                )
            log_density = -energy + self.log_grid_weights
            log_sum = logsumexp(log_density)
            probabilities = np.exp(log_density - log_sum)
            self.log_partition = float(log_sum + self.log_measure)
            self.fun = float(data_energy + self.log_partition)
            self.jac = self.data_moment - probabilities @ self.grid
            self.integration_ess = float(1.0 / (probabilities @ probabilities))
            if not np.isfinite(self.fun) or not np.isfinite(self.jac).all():
                raise ValueError("Non-finite primal objective; rescale the input data")
            self._coefficients = coefficients.copy()
            self.evaluations += 1
        return self.fun, self.jac.copy()


def estimate_primal_B(data, degree=2, n_grid=10000, *, sample_weight=None,
                      basis="eigen", covariance_regularization=1e-8,
                      min_coefficient=1e-6, max_iterations=1000,
                      min_gradient_norm=1e-6, verbosity=1, return_info=False,
                      integration_points=None, proposal_log_density=None):
    """Fit positive coefficients of a fixed FMLE-derived sum-of-squares basis.

    For q nonconstant monomials, form q centered polynomials h_j and fit
    E(x) = sum_j coefficient_j * h_j(x)**2. Coefficients start at one (FMLE)
    and stay >= min_coefficient. This is a convex restricted-family fit, not
    optimization over every PSD matrix. Uses SciPy L-BFGS-B, not Pymanopt.

    Parameters
    ----------
    data : DataFrame or array of shape (n_samples, n_variables)
        Finite numeric observations; columns specify the feature order.
    degree, n_grid : int
        Positive polynomial degree and uniform integration sample count.
        The box is derived from all observations, including zero-weight rows,
        using the same rule as the matrix fitter. Seed NumPy for repeatability.
    sample_weight : array of shape (n_samples,), optional
        Nonnegative finite sample weights with positive total. Normalized
        internally; used in BOTH the feature-space FMLE and the likelihood.
    basis : {"eigen", "cholesky"}
        Decomposition of the scaled, regularized feature covariance. Both
        reconstruct the same initial energy, but allow different reweightings.
    covariance_regularization : float
        Nonnegative ridge on the unit-variance feature covariance. Unlike the
        legacy FMLE's n-1 covariance, this initializer uses weighted ML moments.
    min_coefficient : float
        Positive lower bound, at most one. Retains a positive feature-space
        precision even when the constrained optimum reaches the bound.
    max_iterations : int
        Nonnegative iteration limit; zero returns the FMLE initialization.
    min_gradient_norm : float
        Positive tolerance on the infinity norm of the projected gradient
        lambda - maximum(lambda - gradient, min_coefficient).
    verbosity : int
        Zero is silent; positive values print the final stopping diagnostics.
    return_info : bool
        If True, return (B, info), where info is a SciPy OptimizeResult with
        coefficients, basis, initial_B, initial_fun, log_partition, grid ESS,
        projected_gradient_norm and gradient_tolerance_met. Solver success
        alone does not imply that the requested gradient tolerance was met.
    integration_points : array of shape (n_draws, n_variables), optional
        Independent draws from a known proposal density, in the same physical
        coordinates as data. Overrides n_grid. The proposal must cover the
        entire integration box. Keep ALL draws, including outside-box points.
    proposal_log_density : array of shape (n_draws,), optional
        Log of the normalized proposal density at EVERY supplied draw. Required
        together with integration_points. Uses the unbiased integral estimate
        mean(1_box * exp(-energy) / proposal_density); its logarithm is still a
        finite-sample approximation. Outside-box points contribute zero but
        remain in the denominator. Do not supply only accepted/rejected samples
        with an unadjusted proposal density.

    Returns
    -------
    B : ndarray of shape (q+1, q+1)
        PSD Gram matrix in the existing constant-inclusive feature order,
        usable with the existing likelihood and MCMC helpers. It need not be
        strictly positive definite as an augmented matrix. The existing
        sampler remains unbounded, while fitting uses a bounded box.

    Notes
    -----
    The likelihood has no partition floor or matrix penalty. Cost/gradient
    use cached grid squares, requiring O(n_grid * q) work per evaluation.
    Convexity and stopping concern this fixed numerical grid; independent
    integration checks are still necessary. Subgroup marginals are not fixed.
    Raises ValueError if a positive coefficient direction proves the numerical
    grid likelihood is unbounded. Passing this check does not ensure accurate
    normalization or exclude every possible unbounded direction.
    """
    for name, value, minimum in (("degree", degree, 1), ("n_grid", n_grid, 1),
                                 ("max_iterations", max_iterations, 0)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < minimum):
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if basis not in ("eigen", "cholesky"):
        raise ValueError("basis must be 'eigen' or 'cholesky'")
    if not np.isfinite(covariance_regularization) or covariance_regularization < 0:
        raise ValueError("covariance_regularization must be finite and nonnegative")
    if not np.isfinite(min_coefficient) or not 0 < min_coefficient <= 1:
        raise ValueError("min_coefficient must be finite and in (0, 1]")
    if not np.isfinite(min_gradient_norm) or min_gradient_norm <= 0:
        raise ValueError("min_gradient_norm must be finite and positive")
    values = np.asarray(data, dtype=np.float64)
    if (values.ndim != 2 or len(values) < 2 or values.shape[1] == 0
            or not np.isfinite(values).all()):
        raise ValueError("data must be a finite 2D array with at least two rows")
    weights = _normalize_sample_weight(sample_weight, len(values))
    if weights is None:
        weights = np.full(len(values), 1.0 / len(values))
    low = values.min(axis=0) - 0.01 * np.abs(values.min(axis=0))
    high = values.max(axis=0) + 0.01 * np.abs(values.max(axis=0))
    volume = np.prod(high - low)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("data must define a positive finite integration box")

    log_grid_weights = None
    if (integration_points is None) != (proposal_log_density is None):
        raise ValueError("Supply integration_points and proposal_log_density together")
    if integration_points is None:
        Z, Z_grid, volume = get_F_and_Random_Samples(pd.DataFrame(values), degree, n_grid)
        integration_count = int(n_grid)
    else:
        points = np.asarray(integration_points, dtype=np.float64)
        log_q = np.asarray(proposal_log_density, dtype=np.float64)
        if (points.ndim != 2 or not len(points) or points.shape[1] != values.shape[1]
                or not np.isfinite(points).all()):
            raise ValueError("integration_points must be finite rows matching the data columns")
        if log_q.shape != (len(points),) or not np.isfinite(log_q).all():
            raise ValueError("proposal_log_density must be finite with one value per draw")
        inside = np.all((points >= low) & (points <= high), axis=1)
        if not inside.any():
            raise ValueError("No integration_points lie inside the fitting box")
        integration_count = len(points)
        log_grid_weights = -log_q[inside] - np.log(integration_count)
        Z = Features(pd.DataFrame(values), degree)
        Z_grid = Features(pd.DataFrame(points[inside]), degree)
    Z = Z.to_numpy(dtype=np.float64)
    Z_grid = Z_grid.to_numpy(dtype=np.float64)
    if not np.isfinite(Z).all() or not np.isfinite(Z_grid).all():
        raise ValueError("Polynomial features are not finite; rescale the input data")
    sos_basis = _fmle_sos_basis(Z[:, 1:], weights, basis, covariance_regularization)
    # Evaluate centered features directly rather than subtracting two large
    # quantities through the constant row of the augmented basis.
    feature_mean = weights @ Z[:, 1:]
    data_squares = ((Z[:, 1:] - feature_mean) @ sos_basis[1:]) ** 2
    grid_squares = ((Z_grid[:, 1:] - feature_mean) @ sos_basis[1:]) ** 2
    if not np.isfinite(data_squares).all() or not np.isfinite(grid_squares).all():
        raise ValueError("Squared polynomials are not finite; rescale the input data")
    inside_count = len(grid_squares)
    objective = _PrimalObjective(data_squares, grid_squares, volume, weights,
                                 log_grid_weights=log_grid_weights)
    del Z, Z_grid, data_squares, grid_squares
    initial = np.ones(sos_basis.shape[1])
    initial_fun, initial_jac = objective(initial)
    initial_projected = np.where(initial_jac > 0,
                                 np.minimum(initial_jac, initial - min_coefficient),
                                 initial_jac)
    if np.linalg.norm(initial_projected, ord=np.inf) <= min_gradient_norm:
        result = OptimizeResult(x=initial, success=True, status=0, nit=0,
                                message="Initial coefficients meet projected-gradient tolerance.")
    elif max_iterations == 0:
        result = OptimizeResult(x=initial, success=False, status=1, nit=0,
                                message="Iteration limit reached at initialization.")
    else:
        result = minimize(
            objective, initial, method="L-BFGS-B", jac=True,
            bounds=[(min_coefficient, None)] * len(initial),
            options={"maxiter": int(max_iterations), "gtol": min_gradient_norm,
                     "ftol": 0.0, "maxls": 40},
        )
    # Recompute every diagnostic at the coefficients actually returned.
    result.fun, result.jac = objective(result.x)
    result.coefficients = result.x.copy()
    # Equivalent to x - projection(x - jac), without subtracting nearby large
    # coefficients when the unconstrained derivative is small.
    result.projected_gradient = np.where(
        result.jac > 0, np.minimum(result.jac, result.x - min_coefficient), result.jac)
    result.projected_gradient_norm = float(np.linalg.norm(result.projected_gradient, ord=np.inf))
    result.gradient_tolerance_met = bool(result.projected_gradient_norm <= min_gradient_norm)
    result.basis = sos_basis.copy()
    result.initial_B = sos_basis @ sos_basis.T
    result.initial_fun = initial_fun
    result.log_partition = objective.log_partition
    result.integration_ess = objective.integration_ess
    result.integration_bounds = np.stack((low, high))
    result.integration_method = "uniform" if integration_points is None else "importance"
    result.integration_sample_count = integration_count
    result.integration_in_bounds_count = inside_count
    result.effective_sample_size = float(1.0 / (weights @ weights))
    result.nfev = objective.evaluations
    result.njev = objective.evaluations
    B = (sos_basis * result.x) @ sos_basis.T
    B = (B + B.T) / 2
    if verbosity:
        print(f"Primal Sliced Normal: {len(initial)} coefficients, {result.nit} iterations.")
        print(result.message)
        print(f"Final projected gradient norm: {result.projected_gradient_norm:.8g}")
        print(f"Requested gradient tolerance met: {result.gradient_tolerance_met}")
    return (B, result) if return_info else B


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
