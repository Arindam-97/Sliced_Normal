import autograd.numpy as np
from autograd.scipy.special import logsumexp
from pymanopt import Problem
from pymanopt.manifolds import SymmetricPositiveDefinite
from pymanopt.optimizers import SteepestDescent
import pymanopt
from .features import get_F_and_Random_Samples
from .fmle import FMLE
from pymanopt.function import numpy as pymanopt_numpy

def estimate_optimal_B(data, degree=2, n_grid=10000, verbosity=2,
                       max_iterations=1000, min_gradient_norm=1e-4,
                       penalty_lambda=0.0, winsor_threshold=1e-10):
    Z_df, Z_grid_df, volume = get_F_and_Random_Samples(data, degree, n_grid)
    Z = np.asarray(Z_df.values, dtype=np.float64)
    Z_grid = np.asarray(Z_grid_df.values, dtype=np.float64)
    B_fmle = FMLE(Z_df.iloc[:, 1:].to_numpy(dtype=float))
    frobenius_fmle = np.sqrt(np.sum(B_fmle ** 2))
    manifold = SymmetricPositiveDefinite(Z.shape[1])

    @pymanopt.function.autograd(manifold)
    def cost(B):
        # First term: empirical quadratic form
        term1 = np.mean(np.einsum("ij,jk,ik->i", Z, B, Z))

        # Second term: integral approximation via log-sum-exp
        quad_terms = np.einsum("ij,jk,ik->i", Z_grid, B, Z_grid)
        log_integral_raw = logsumexp(-quad_terms) - np.log(len(Z_grid)) + np.log(volume)
        log_integral = max(log_integral_raw, np.log(winsor_threshold))  # Winsorization

        # Third term: Frobenius norm penalty
        penalty = penalty_lambda * np.sqrt(np.sum(B**2))

        return term1 + log_integral + penalty

    problem = Problem(manifold=manifold, cost=cost)
    optimizer = SteepestDescent(verbosity=verbosity,
                                 max_iterations=max_iterations,
                                 min_gradient_norm=min_gradient_norm)
    result = optimizer.run(problem, initial_point=B_fmle)
    return result.point


def estimate_optimal_B_with_grad(data, degree=2, n_grid=10000, verbosity=2,
                       max_iterations=1000, min_gradient_norm=1e-6,
                       penalty_lambda=0.0):

    # Compute feature matrices and volume
    Z_df, Z_grid_df, volume = get_F_and_Random_Samples(data, degree, n_grid)
    Z = np.asarray(Z_df.values, dtype=np.float64)
    Z_grid = np.asarray(Z_grid_df.values, dtype=np.float64)

    n, feature_dim = Z.shape
    Z_fmle = Z_df.iloc[:, 1:].to_numpy(dtype=float)
    B_fmle = FMLE(Z_fmle)

    manifold = SymmetricPositiveDefinite(feature_dim)

    # Define cost and gradient as regular Python functions (not decorators)
    def cost(B):
        ZBZ = np.einsum("ij,jk,ik->i", Z, B, Z)
        term1 = np.mean(ZBZ)

        ZBZ_grid = np.einsum("ij,jk,ik->i", Z_grid, B, Z_grid)
        log_integral = logsumexp(-ZBZ_grid) - np.log(len(ZBZ_grid)) + np.log(volume)

        penalty = penalty_lambda * np.sqrt(np.sum(B**2)) if penalty_lambda > 0 else 0.0
        return term1 + log_integral + penalty

    def grad(B):
        term1_grad = (Z.T @ Z) / n

        ZBZ_grid = np.einsum("ij,jk,ik->i", Z_grid, B, Z_grid)
        weights = np.exp(-ZBZ_grid - np.max(-ZBZ_grid))
        weights = weights / np.sum(weights)
        weighted_Z = Z_grid * np.sqrt(weights[:, np.newaxis])
        term2_grad = -(weighted_Z.T @ weighted_Z) / volume

        penalty_grad = 0.0
        frob = np.sqrt(np.sum(B**2))
        if penalty_lambda > 0 and frob > 0:
            penalty_grad = penalty_lambda * B / frob

        return term1_grad + term2_grad + penalty_grad

    # Wrap with backend for Pymanopt
    cost_fn = pymanopt_numpy(manifold)(cost)
    grad_fn = pymanopt_numpy(manifold)(grad)

    # Set up problem
    problem = Problem(manifold=manifold, cost=cost_fn, euclidean_gradient=grad_fn)

    optimizer = SteepestDescent(
        verbosity=verbosity,
        max_iterations=max_iterations,
        min_gradient_norm=min_gradient_norm,
    )
    result = optimizer.run(problem, initial_point=B_fmle)

    return result.point



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
