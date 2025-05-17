import autograd.numpy as np
from autograd.scipy.special import logsumexp
from pymanopt import Problem
from pymanopt.manifolds import SymmetricPositiveDefinite
from pymanopt.optimizers import SteepestDescent
import pymanopt
from .features import get_F_and_Random_Samples
from .fmle import FMLE

def estimate_optimal_B(data, degree=2, n_grid=10000, verbosity=2,
                       max_iterations=1000, min_gradient_norm=1e-6,
                       penalty_lambda=0.0, penalize_excess_only=False):
    Z_df, Z_grid_df, volume = get_F_and_Random_Samples(data, degree, n_grid)
    Z = np.asarray(Z_df.values, dtype=np.float64)
    Z_grid = np.asarray(Z_grid_df.values, dtype=np.float64)
    B_fmle = FMLE(Z_df.iloc[:, 1:].to_numpy(dtype=float))
    frobenius_fmle = np.sqrt(np.sum(B_fmle ** 2))
    manifold = SymmetricPositiveDefinite(Z.shape[1])

    @pymanopt.function.autograd(manifold)
    def cost(B):
        term1 = np.mean(np.einsum("ij,jk,ik->i", Z, B, Z))
        log_integral = logsumexp(-np.einsum("ij,jk,ik->i", Z_grid, B, Z_grid)) - np.log(len(Z_grid)) + np.log(volume)
        penalty = penalty_lambda * max(np.sqrt(np.sum(B**2)) - frobenius_fmle, 0) if penalize_excess_only else penalty_lambda * np.sqrt(np.sum(B**2))
        return term1 + log_integral + penalty

    problem = Problem(manifold=manifold, cost=cost)
    optimizer = SteepestDescent(verbosity=verbosity, max_iterations=max_iterations, min_gradient_norm=min_gradient_norm)
    result = optimizer.run(problem, initial_point=B_fmle)
    return result.point
