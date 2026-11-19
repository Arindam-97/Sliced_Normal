"""Shared complete workflow for the notebook and command-line demo."""
from importlib.resources import files
import json
from pathlib import Path
import numpy as np
import pandas as pd
from pymanopt.manifolds import SymmetricPositiveDefinite
from sliced_normals import (get_FMLE, estimate_optimal_B, evaluate_true_log_likelihood,
                            run_mcmc_sampler, get_F_and_Random_Samples, CachedObjective)


def fit_demo(iterations=2000, grid=5000, evaluation_grid=50000, seed=42):
    data = pd.read_csv(files('sliced_normals').joinpath('Sample_Data.csv'))[['X_1','X_2','X_3']]
    np.random.seed(seed)
    B_fmle = get_FMLE(data, 2)
    np.random.seed(seed)
    B = estimate_optimal_B(data, degree=2, n_grid=grid, verbosity=1,
                           max_iterations=iterations, min_gradient_norm=5e-4)
    np.random.seed(seed)
    Z, G, volume = get_F_and_Random_Samples(data, 2, grid)
    objective = CachedObjective(Z.to_numpy(), G.to_numpy(), volume)
    manifold = SymmetricPositiveDefinite(len(B))
    gradient = manifold.euclidean_to_riemannian_gradient(B, objective.gradient(B))
    norm = float(manifold.norm(B, gradient))
    metrics = {'final_objective': objective.cost(B), 'final_gradient_norm': norm,
               'gradient_tolerance_met': norm < 5e-4,
               'minimum_eigenvalue': float(np.linalg.eigvalsh(B).min()),
               'degree': 2, 'fit_grid': grid, 'evaluation_grid': evaluation_grid,
               'seed': seed, 'iteration_cap': iterations}
    for name, matrix in [('fmle', B_fmle), ('fast', B)]:
        np.random.seed(seed+1000)
        metrics[name+'_approximate_mean_log_likelihood'] = float(
            evaluate_true_log_likelihood(data, matrix, 2, n_grid=evaluation_grid))
    return data, B_fmle, B, metrics


def sample_and_plot(data, B_fmle, B, output, steps=2000, burnin=300, thin=5, seed=42):
    import matplotlib.pyplot as plt
    from pandas.plotting import scatter_matrix
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = {'data': data}
    for offset, (name, matrix) in enumerate([('fmle', B_fmle), ('fast', B)]):
        np.random.seed(seed+2000+offset)
        frames[name] = run_mcmc_sampler(data, matrix, d=2, num_samples=steps,
                                        burnin=burnin, thin=thin)
        frames[name].to_csv(output / (name+'_samples.csv'), index=False)
    limits = {column: (min(frame[column].min() for frame in frames.values()),
                       max(frame[column].max() for frame in frames.values()))
              for column in data.columns}
    for name, frame in frames.items():
        axes = scatter_matrix(frame, figsize=(8,8), diagonal='hist', alpha=0.35, s=5)
        for i, y in enumerate(frame.columns):
            for j, x in enumerate(frame.columns):
                axes[i,j].set_xlim(limits[x])
                if i != j:
                    axes[i,j].set_ylim(limits[y])
        fig = axes[0,0].figure
        fig.suptitle(name+' (exploratory; see support and MCMC limitations)')
        fig.tight_layout()
        fig.savefig(output / (name+'_pairplot.png'), dpi=120)
        plt.close(fig)
    return frames
