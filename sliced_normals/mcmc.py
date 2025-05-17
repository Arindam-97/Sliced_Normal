import autograd.numpy as np
import pandas as pd
import emcee
from itertools import product

def precompute_feature_exponents(n_features, degree):
    return [exp for exp in product(range(degree + 1), repeat=n_features) if sum(exp) <= degree]

def evaluate_features(x, exponents):
    return np.array([np.prod(x ** exp) for exp in exponents])

def run_mcmc_sampler(data, B, d, num_samples=1000, burnin=200, thin=1, nwalkers=None):
    x0 = data.iloc[0].to_numpy()
    ndim = len(x0)
    if nwalkers is None:
        nwalkers = 2 * ndim
    initial_state = x0 + 1e-4 * np.random.randn(nwalkers, ndim)
    exponents = precompute_feature_exponents(ndim, d)

    def log_density(x):
        z = evaluate_features(x, exponents)
        return (-z @ B @ z).item()

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_density)
    sampler.run_mcmc(initial_state, num_samples, progress=True)
    samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)
    return pd.DataFrame(samples, columns=data.columns)
