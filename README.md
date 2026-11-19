# Sliced Normal

Model dependent variables using a polynomial feature vector `z(x)` containing
all monomials of total degree at most `d`, including 1. The fitted density is
`f_B(x) = exp(-z(x).T @ B @ z(x)) / c(B)`, with positive semidefinite B.
For three variables and degree 2 there are 10 features and a quartic energy.

This branch uses fast fitting in the existing `sliced_normals` package. It caches
empirical moments `S = Z.T @ Z / n`, evaluates their contribution as `<B,S>`,
and shares integration energies between cost and analytic-gradient calculations.
The full feature set/order, FMLE initialization, integration box/grid rule,
SPD manifold, steepest-descent optimizer, default partition floor, and penalty
are retained. Likelihood evaluation and MCMC sampling use the original methods.

## Install

Use Python 3.11 (supported: 3.10–3.12). From the repository root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e sliced_normals
python -m ipykernel install --user --name sliced-normal --display-name 'Python (Sliced Normal)'
```

On Windows, activate `.venv\Scripts\Activate.ps1` in PowerShell. For notebooks,
select **Python (Sliced Normal)** as the kernel. The demo includes a setup cell
that creates this environment and installs dependencies automatically.

## Usage

```python
import numpy as np
import pandas as pd
from sliced_normals import estimate_optimal_B, run_mcmc_sampler

data = pd.read_csv('sliced_normals/Sample_Data.csv')[['X_1', 'X_2', 'X_3']]
np.random.seed(42)
B = estimate_optimal_B(data, degree=2, n_grid=5000, max_iterations=2000)
samples = run_mcmc_sampler(data, B, d=2, num_samples=2000, burnin=300, thin=5)
```

[Demo/Sliced_Normal_Usage.ipynb](Demo/Sliced_Normal_Usage.ipynb) contains the
complete FMLE, fast fitting, approximate likelihood, sampling, and plotting
workflow. Run from a checkout; it has no cloning/deletion commands or fixed
Colab paths. No separate fast package or project configuration is required.

## Numerical notes

- Full training/grid features remain in memory and feature count grows
  combinatorially. FMLE covariance inversion can fail for insufficient or
  poorly conditioned data.
- Pymanopt 2.2.1 reports a pre-step gradient while returning a post-step matrix.
  The demo recomputes the gradient at the returned matrix. A small manifold
  gradient is not a certificate of global PSD optimality.
- The likelihood helper returns an approximate mean log likelihood. Its support
  is derived from its input data; the demo uses the same data/support for both
  models and fresh integration points, not held-out validation.
- Fitting integrates over a bounded box, while the retained MCMC sampler is
  unbounded. The demo's sample plots are exploratory; sampler convergence and
  support consistency require separate investigation.
- The partition floor changes the objective when active. The alternate helper
  `estimate_optimal_B_with_grad` uses the same cached implementation without
  the floor, and its normalized gradient has no extra division by volume.

## Reference

Luis G. Crespo, Brendon K. Colbert, Sean P. Kenny, Daniel P. Giesy,
*On the quantification of aleatory and epistemic uncertainty using Sliced-Normal
distributions*, Systems & Control Letters.
