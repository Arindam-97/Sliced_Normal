# Sliced Normal — fast fitting

A complete version of [Sliced_Normal](https://github.com/Arindam-97/Sliced_Normal)
with cached fitting as the default. The original modules, sample dataset, FMLE,
likelihood evaluation, MCMC sampler, and dependence-clustering module are retained.

For a monomial vector z(x) of total degree at most d, including 1, the model is
`f_B(x) = exp(-z(x).T @ B @ z(x)) / c(B)`, with positive semidefinite B.
The implementation fits positive definite B using Pymanopt steepest descent.
The normalizer is approximated on a fixed uniform Monte Carlo grid in a box
around the training observations.

## Install

Python 3.11 is recommended (supported: 3.10–3.12). From this repository's root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-reproducible.txt
python -m pip install -e '.[demo]'
python -m ipykernel install --user --name sliced-normal-fast --display-name 'Python (Sliced Normal Fast)'
```

On Windows use `.venv\Scripts\Activate.ps1` in PowerShell. Select **Python
(Sliced Normal Fast)** as the notebook kernel. A minimal library install is
`python -m pip install -e .`; optional clustering requires `'.[clustering]'`.

## Fit and sample

```python
from importlib.resources import files
import numpy as np
import pandas as pd
from sliced_normals import estimate_optimal_B, run_mcmc_sampler

data = pd.read_csv(files('sliced_normals').joinpath('Sample_Data.csv'))
data = data[['X_1', 'X_2', 'X_3']]
np.random.seed(42)
B = estimate_optimal_B(data, degree=2, n_grid=5000, max_iterations=2000)
samples = run_mcmc_sampler(data, B, d=2, num_samples=2000, burnin=300, thin=5)
```

Open [Demo/Sliced_Normal_Usage.ipynb](Demo/Sliced_Normal_Usage.ipynb) for the full
FMLE → fast fit → independent normalization check → sampling → plotting workflow.
It runs from a checkout without Colab paths, cloning, or deleting folders.
A command-line version is available as `python Demo/run_demo.py --output Demo/outputs`.
[Demo/Compare_Fitters.ipynb](Demo/Compare_Fitters.ipynb) compares convergence of
the retained legacy fitter and the fast default, including matched seeds, full
traces, endpoint checks, and independent normalization diagnostics.

## What changed

The public import remains `from sliced_normals import estimate_optimal_B`.
The default implementation now:

- Computes `S = Z.T @ Z / n` once and uses `<B,S>` for the empirical energy.
- Supplies the analytic derivative of the same clipped objective.
- Reuses grid energies and weights between cost/gradient calls at the same B.

All monomials/order, FMLE initialization, integration support/rule/count,
SPD manifold, steepest-descent settings, normalization floor, and Frobenius
penalty are preserved. Sampling and likelihood helpers are unchanged.
`from sliced_normals_fast import estimate_optimal_B` is a compatibility alias.
The prior default fitter is available under `sliced_normals.legacy_optimization`
for comparisons; its alternate gradient function is experimental and has a known
extra-volume-division defect. It is not used by the new default fitter.

Saved local comparisons on the bundled 1,000-row, degree-2 case with K=5000
showed 2.33–3.26x shorter instrumented fit times over three integration seeds,
with closely agreeing objectives and independently normalized densities.
These timings depend on hardware/settings and are not a scalability guarantee.
See [docs/validation.md](docs/validation.md) for settings and limitations.

## Validation

```sh
python -m unittest discover -s tests -v
python Demo/run_demo.py --iterations 15 --grid 300 --steps 50 --burnin 10 --thin 2 --evaluation-grid 1000 --output Demo/outputs
```

## Numerical and modeling limits

- Pymanopt 2.2.1 checks a pre-step gradient and returns a post-step matrix. Its
  reported gradient is not a verified gradient at the returned matrix. The demo
  separately recomputes the final gradient and reports whether tolerance is met.
- FMLE can be nearly singular; full feature covariance inversion needs enough
  observations and can fail for redundant or poorly scaled inputs.
- Training/grid features remain in memory. Feature count grows combinatorially.
- A fixed integration grid can give inaccurate normalization. The likelihood
  helper is approximate and reports a mean, despite its historical name.
- Fitting/likelihood use a bounded box; the retained MCMC sampler has unbounded
  support. Sample plots are exploratory and do not certify convergence or an
  exact draw from the box-normalized fitted distribution.
- The partition floor modifies the objective when active. Constant energy shifts
  cease to leave that clipped objective invariant when they activate its floor.
- The bundled sample has three variables; it is not identified as the paper's
  NASA flight dataset. Paper/Overleaf files are not included here.

## Reference

Luis G. Crespo, Brendon K. Colbert, Sean P. Kenny, Daniel P. Giesy,
*On the quantification of aleatory and epistemic uncertainty using Sliced-Normal
distributions*, Systems & Control Letters.
