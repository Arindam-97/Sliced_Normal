# Sliced Normal

Model dependent variables using a polynomial feature vector `z(x)` containing
all monomials of total degree at most `d`, including 1. The fitted density is
`f_B(x) = exp(-z(x).T @ B @ z(x)) / c(B)`, with positive semidefinite B.
For three variables and degree 2 there are 10 features and a quartic energy.

The default `sliced_normals` implementation caches empirical moments
`S = Z.T @ Z / n` (or `S = Z.T @ (w[:, None] * Z)` for normalized sample
weights), evaluates their contribution as `<B,S>`, and shares
integration energies between cost and analytic-gradient calculations.
The full feature set/order, FMLE initialization, integration box/grid rule,
SPD manifold, steepest-descent optimizer, default partition floor, and penalty
are retained. Likelihood evaluation and MCMC sampling use the original methods.

An optional primal SN fitter fixes the squared polynomials obtained from an
FMLE decomposition and optimizes only their positive coefficients. It supports
sample weights and returns the same kind of `B` matrix as the matrix fitter.

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
complete FMLE, density fitting, approximate likelihood, sampling, and plotting
workflow. The notebook runs from the repository checkout using relative paths.
Its separate **Primal SN: FMLE sum-of-squares coefficient fitting** section
shows coefficient-only fitting, diagnostics, fresh normalization checks, and
sample plots against its matched FMLE initialization.
The final **Weighted fitting and comparison** section demonstrates using the
same sample weights for FMLE, both fitters, and likelihood evaluation.

## Sample weights

Pass `sample_weight=w` to `get_FMLE`, `FMLE`, `estimate_optimal_B`,
`estimate_primal_B`, or `evaluate_true_log_likelihood`. The alternate
`estimate_optimal_B_with_grad` and `CachedObjective` also accept it.

```python
from sliced_normals import get_FMLE, estimate_primal_B, evaluate_true_log_likelihood

# data: one observation per row; w: matching sample weights in the same row order
B_initial = get_FMLE(data, d=2, sample_weight=w)
B, info = estimate_primal_B(data, degree=2, sample_weight=w, return_info=True)
mean_log_likelihood = evaluate_true_log_likelihood(
    data, B, d=2, n_grid=50000, sample_weight=w)

# The full matrix fitter supports the same weights:
# B_full = estimate_optimal_B(data, degree=2, sample_weight=w)
```

Weights must be a one-dimensional array with one finite nonnegative value per
row and positive total. They are normalized internally, so scaling all weights
by the same positive constant changes neither the statistical objective nor
the initialization. Weights are positional, including when passed as a pandas
Series; labels are not aligned automatically.

The weighted mean log likelihood is `sum(w_normalized * log_density)`. The
normalization integral is unchanged: sample weights apply to observations,
not integration points. Zero-weight rows have no empirical contribution but
still participate in defining the integration box. Use the same data rows,
weights, and fresh integration points when comparing models.

For backward compatibility, omitting weights preserves the legacy full-matrix
FMLE covariance with denominator `n-1`. Explicit weights use weighted ML
covariance; uniform explicit weights therefore use denominator `n`. The primal
initializer always uses weighted ML moments and feature-scaled regularization.
Weighted fitting does not require resampling the observations. After fitting,
the MCMC sampler draws from the resulting density and takes no sample weights.

## Coefficient-only fitting (primal SN)

```python
from sliced_normals import estimate_primal_B

np.random.seed(42)
B, info = estimate_primal_B(
    data, degree=2, n_grid=5000, sample_weight=None,
    basis='eigen', max_iterations=2000, min_gradient_norm=1e-6,
    return_info=True,
)
print(info.coefficients)
print(info.initial_fun, info.fun)  # Same-grid initial/final negative log likelihood
print(info.message)
print(info.projected_gradient_norm, info.gradient_tolerance_met)
```

Let `h(x) = z(x) @ info.basis`. The fitted energy is
`sum(info.coefficients * h(x)**2)` and
`B = (info.basis * info.coefficients) @ info.basis.T`. All coefficients start
at one. Only nonconstant feature directions are included; an additive constant
in the energy does not affect the normalized density. Eight variables at degree
2 require 44 coefficients, compared with 1,035 raw symmetric matrix entries.

`sample_weight` accepts one finite nonnegative weight per observation, with
positive total; weights are normalized internally. They determine both the
FMLE basis and the weighted likelihood. `None` gives equal weights. Integration
bounds still use all input observations, including zero-weight rows.

The basis uses weighted ML covariance moments and a default `1e-8` ridge after
feature scaling. This differs slightly from the legacy FMLE's `n-1` covariance.
`basis='cholesky'` is also supported. The decompositions have the same initial
energy but permit different coefficient reweightings. `info.initial_B` is the
exact initializer used by this fitter.

SciPy L-BFGS-B optimizes the convex fixed-basis likelihood with an analytic
gradient; this path does not use Pymanopt or a partition floor. The default
coefficient lower bound is `1e-6`. Grid squares and weighted data moments are
cached, giving O(n_grid * number_of_coefficients) work per cost/gradient call.
No matrix decomposition is performed inside the coefficient optimization.

`info.gradient_tolerance_met` independently checks the projected gradient at
the returned coefficients. The projection accounts for coefficients at their
lower bound, where a nonzero raw gradient can be correct. A solver success
message or an iteration limit alone is not proof of this tolerance. With
`max_iterations=0`, the fitter returns its initialization without taking a step.

This fits the best coefficients in the fixed FMLE basis, rather than the whole
SN family. It does not rotate or recenter that basis, preserve subgroup
marginals, or remove the need for accurate multidimensional normalization.
`info.integration_ess` diagnoses concentration of numerical integration
weights; it is distinct from `info.effective_sample_size` for the input weights.
Independent integration checks are needed even when the coefficient solver
converges. The existing unbounded MCMC sampler can accept `B`, but still differs
from the bounded fitting support described below.

A sparse integration grid can make the numerical likelihood unbounded even
when the continuous model is well-defined. The primal fitter raises `ValueError`
if it detects a positive coefficient direction with weighted data energy below
every grid energy, which proves this failure. Increase the integration budget
or reduce degree before retrying. Passing this diagnostic does not establish
integration accuracy or rule out every possible unbounded direction. Narrow
high-dimensional data can require a better integration design than uniform
box sampling.

For a known proposal density, the primal fitter can use importance integration:

```python
B, info = estimate_primal_B(
    data, degree=2, integration_points=proposal_draws,
    proposal_log_density=log_q_at_draws, return_info=True,
)
```

Supply **all** independent proposal draws in data coordinates and their
normalized log density. The proposal must cover the fitting box. Outside-box
draws contribute zero while remaining in the Monte Carlo denominator; do not
filter them yourself without adjusting the proposal density. `n_grid` is
overridden by the supplied draw count. This changes numerical integration,
not the primal SN family. `info.integration_method`, `integration_sample_count`,
and `integration_in_bounds_count` identify the integration used. Fresh proposal
draws, integration ESS and batch variability should be checked independently.

## Numerical notes

- Full training/grid features remain in memory and feature count grows
  combinatorially. FMLE covariance inversion can fail for insufficient or
  poorly conditioned data.
- The fitter uses Pymanopt's SPD geometry and line search with a local correction
  to its stopping order: each stopping test uses the matrix that is returned.
  Time/iteration limits and tiny-step stops can still leave the gradient above
  tolerance. The demo independently checks the returned matrix. A small manifold
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
