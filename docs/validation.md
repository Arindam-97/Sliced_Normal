# Saved comparison evidence

Bundled 3D sample, n=1000, d=2, K=5000; Pymanopt 2.2.1; NumPy 1.26.4;
SciPy 1.13.1; pandas 1.5.3; autograd 1.8.0; emcee 3.1.6; Python 3.11.8;
macOS ARM64, one BLAS thread. Same data/grid seed/FM​LE start for each pair.
Limits: 10000 iterations, gradient threshold 5e-4, 120 seconds per fit;
zero penalty, partition threshold 1e-10. Each seed was one instrumented run,
not a distribution of repeated timing trials. All runs stopped on the library's
pre-step gradient check. No endpoint passed the independently recomputed
5e-4 tolerance; this affects both old and fast implementations.

| Seed | Original seconds | Fast seconds | Ratio |
| --- | ---: | ---: | ---: |
| 561 | 2.7468 | 0.8674 | 3.17x |
| 562 | 2.0531 | 0.6293 | 3.26x |
| 563 | 2.0114 | 0.8622 | 2.33x |

Final paired objective differences were at most 2.82e-5. Fresh normalization
batches (3 x 50000 points on the same training box) gave paired mean log
likelihood differences at most 4.29e-5 averaged per seed. The quadrature-weight
Hellinger distances were below 0.001. These are in-sample diagnostics and do
not establish held-out accuracy, global optimality, or integration accuracy.
The raw local notes/experimental notebooks remain outside this distribution.

The unit tests check cost/gradient against the legacy autograd objective, floor
branches/tie, finite differences, caching/mutation, constant shifts, input
validation, and a matched short fit. Production solver behavior is deliberately
retained in this release; numerical redesign is separate work.

## Release checks

The complete package installed successfully with the demo extra in a fresh
Python 3.11 environment; pip check found no broken dependencies. All eight
numerical tests passed in both the reference environment and a fresh environment
with pandas 2.3.3. Both notebooks' Python cells completed smoke runs with reduced
fit/sampling/integration budgets. The setup cell was skipped and the kernel-path
guard adapted only in memory for that validation environment. Source notebooks
remain unexecuted with full default budgets. The command-line demo also completed
fitting, likelihood evaluation, both MCMC chains, and plot/CSV/matrix export.
These smoke runs validate the workflow, not convergence under the reduced caps.
