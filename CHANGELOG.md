# Changes

## 0.2.0

- Make cached empirical moments and analytic-gradient fitting the default
  `sliced_normals.estimate_optimal_B` implementation.
- Retain the prior fitter under `sliced_normals.legacy_optimization` for comparisons.
- Preserve full features, initialization, solver, integration rule/count, floor,
  penalty, sampling, and likelihood behavior.
- Add root-level installable packaging and bundle the sample CSV in the package.
- Provide complete usage and convergence notebooks, a command-line demo,
  numerical tests, and CI checks.
- Document observed speedups and existing stopping, conditioning, support,
  normalization, and memory limitations.
