import autograd.numpy as np
import pandas as pd
from .features import get_F_and_Random_Samples, _normalize_sample_weight

def FMLE(Z, *, sample_weight=None):
    """Feature-space Gaussian initializer, with optional observation weights.

    Explicit weights use normalized weighted ML mean/covariance. Omitting
    weights preserves the historical n-1 covariance. Uniform explicit weights
    therefore have the ML n denominator, unlike the legacy unweighted path.
    Z excludes the constant feature; the returned B restores that feature.
    """
    n, d = Z.shape
    weights = _normalize_sample_weight(sample_weight, n)
    if weights is None:
        mu_hat = np.mean(Z, axis=0)
        Z_centered = Z - mu_hat
        Sigma_hat = (Z_centered.T @ Z_centered) / (n - 1)
    else:
        Z = np.asarray(Z, dtype=float)
        if not np.isfinite(Z).all():
            raise ValueError("Features must be finite")
        mu_hat = weights @ Z
        Z_centered = Z - mu_hat
        Sigma_hat = Z_centered.T @ (weights[:, None] * Z_centered)
    P_hat = np.linalg.inv(Sigma_hat)
    top_right = -0.5 * mu_hat.T @ P_hat
    bottom_left = top_right.reshape(-1, 1)
    bottom_right = 0.5 * P_hat
    schur_term = top_right @ np.linalg.solve(bottom_right, top_right)
    top_left = np.array([[schur_term + 1e-8]])
    return np.block([
        [top_left, top_right.reshape(1, -1)],
        [bottom_left, bottom_right]
    ])

def get_FMLE(data, d, *, sample_weight=None):
    """Compute the FMLE B from data, optionally using positional sample weights."""
    _normalize_sample_weight(sample_weight, len(data))
    Z_df, _, _ = get_F_and_Random_Samples(data, d, n_grid=1000)
    Z = Z_df.iloc[:, 1:].to_numpy(dtype=float)
    return FMLE(Z, sample_weight=sample_weight)
