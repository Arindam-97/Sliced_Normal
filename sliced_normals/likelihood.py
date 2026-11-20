import autograd.numpy as np
from autograd.scipy.special import logsumexp
import pandas as pd
from .features import get_F_and_Random_Samples, _normalize_sample_weight

def evaluate_true_log_likelihood(data, B, d, n_grid=1000, *, sample_weight=None):
    """Approximate (weighted) mean log likelihood on the data-derived box.

    Weights are normalized to sum to one and apply to observations only.
    Numerical integration is unchanged and uses all data rows to set support,
    including zero-weight rows. This returns a mean, not a total likelihood.
    """
    weights = _normalize_sample_weight(sample_weight, len(data))
    Z, Z_grid, vol = get_F_and_Random_Samples(pd.DataFrame(data), d, n_grid)
    quad_data = np.einsum("ij,jk,ik->i", Z, B, Z)
    quad_grid = np.einsum("ij,jk,ik->i", Z_grid, B, Z_grid)
    log_partition = logsumexp(-quad_grid) - np.log(n_grid) + np.log(vol)
    if weights is None:
        return np.mean(-quad_data - log_partition)
    return -weights @ quad_data - log_partition
