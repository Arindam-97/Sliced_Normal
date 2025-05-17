import autograd.numpy as np
from autograd.scipy.special import logsumexp
import pandas as pd
from .features import get_F_and_Random_Samples

def evaluate_true_log_likelihood(data, B, d, n_grid=1000):
    Z, Z_grid, vol = get_F_and_Random_Samples(pd.DataFrame(data), d, n_grid)
    quad_data = np.einsum("ij,jk,ik->i", Z, B, Z)
    quad_grid = np.einsum("ij,jk,ik->i", Z_grid, B, Z_grid)
    log_partition = logsumexp(-quad_grid) - np.log(n_grid) + np.log(vol)
    return np.mean(-quad_data - log_partition)
