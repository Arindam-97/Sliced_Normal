import autograd.numpy as np
import pandas as pd
from .features import get_F_and_Random_Samples

def FMLE(Z):
    n, d = Z.shape
    mu_hat = np.mean(Z, axis=0)
    Z_centered = Z - mu_hat
    Sigma_hat = (Z_centered.T @ Z_centered) / (n - 1)
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

def get_FMLE(data, d):
    Z_df, _, _ = get_F_and_Random_Samples(data, d, n_grid=1000)
    Z = Z_df.iloc[:, 1:].to_numpy(dtype=float)
    return FMLE(Z)
