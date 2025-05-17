import autograd.numpy as np
import pandas as pd
from itertools import product

def Features(data, d):
    cols = data.columns
    coeffs = [i for i in product(range(d+1), repeat=len(cols)) if sum(i) <= d]
    features = {}
    for coef in coeffs:
        col_value = np.ones(data.shape[0])
        for i, power in enumerate(coef):
            col_value *= data[cols[i]] ** power
        features[str(coef)] = col_value
    return pd.DataFrame(features)

def get_F_and_Random_Samples(data, d, n_grid):
    F = Features(data, d)
    low = np.array(data.min()) - 0.01 * np.abs(data.min())
    high = np.array(data.max()) + 0.01 * np.abs(data.max())
    volume = np.prod(high - low)
    sample = np.random.uniform(low, high, size=(n_grid, len(low)))
    feature_sample = Features(pd.DataFrame(sample), d)
    return F, feature_sample, volume
