import numpy as np
from hyppo.independence import Hsic
from joblib import Parallel, delayed
from itertools import combinations

def compute_hsic_matrix(data, use_pval=False, reps=1, n_jobs=-1, verbose=False):
    """
    Compute the pairwise HSIC dependence matrix for columns in a DataFrame.

    Parameters:
    - data: pd.DataFrame (n_samples x n_features)
    - use_pval: bool, if True compute p-values, else use test statistic
    - reps: int, number of permutations for p-value estimation
    - n_jobs: int, number of parallel jobs (-1 for all cores)
    - verbose: bool, if True, print progress

    Returns:
    - matrix: np.ndarray of shape (n_features, n_features)
    - colnames: list of column names
    """
    colnames = data.columns.tolist()
    n_cols = data.shape[1]

    def test_pair(i, j):
        hsic = Hsic()
        x = data.iloc[:, i].values.reshape(-1, 1)
        y = data.iloc[:, j].values.reshape(-1, 1)
        if use_pval:
            _, pval = hsic.test(x, y, reps=reps)
            return (i, j, pval)
        else:
            stat = hsic.statistic(x, y)
            return (i, j, stat)

    if verbose:
        print(f"Computing HSIC {'p-values' if use_pval else 'statistics'}...")

    results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
        delayed(test_pair)(i, j) for i, j in combinations(range(n_cols), 2)
    )

    matrix = np.zeros((n_cols, n_cols))
    for i, j, val in results:
        matrix[i, j] = matrix[j, i] = val
    np.fill_diagonal(matrix, 0)

    return matrix, colnames

import heapq

def group_columns_by_dependence(matrix, colnames, use_pval=False, max_group_size=5):
    """
    Cluster columns based on dependence matrix with group size constraint.

    Parameters:
    - matrix: np.ndarray of shape (n_features, n_features), HSIC stats or p-values
    - colnames: list of column names
    - use_pval: bool, if True, lower values mean stronger dependence
    - max_group_size: int, maximum number of columns per group

    Returns:
    - groups: list of lists of column names (each group)
    """
    n_cols = len(colnames)
    clusters = {i: [i] for i in range(n_cols)}
    cluster_ids = list(range(n_cols))

    # Prepare heap: negative stat = max heap, or pval = min heap
    heap = []
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            val = matrix[i, j]
            score = val if use_pval else -val
            heap.append((score, i, j))
    heapq.heapify(heap)

    while heap and len(set(cluster_ids)) > 1:
        score, i, j = heapq.heappop(heap)

        # If using p-values, discard weak dependencies
        if use_pval and score > 0.05:
            break

        # Find cluster indices
        ci = next(k for k, v in clusters.items() if i in v)
        cj = next(k for k, v in clusters.items() if j in v)
        if ci == cj:
            continue
        if len(clusters[ci]) + len(clusters[cj]) > max_group_size:
            continue
        # Merge
        clusters[ci].extend(clusters[cj])
        del clusters[cj]
        for idx in clusters[ci]:
            cluster_ids[idx] = ci

    final_groups = [
        [colnames[i] for i in sorted(group)]
        for group in clusters.values()
    ]
    return final_groups

