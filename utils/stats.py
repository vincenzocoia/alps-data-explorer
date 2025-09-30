"""
Statistical utility functions for Alps Data Explorer.
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from typing import Union, Tuple


# Kendall's tau is now handled directly by scipy.stats.kendalltau
# No custom wrapper needed - scipy handles NaN values appropriately


def nscore(x: Union[np.ndarray, list], a: float = -0.5) -> np.ndarray:
    """
    Convert data to normal scores using rank-based uniformization.
    
    This transformation converts any data to follow a normal distribution
    by ranking the data and applying the inverse normal CDF. Useful for
    investigating dependence between two variables.
    
    Parameters
    ----------
    x : array-like
        Input data to transform.
    a : float, default=-0.5
        Parameter for the uniformization; should be -1 < a < 0.
        Common values: -0.5 (default), -0.375 (Blom), -1/3 (Tukey)
        
    Returns
    -------
    np.ndarray
        Array of normal scores with same shape as input.
    Notes
    -----
    The transformation follows these steps:
    1. Rank the non-NaN values (ties get average rank).
    2. Convert ranks to uniform scores: (rank + a) / (n + 1 + 2*a).
    3. Apply inverse normal CDF to get normal scores.
    
    Examples
    --------
    >>> data = np.array([1, 5, 3, 9, 2])
    >>> normal_scores = nscore(data)
    >>> np.abs(np.mean(normal_scores)) < 0.1  # Should be close to 0
    True
    >>> np.abs(np.std(normal_scores) - 1) < 0.2  # Should be close to 1
    True
    
    >>> data_with_nan = np.array([1, np.nan, 3, 9, 2])
    >>> scores = nscore(data_with_nan)
    >>> np.isnan(scores[1])  # NaN preserved
    True
    """
    x = np.asarray(x, dtype=float)
    mask = ~np.isnan(x)
    
    # If no valid data, return array of NaNs
    if mask.sum() == 0:
        return np.full_like(x, np.nan, dtype=float)
    
    # Rank non-NaN values (1 to n), with ties averaged using scipy
    ranks = stats.rankdata(x[mask], method='average')
    n = len(ranks)
    
    # Convert ranks to uniform scores in (0,1)
    # Using (rank + a) / (n + 1 + 2*a) formula
    u = (ranks + a) / (n + 1 + 2 * a)
    
    # Avoid exactly 0 or 1 to prevent +/-inf from inverse normal CDF
    eps = np.finfo(float).eps
    u = np.clip(u, eps, 1.0 - eps)
    
    # Apply inverse normal CDF to get normal scores
    z = norm.ppf(u)
    
    # Create output array with NaNs preserved
    out = np.full_like(x, np.nan, dtype=float)
    out[mask] = z
    
    return out
