"""
Various utility functions
"""

import numpy as np

def _gaussian_fn(p, mu, sigma, A):
    """
    Returns: A * exp(-(p - mu)^2 / (2.0 * sigma^2))
    """
    return A * np.exp(-(p - mu)**2 / (2.0 * sigma**2))

def _safety(janet):
    janet.results.to_csv(f'{janet.output_path}/{janet.gaiaid}/results.csv')
    if janet.verbose:
        print('### Saved results ###')
