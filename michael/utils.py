"""
Various utility functions
"""

import numpy as np

_random_seed = 802

def _gaussian_fn(p, mu, sigma, A):
    """
    Returns: A * exp(-(p - mu)^2 / (2.0 * sigma^2))
    """
    return A * np.exp(-(p - mu)**2 / (2.0 * sigma**2))

def _safety(janet):
    janet.results.to_csv(f'{janet.output_path}/{janet.gaiaid}/results.csv')

def _decode(flag):
    """
    1 - No robust matches, CACF assumed best
    2 - No robust matches, SW assumed best
    4 - Only two out of three estimates agreed
    8 - One or more sectors disagree strongly across all estimates
    """

    STRINGS = {
        1 : "1: None of the 3 estimates agreed with one another to within 1 "+
             "sigma. The CACF estimate is assumed to be the best in this case, "+
             "if it is available.",
        2 : "1: None of the 3 estimates agreed with one another to within 1 "+
              "sigma. The SW estimate is assumed to be the best in this case, "+
              "if it is available.",
        4 : "128: Only 2 of the 3 estimates of rotation agreed with one another " +
             "to within 1 sigma.",
        8 :  "8: One or more sectors disagrees strongly across all estimates with " +
            "the others. This may indicate signal from a background star present " +
            "in those sectors.",
    }

    val = np.copy(flag)
    message = ''
    keys = np.flip(2**np.arange(10))

    for key in keys:
        if val >= key:
            message += STRINGS[key] + '\n'
            val -= key

    return message
