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

def _decode(flag):
    """
    1 - SLS-obtained value
    2 - SW-obtained value
    4 - CACF-obtained value
    8 - GP-obtained value
    16 - Only two out of three estimates agreed
    32 - No robust matches, CACF assumed best
    64 - No ACF measured
    128 - ACF does not match 'best' period within 2 sigma
    256 - ACF indicates that 'best' period is a potential harmonic
    512 - One or more sectors disagree strongly across all estimates
    """

    STRINGS = {
        1 : "1: Best rotation is from the Simple Lomb Scargle (SLS) method.",
        2 : "2: Best rotation is from the Simple Wavelet (SW) method.",
        4 : "4: Best rotation is from the Composite Autocorrelation Function (CACF) method.",
        8 : "8: Best rotation is from the Gaussian Process (GP) method.",
        16 : "16: Only 2 of the 3 estimates of rotation agreed with one another " +
             "to within 1 sigma.",
        32 : "32: None of the 3 estimates agreed with one another to within 1 "+
             "sigma. The CACF estimate is assumed to be the best in this case.",
        64 : "64: No ACF period could be reliably measured (indicating low power or long periods).",
        128 : "128: The ACF period does not match the 'best' period within 2 sigma.",
        256 : "256: The ACF period is potentially a harmonic of the 'best' period (or vice versa!)",
        512: "512: One or more sectors disagrees strongly across all estimates with " +
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
