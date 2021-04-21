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
    2 - WS-obtained value
    4 - ACF-obtained value
    8 - GP-obtained value
    16 - Validation done using a SLS value that wasn't 'best'
    32 - No robust matches
    34 - No robust matches, WS-obtained value (ditto for other combos)
    64 - No ACF measured
    128 - ACF does not match 'best' period within 2 sigma
    256 - ACF indicates that 'best' period is a potential harmonic
    """

    STRINGS = {
        1 : "1: Best rotation is from the Simple Lomb Scargle (SLS) method.",
        2 : "2: Best rotation is from the Simple Wavelet (SW) method.",
        4 : "4: Best rotation is from the Simple Autocorrelation Function (ACF) method.",
        8 : "8: Best rotation is from the Gaussian Process (GP) method.",
        16 : "16: Valididation between WS and SLS was done using a SLS period that was not " +
            "the 'best' SLS value (lowest uncertainty without flags), because there was no 2" +
            " sigma agreement with the SW period.",
        32 : "32: No robust matches were found between the SW period and any unflagged SLS periods.",
        64 : "64: No ACF period could be reliably measured (indicating low power or long periods).",
        128 : "128: The ACF period does not match the 'best' period within 2 sigma.",
        256 : "256: The ACF period is potentially a harmonic of the 'best' period (or vice versa!)",
    }

    val = np.copy(flag)
    message = ''
    keys = np.flip(2**np.arange(9))

    for key in keys:
        if val >= key:
            message += STRINGS[key] + '\n'
            val -= key

    return message
