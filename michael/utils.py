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
    janet.results.to_csv(f'{janet.output_path}/{janet.gaiaid}/{janet.pl}_results.csv')

def _decode(flag):
    """
    1 - No robust agreement, highest SNR assumed best
    2 - Empty
    4 - Only two out of three estimates agreed
    8 - One or more sectors disagree strongly across all estimates
    16 - P2P check isn't cleared for best overall target.
    """

    STRINGS = {
        1 : "1: None of the 3 estimates agreed with one another to within 1 "+
             "sigma. The estimate with highest SNR is assumed to be the best in"+
             " this case.",
        4 : "4: Only 2 of the 3 estimates of rotation agreed with one another " +
             "to within 1 sigma.",
        8 :  "8: One or more sectors disagrees strongly across all estimates with " +
            "the others. This may indicate signal from a background star present " +
            "in those sectors.",
        16: "16: The best overal value does not have a peak-to-peak height that" +
            " exceeds the mean absolute deviation of the detrended light curve" +
            ". Proceed with caution."
    }

    val = np.copy(flag)
    message = ''
    keys = np.flip(2**np.arange(10))

    for key in keys:
        if val >= key:
            message += STRINGS[key] + '\n'
            val -= key

    return message

def longest_sectors(j):
    if len(j.sectors) == 1:
        return j.sectors

    diffs = np.zeros(len(j.sectors))
    for idx, s in enumerate(j.sectors):
        d = np.diff(np.array(s.split('-')).astype(int))
        if np.isfinite(d):
            diffs[idx] = d

    # Returns the longest sectors
    dmax = np.nanmax(diffs)
    sel = np.where(diffs == dmax)
    return list(np.array(j.sectors)[sel])
