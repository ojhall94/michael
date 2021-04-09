"""
The `methods` script contains functions for estimating the period of a star.
"""

import lightkurve as lk
import astropy.units as u
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

from .utils import _gaussian_fn, _safety

def simple_astropy_lombscargle(j, sector='all'):
    """
    Following the criteria in Feinstein+2020:
        1) Period must be less than 12 days
            - Set maximum LombScargle period to 12 days
        2) The FWHM of the Gaussian fit to the peak power must be < 40% peak period
        3) Secondary peak must be 10% weaker than the primary peak

    We calculate a periodogram using Lightkurve. We set the maximum period to 12 (to comply with
    condition (1). We set the normalization to 'psd' to reduce the dynamic range, making the
    data easier to fit. We set an oversample factor of 100 to improve the smoothness of the fit.

    We fit using Scipy.Optimize.curve_fit. This is a bit of a hack-- really a PSD like this should
    be fit using a Gamma function. However for a clear signal such as this that would be overkill.
    Curve_fit will do the job perfectly well.

    Parameters
    ----------

    j: class
        The `janet` class containing the metadata on our star.

    sector: int
        The sector for which to calculate the simple astropy lombscargle period.
        If 'all', calculates for all sectors stitched together.

    """
    if j.verbose:
        print(f'### Running Simple Astropy Lomb-Scargle on Sector {sector} on star {j.gaiaid} ###')

    # Call the relevant light curve
    clc = j.void[f'clc_{sector}']

    pg = clc.to_periodogram(maximum_period = 12., normalization='psd', oversample_factor=100,
                                freq_unit = 1/u.day)

    # Select the region around the highest peak
    max_period = pg.period_at_max_power.value
    max_power = pg.max_power.value
    s = (pg.period.value > 0.6*max_period) & (pg.period.value < 1.4*max_period)
    p = pg[s].period.value
    P = pg[s].power.value

    # Store the periodogram for plotting
    j.void[f'pg_{sector}'] = pg
    j.void[f'p_{sector}'] = p
    j.void[f'P_{sector}'] = P

    # Fit a Gaussian
    ## Params are mu, sigma, Amplitude
    popt, pcov = curve_fit(_gaussian_fn, p, P, p0 = [max_period, 0.1*max_period, max_power],
                            bounds = ([0.8*max_period, 0., 0.9*max_power],[1.2*max_period, 0.25*max_period, 1.1*max_power]))

    j.results.loc[sector, 'SLS'] = popt[0]
    j.results.loc[sector, 'e_SLS'] = popt[1]
    j.results.loc[sector, 'f_SLS'] = 0

    # Perform quality checks
    ## Condition (2)
    if popt[1]*2.355 > 0.4*popt[0]:
        j.results.loc[sector, 'f_SLS'] += 2

    ## Condition (3)
    peaks, _ = find_peaks(pg.power.value, height = 0.9*max_power)
    if len(peaks) > 1:
        j.results.loc[sector, 'f_SLS'] += 3

        # Double check if the presence of a second peak has upset the fits
        # If so, repeat the fit in a smaller range
        peaks, _ = find_peaks(P, height=0.9*max_power)
        if len(peaks) > 1:
            s = (pg.period.value > 0.8*max_period) & (pg.period.value < 1.2*max_period)
            popt, pcov = curve_fit(_gaussian_fn, pg[s].period.value, pg[s].power.value,
                                    p0 = [max_period, 0.2*max_period, max_power],
                                    bounds = ([0.8*max_period, 0., 0.9*max_power],[1.2*max_period, 0.25*max_period, 1.1*max_power]))

            j.results.loc[sector, 'SLS'] = popt[0]
            j.results.loc[sector, 'e_SLS'] = popt[1]

    # Save the gaussian fit
    j.void[f'popt_{sector}'] = popt


    if j.verbose:
        print(f'### Completed Simple Astropy Lomb-Scargle for Sector {sector} on star {j.gaiaid} ###')

    _safety(j)

def simple_wavelet(j, sector='all'):
    return 0
