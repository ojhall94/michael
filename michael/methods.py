"""
The `methods` script contains functions for estimating the period of a star.
"""

import lightkurve as lk
import astropy.units as u
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

import jazzhands

from .utils import _gaussian_fn, _safety

def simple_astropy_lombscargle(j, sector, period_range):
    """
    Following some criteria from Feinstein+2020 [1-3] and Nielsen+2013 [4]
        1) Period must be less than 12 days
            - Set maximum LombScargle period to 12 days
        2) The FWHM of the Gaussian fit to the peak power must be < 40% peak period
        3) Secondary peak must be 10% weaker than the primary peak
        4) Peak must be at least 4x above the time-series RMS noise, where

        $\sigma_{\textrm PS} = 4\sigma^2_{\textrm RMS} / N$

        where $N$ is the number of data points in the light curve.

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

    pg = clc.to_periodogram(minimum_period = period_range[0], maximum_period = period_range[1], normalization='psd', oversample_factor=100,
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

    ## Condition (4)
    sig_rms = np.sqrt(np.mean((clc.flux.value - 1)**2))
    sig_ps = 4 * sig_rms**2 / len(clc)
    if popt[2] < 4 * sig_ps:
        j.reuslts.loc[sector, 'f_SLS'] += 4

    # Save the gaussian fit
    j.void[f'popt_{sector}'] = popt

    if j.verbose:
        print(f'### Completed Simple Astropy Lomb-Scargle for Sector {sector} on star {j.gaiaid} ###')

    _safety(j)

def simple_wavelet(j, period_range):
    """
    We use the 'jazzhands' Python package to perform our wavelet analysis.
    The `jazzhands` package performs a wavelet analysis based on the procedures
    set out in Foster (1996) and Torrence & Compo (1998).

    #TO DO: needs an expansion of the WWZ, what does it mean?

    After the wavelet is calculated, we collapse it along the x-axis. We fit the
    largest peak in the resulting spectrum using Scipy.Optimize.curve_fit. The
    resulting mean and width of the Gaussian function approximating the peak is
    reported as the period and associated uncertainty.

    Parameters
    ----------

    j: class
        The `janet` class containing the metadata on our star.

    sector: int
        The sector for which to calculate the simple astropy lombscargle period.
        If 'all', calculates for all sectors stitched together.

    period_range: tuple
        The lower and upper limit on period range to search for a rotational
        signal. Default is (0.2, 12.) based on the McQuillan et al. (2014)
        search range and the limitations of TESS earthshine.

    """

    if j.verbose:
        print(f'### Running Wavelet Estimation on star {j.gaiaid} ###')

    # Call the relevant light curve
    clc = j.void[f'clc_all']

    t = clc.time.value
    f = clc.flux.value
    wt = jazzhands.WaveletTransformer(t, f)
    _, _, wwz, wwa = wt.auto_compute(nu_min = 1./period_range[1], nu_max = 1./period_range[0])

    j.void[f'wt'] = wt
    j.void[f'wwz'] = wwz
    j.void[f'wwa'] = wwa

    # Create data to fit
    w = np.sum(wwz, axis=1)
    w /= w.max() #Normalize
    p = 1/wt.nus

    max_w = np.max(w)
    max_p = p[np.argmax(w)]

    s = (p > 0.6*max_p) & (p < 1.4*max_p)
    w = w[s]
    p = p[s]

    # Fit a Gaussian
    ## Params are mu, sigma, Amplitude
    popt, pcov = curve_fit(_gaussian_fn, p, w, p0 = [max_p, 0.1*max_p, max_w],
                            bounds = ([0.8*max_p, 0., 0.9*max_w],[1.2*max_p, 0.25*max_p, 1.1*max_w]))

    j.results.loc['all', 'SW'] = popt[0]
    j.results.loc['all', 'e_SW'] = popt[1]

    # Save the gaussian fit
    j.void[f'wavelet_popt'] = popt

    if j.verbose:
        print(f'### Completed Wavelet Estimation on star {j.gaiaid} ###')

    _safety(j)

def simple_ACF(j, period_range):
    """


    Parameters
    ----------

    j: class
        The `janet` class containing the metadata on our star.

    sector: int
        The sector for which to calculate the simple astropy lombscargle period.
        If 'all', calculates for all sectors stitched together.

    period_range: tuple
        The lower and upper limit on period range to search for a rotational
        signal. Default is (0.2, 12.) based on the McQuillan et al. (2014)
        search range and the limitations of TESS earthshine.

    """

    if j.verbose:
        print(f'### Running ACF Estimation on star {j.gaiaid} ###')


    if j.verbose:
        print(f'### Completed ACF Estimation on star {j.gaiaid} ###')

    _safety(j)
