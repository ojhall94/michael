"""
The `methods` script contains functions for estimating the period of a star.
"""

import lightkurve as lk
import astropy.units as u
import numpy as np
from scipy.signal import find_peaks
from scipy import interpolate
from scipy.optimize import curve_fit
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
import warnings

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
    lolim = 0.8*max_period
    if lolim < period_range[0]:
        lolim = period_range[0]
    uplim = 1.2*max_period
    if uplim > period_range[1]:
        uplim = period_range[1]

    popt, pcov = curve_fit(_gaussian_fn, p, P, p0 = [max_period, 0.1*max_period, max_power],
                            bounds = ([lolim, 0., 0.9*max_power],[uplim, 0.25*max_period, 1.1*max_power]))

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
                                    bounds = ([lolim, 0., 0.9*max_power],[uplim, 0.25*max_period, 1.1*max_power]))

            j.results.loc[sector, 'SLS'] = popt[0]
            j.results.loc[sector, 'e_SLS'] = popt[1]

    ## Condition (4)
    sig_rms = np.sqrt(np.mean((clc.flux.value - 1)**2))
    sig_ps = 4 * sig_rms**2 / len(clc)
    if popt[2] < 4 * sig_ps:
        j.results.loc[sector, 'f_SLS'] += 4

    # Save the gaussian fit
    j.void[f'popt_{sector}'] = popt

    if j.verbose:
        print(f'### Completed Simple Astropy Lomb-Scargle for Sector {sector} on star {j.gaiaid} ###')

    _safety(j)

def _calculate_wavelet(clc, period_range, sector, j):
    t = clc.time.value
    f = clc.flux.value
    wt = jazzhands.WaveletTransformer(t, f)
    _, _, wwz, wwa = wt.auto_compute(nu_min = 1./period_range[1], nu_max = 1./period_range[0])

    j.void[f'{sector}_wt'] = wt
    j.void[f'{sector}_wwz'] = wwz
    j.void[f'{sector}_wwa'] = wwa

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
    lolim = 0.8*max_p
    uplim = 1.2*max_p

    # If the max period somehow lies outside the period range, don't adjust
    # the period limits
    if (max_p > period_range[0]) & (max_p < period_range[1]):
        if lolim < period_range[0]:
            lolim = period_range[0]
        if uplim > period_range[1]:
            uplim = period_range[1]


    popt, pcov = curve_fit(_gaussian_fn, p, w, p0 = [max_p, 0.1*max_p, max_w],
                            bounds = ([lolim, 0., 0.9*max_w],[uplim, 0.25*max_p, 1.1*max_w]))
    return popt, pcov

def simple_wavelet(j, sector, period_range):
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
        print(f'### Running Wavelet Estimation for Sector {sector} on star {j.gaiaid} ###')

    # Call the relevant light curve
    clc = j.void[f'clc_{sector}']

    popt, pcov = _calculate_wavelet(clc, period_range, sector, j)

    j.results.loc[sector, 'SW'] = popt[0]
    j.results.loc[sector, 'e_SW'] = popt[1]

    # Save the gaussian fit
    j.void[f'{sector}_wavelet_popt'] = popt

    if j.verbose:
        print(f'### Completed Wavelet Estimation for Sector {sector} on star {j.gaiaid} ###')

    _safety(j)

def composite_ACF(j, sector, period_range):
    """
    For the composite ACF (CACF) estimator, we follow the guidelines presented in
    Ceiller et al. (2016, 2017) and Santos et al. (2020, 2021), amongst others.

    The CACF is the product between a normalised collapsed wavelet spectrum and
    the normalised simple ACF.

    We fit the first peak in the resulting spectrum using Scipy.Optimize.curve_fit.
    The resulting mean and width of the Gaussian function approximating the peak is
    reported as the period and associated uncertainty.

    If no peaks are found, no value is reported and a flag is raised when
    validating the rotation periods.

    Parameters
    ----------
    j: class
        The `janet` class containing the metadata on our star.

    sector: int
        The sector for which to calculate the simple astropy lombscargle period.
        If 'all', calculates for all sectors stitched together.

    period_range: tuple
        The lower and upper limit on period range to search for a rotational
        signal. Default is (0.2, 13.7) based on the McQuillan et al. (2014)
        search range and the limitations of TESS earthshine.
    """
    if j.verbose:
        print(f'### Running Composite ACF estimation for Sector {sector} on star {j.gaiaid} ###')

    # Extract the wavelet information
    w = np.flip(np.sum(j.void[f'{sector}_wwz'], axis=1))
    x = np.flip(1./j.void[f'{sector}_wt'].nus)
    f = interpolate.interp1d(x, w)

    # Calculate the ACF for the relevant sector
    lc = j.void[f'clc_{sector}']
    acf = np.correlate(lc.flux.value-1, lc.flux.value-1, mode='full')[len(lc)-1:]
    lag = lc.time.value - lc.time.value.min()
    norm_acf = acf/np.nanmax(acf)
    acflc = lk.LightCurve(time=lag, flux=norm_acf)

    vizacf = acflc[(acflc.time.value <= period_range[1])]
    vizacf = vizacf[(vizacf.time.value >= period_range[0])]

    # Calculate the composite ACF by interoplating the wavelet onto a new x axis
    xnew = vizacf.time.value
    wnew = f(xnew)
    cacf = vizacf * (wnew/np.nanmax(wnew))

    # Smooth the  CACF
    sd = np.ceil(.1 / np.median(np.diff(cacf.time.value)))
    gauss = Gaussian1DKernel(sd)
    cacfsmoo = convolve(cacf.flux.value, gauss, boundary='extend')

    # Identify the first 10 maxima above a threshold of 0.01
    cpeaks, _ = find_peaks(cacfsmoo, height = 0.01)

    # No peaks found
    if len(cpeaks) == 0:
        j.results.loc[sector, 'CACF'] = np.nan

    # Save the metadata
    j.void[f'{sector}_vizacf'] = vizacf
    j.void[f'{sector}_cacf'] = cacf
    j.void[f'{sector}_cacfsmoo'] = cacfsmoo
    j.void[f'{sector}_cpeaks'] = cpeaks

    if len(cpeaks >= 1):
        Px = cacf[cpeaks[0]]['time'].value
        Py = cacfsmoo[cpeaks[0]]

        lolim = 0.8*Px
        if lolim < period_range[0]:
            lolim = period_range[0]
        uplim = 1.2*Px
        if uplim > period_range[1]:
            uplim = period_range[1]

        popt, pcov = curve_fit(_gaussian_fn, cacf.time.value, cacfsmoo,
                               p0 = [Px, 0.1*Px, Py],
                               bounds = ([lolim, 0., 0.9*Py],
                                         [uplim, 0.25*Px, 1.1*Py]))

        j.results.loc[sector, 'CACF'] = popt[0]
        j.results.loc[sector, 'e_CACF'] = popt[1]
        j.void[f'{sector}_cacf_popt'] = popt

    else:
        j.results.loc[sector, 'CACF'] = np.nan
        j.results.loc[sector, 'e_CACF'] = np.nan
        j.void[f'{sector}_cacf_popt'] = np.nan

    if j.verbose:
        print(f'### Completed Composite ACF estimation for Sector {sector} on star {j.gaiaid} ###')

    _safety(j)

def simple_ACF(j, period_range):
    """
    For the ACF estimator, we follow the guidelines presented in Garcia et al.
    (2014), which builds upon the work by McQuillan et al. (2013a, b). There is
    no easy way to reliably estimate an uncertainty for the ACF, so instead we
    will use it as a check on the SLS and WS period estimates.

    First, we take the autocorrelation of the time series, and shifting the
    time series over itself. We then take a periodogram of the ACF, and use the
    period of the peak of highest power as the first-guess ACF period.

    The ACF is then smoothed by convolving with a Gaussian Kernel with a
    standard deviation of 0.1x the first-guess ACF period. We use a peak-
    finding algorithm to identify any peaks in the smoothed spectrum above
    an arbitrary threshold of 0.05.

    If peaks are found, the first (lowest period) peak is used as the ACF period.

    If no peaks are found, no value is reported and a flag is raised when
    validating the rotation periods.

    Parameters
    ----------
    j: class
        The `janet` class containing the metadata on our star.

    sector: int
        The sector for which to calculate the simple astropy lombscargle period.
        If 'all', calculates for all sectors stitched together.

    period_range: tuple
        The lower and upper limit on period range to search for a rotational
        signal. Default is (0.2, 13.7) based on the McQuillan et al. (2014)
        search range and the limitations of TESS earthshine.
    """

    if j.verbose:
        print(f'### Running ACF Estimation on star {j.gaiaid} ###')

    clc = j.void['clc_all']

    # Calculate the ACF between 0 and 12 days.
    acf = np.correlate(clc.flux.value-1, clc.flux.value-1, mode='full')[len(clc)-1:]
    lag = clc.time.value - np.nanmin(clc.time.value)

    # Cut up and normalize the ACF
    secmin = j.sectors[0]
    norm_acf = acf/np.nanmax(acf)
    acflc = lk.LightCurve(time=lag, flux=norm_acf)
    acflc = acflc[acflc.time.value < (j.void[f'clc_{secmin}'].time.value - j.void[f'clc_{secmin}'].time.value.min()).max()]

    # Estimate a first-guess period
    acfpg = acflc.to_periodogram()
    first_guess = acfpg.period_at_max_power

    # Limit the search range
    if not period_range[0] < first_guess.value < period_range[1]:
        warnings.warn("The highest peak in the ACF lies outside the period range of your search.")
    vizacf = acflc[(acflc.time.value <= period_range[1])]
    vizacf = vizacf[(vizacf.time.value >= period_range[0])]

    # Smooth the  ACF
    sd = np.ceil(.1 / np.median(np.diff(vizacf.time.value)))
    gauss = Gaussian1DKernel(sd)
    acfsmoo = convolve(vizacf.flux.value, gauss, boundary='extend')

    # Identify the first 10 maxima above a threshold of 0.01
    peaks, _ = find_peaks(acfsmoo, height = 0.01)

    # Save the metadata
    j.void['acflc'] = acflc
    j.void['vizacf'] = vizacf
    j.void['acfsmoo'] = acfsmoo
    j.void['peaks'] = peaks

    # The first of these maxima (with the shortest period) corresponds to Prot
    if len(peaks) >= 1:
        acf_period = vizacf.time.value[peaks[0]]
        j.results.loc['all', 'ACF'] = acf_period

    # No peaks found
    else:
        j.results.loc['all', 'ACF'] = np.nan

    if j.verbose:
        print(f'### Completed ACF Estimation on star {j.gaiaid} ###')

    _safety(j)
