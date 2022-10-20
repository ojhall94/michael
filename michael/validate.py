import pandas as pd
import numpy as np
import warnings
from scipy.ndimage import gaussian_filter1d
import scipy.stats
from .utils import _safety

def longest_sectors(j):
    if len(j.sectors) == 1:
        return j.sectors[0]

    diffs = np.zeros(len(j.sectors))
    for idx, s in enumerate(j.sectors):
        d = np.diff(np.array(s.split('-')).astype(int))
        if np.isfinite(d):
            diffs[idx] = d

    # Returns the longest sectors
    dmax = np.nanmax(diffs)
    sel = np.where(diffs == dmax)
    return list(np.array(j.sectors)[sel])

def validate_SLS(j):
    j.results['s_SLS'] = np.nan

    # Check for p2p-validated sectors
    indices = j.results[j.results['f_p2p_SLS'] == 1].index

    # If all targets have a p2p flag of 0, consider all sectors together
    if len(indices) == 0:
        indices = j.sectors

    # Peak the value with highest SNR on an unflagged value
    s = j.results.loc[indices]['f_SLS'] == 0
    if len(j.results.loc[indices][s]) > 0:
        best = j.results.loc[indices][s]['snr_SLS'].idxmax()
    # It may be the case that there are only flagged values. In this
    # case, ignore the flags
    else:
        best = j.results.loc[indices]['snr_SLS'].idxmax()

    j.results.loc['best', 'SLS'] = j.results.loc[best, 'SLS']
    j.results.loc['best', 'e_SLS'] = j.results.loc[best, 'e_SLS']
    j.results.loc['best', 'h_SLS'] = j.results.loc[best, 'h_SLS']
    j.results.loc['best', 's_SLS'] = best
    j.results.loc['best', 'f_SLS'] = j.results.loc[best, 'f_SLS']
    j.results.loc['best', 'p2p_SLS'] = j.results.loc[best, 'p2p_SLS']

    _safety(j)

def validate_SW(j):
    j.results['s_SW'] = np.nan

    best = j.results.snr_SW.idxmax()

    #  Consider the longest sector the default best
    j.results.loc['best', 'SW'] = j.results.loc[best, 'SW']
    j.results.loc['best', 'e_SW'] = j.results.loc[best, 'e_SW']
    j.results.loc['best', 'h_SW'] = j.results.loc[best, 'h_SW']
    j.results.loc['best', 's_SW'] = best
    j.results.loc['best', 'p2p_SW'] = j.results.loc[best, 'p2p_SW']

    _safety(j)

def validate_CACF(j):
    j.results['s_CACF'] = np.nan

    best = j.results.snr_CACF.idxmax()

    j.results.loc['best', 'CACF'] = j.results.loc[best, 'CACF']
    j.results.loc['best', 'e_CACF'] = j.results.loc[best, 'e_CACF']
    j.results.loc['best', 'h_CACF'] = j.results.loc[best, 'h_CACF']
    j.results.loc['best', 's_CACF'] = best
    j.results.loc['best', 'p2p_CACF'] = j.results.loc[best, 'p2p_CACF']
    _safety(j)

def validate_ACF(j):
    j.results['s_ACF'] = np.nan

    best = j.results.snr_ACF.idxmax()

    #  Consider the longest sector the default best
    j.results.loc['best', 'ACF'] = j.results.loc[best, 'ACF']
    j.results.loc['best', 's_ACF'] = best
    j.results.loc['best', 'p2p_ACF'] = j.results.loc[best, 'p2p_ACF']

    _safety(j)

def validate_best(j):
    # Validate the best estimates against one another
    # Check to see if they agree closely with one another
    methods = ['SLS', 'SW', 'CACF', 'ACF']
    best = j.results.loc['best', ['SLS','SW','CACF', 'ACF']]#.dropna()
    ebest = j.results.loc['best', ['e_SLS','e_SW','e_CACF', 'e_ACF']]#.dropna()
    snrs = j.results.loc['best', ['snr_SLS','snr_SW','snr_CACF', 'snr_ACF']]#.dropna()']
    p2ps = j.results.loc['best', ['p2p_SLS','p2p_SW','p2p_CACF', 'p2p_ACF']]#.dropna()']

    j.results['f_overall'] = np.zeros(len(j.results)).astype(int)

    # If they agree, then pick the one with the highest snr value
    if np.abs(np.diff(best.dropna(), len(best.dropna())-1)) < np.sqrt(np.nansum(ebest**2)):
        s = np.argmax(snrs)
        j.results.loc['best', 'overall'] = best[s]
        j.results.loc['best', 'e_overall'] = ebest[s]
        j.results.loc['best', 'method_overall'] = methods[s]
        j.results.loc['best', 'p2p_overall'] = p2ps[s]

    # If they disagree, see if two of them are in agreement
    else:
        # We check in this order, as the priority is CACF -> SW -> SLS -> ACF
        d = np.abs(np.diff(best[['SLS','SW']])) < np.sqrt(np.sum(ebest[['e_SLS', 'e_SW']]**2))
        c = np.abs(np.diff(best[['ACF', 'CACF']])) < ebest[['e_CACF']].values
        b = np.abs(np.diff(best[['SLS','CACF']])) < np.sqrt(np.sum(ebest[['e_SLS', 'e_CACF']]**2))
        a = np.abs(np.diff(best[['SW','CACF']])) < np.sqrt(np.sum(ebest[['e_SW', 'e_CACF']]**2))

        # SW and CACF are in agreement
        if a:
            s = np.argmax(snrs[['snr_SW','snr_CACF']])
            j.results.loc['best', 'overall'] = best[['SW', 'CACF']][s]
            j.results.loc['best', 'e_overall'] = ebest[['e_SW', 'e_CACF']][s]
            j.results.loc['best', 'p2p_overall'] = p2ps[['p2p_SW', 'p2p_CACF']][s]
            j.results.loc['best', 'method_overall'] = ['SW','CACF'][s]
            j.results.loc['best', 'f_overall'] += int(4)

        # SLS and CACF are in agreement
        elif b:
            s = np.argmax(snrs[['snr_SLS','snr_CACF']])
            j.results.loc['best', 'overall'] = best[['SLS', 'CACF']][s]
            j.results.loc['best', 'e_overall'] = ebest[['e_SLS', 'e_CACF']][s]
            j.results.loc['best', 'p2p_overall'] = p2ps[['p2p_SLS', 'p2p_CACF']][s]
            j.results.loc['best', 'method_overall'] = ['SLS','CACF'][s]
            j.results.loc['best', 'f_overall'] += int(4)

        # CACF and ACF are in agreement
        elif c:
            s = np.argmax(snrs[['snr_ACF','snr_CACF']])
            j.results.loc['best', 'overall'] = best[['ACF', 'CACF']][s]
            j.results.loc['best', 'e_overall'] = ebest[['e_ACF', 'e_CACF']][s]
            j.results.loc['best', 'p2p_overall'] = p2ps[['p2p_ACF', 'p2p_CACF']][s]
            j.results.loc['best', 'method_overall'] = ['ACF','CACF'][s]
            j.results.loc['best', 'f_overall'] += int(4)

        # SLS and SW are in agreement
        elif d:
            s = np.argmax(snrs[['snr_SLS','snr_SW']])
            j.results.loc['best', 'overall'] = best[['SLS', 'SW']][s]
            j.results.loc['best', 'e_overall'] = ebest[['e_SLS', 'e_SW']][s]
            j.results.loc['best', 'p2p_overall'] = p2ps[['p2p_SLS', 'p2p_SW']][s]
            j.results.loc['best', 'method_overall'] = ['SLS','SW'][s]
            j.results.loc['best', 'f_overall'] += 4

        # There is no agreement whatsover, pick the highest SNR
        else:
            s = np.argmax(snrs)
            j.results.loc['best', 'overall'] = best[s]
            j.results.loc['best', 'e_overall'] = ebest[s]
            j.results.loc['best', 'p2p_overall'] = p2ps[s]
            j.results.loc['best', 'method_overall'] = methods[s]
            j.results.loc['best', 'f_overall'] += 1
            warnings.warn("No estimates could agree. Please inspect the results carefully yourself.")

    method = j.results.loc['best','method_overall']
    sector = j.results.loc['best',f's_{method}']
    flag = j.results.loc[sector, f'f_p2p_{method}']

    if flag == 0:
        j.results.loc['best', 'f_overall'] += 16
    _safety(j)

def validate_sectors(j):
    # Check if any individual sectors are wildly out of line, possibly due to binaries
    res = j.results.loc[(j.results.index != 'best'), ['SLS','SW','CACF', 'ACF']]
    err = j.results.loc[(j.results.index != 'best'), ['e_SLS', 'e_SW', 'e_CACF', 'e_ACF']]

    a = np.abs(np.diff(res,axis=0, n = len(res)-1))
    b = np.sqrt(np.nansum(err**2, axis=0))

    # Do any sectors disagree repeatedly over 1 sigma across all sectors?
    if all(list((a-b > 0)[0])):
        j.results.loc['best', 'f_overall'] += 8

    warnings.warn("One or more sectors disagree strongly across all estimates. Please inspect the results carefully yourself.")

def validate_prior(j):
    # Check how the various measurements are consistent with the KDE prior
    # Is the prior in disagreement with the 'best' result?
    condition = (j.results.loc['best','overall'] + j.results.loc['best','e_overall']\
                > j.prot_prior[0]) &\
                (j.results.loc['best','overall'] - j.results.loc['best','e_overall']\
                            < j.prot_prior[2])
    if not condition:
        j.results.loc['best', 'f_overall'] += 1024
        warnings.warn("The prior on rotation disagrees with the best measured value. The prior is not necessarily correct and only a guide. Please inspect the results carefully yourself.")

        # Is the prior an integer multiple of the best result overall?
        protsamp = 10**j.samples[:,2]
        res = np.random.randn(len(protsamp)) * j.results.loc['best','e_overall']\
                + j.results.loc['best', 'overall']

        if np.nanmean(res) > np.nanmean(protsamp):
            div = res/protsamp
        else:
            div = protsamp/res
        pars = np.nanpercentile(div, [16, 50, 84])
        lim = np.round(pars[1], 0)
        condition = (lim > pars[0]) & (lim < pars[1])
        if condition:
            j.results.loc['best', 'f_overall'] += 2048
            warnings.warn("The prior on rotation agrees with an integer multiple of the best measured value. This may indicate that `michael` has measured a harmonic. Please inspect the results carefully yourself.")

def validate_p2p(j):
    """
    A good measurement of a rotation period should be clearly visible in a
    light curve folded on the measured period. If folded on an incorrect
    period, this will decreate the peak-to-peak height of the periodic motion,
    as the data will be scattered.

    In order to validate the rotation rates, we will flag any targets for which
    the peak-to-peak (p2p) height is smaller than the standard deviation on the
    flux values for a given sectors (i.e. the rotation signal could be the
    product of noise).

    The p2p height is measured as the difference between the maximum and minimum
    of a smoothed folded light curve. The folded light curve is smoothed using a
    the `scipy.gaussian_filter1d` package, with a standard deviation of `sqrt(N)`
    where N is the number of cadences in the sector.

    The light curve is then divided by the smoothed light curve, and the median
    absolute deviation (MAD) calculated. In cases where the p2p height exceeds
    the MAD, the target is given a p2p flag of 1, indicating a good
    detection. The Signal-to-noise ratio is cacluacted as

    snr = p2p / MAD,

    and recorded.

    This method is fairly sensitive to systematics or flares, as they will
    cause a large p2p signal. As always, validation should be taken with a grain
    of salt!
    """

    methods = ['SLS', 'SW','CACF','ACF']
    for m in methods:
        j.results[f'p2p_{m}'] = np.zeros(len(j.results)).astype(int)
        j.results[f'snr_{m}']  = np.zeros(len(j.results)).astype(int)
        j.results[f'f_p2p_{m}'] = np.zeros(len(j.results)).astype(int)

        for s in j.sectors:
            period = j.results.loc[s, m]
            lc = j.void[f'{j.pl}lc_{s}'].fold(period = period)
            sd = np.sqrt(len(lc))
            fsmoo = gaussian_filter1d(lc.flux.value, sigma = sd, mode = 'reflect')

            p2p = np.diff([np.nanmin(fsmoo), np.nanmax(fsmoo)])
            j.results.loc[s, f'p2p_{m}'] = p2p

            mad = scipy.stats.median_abs_deviation(lc.flux.value /
                                            gaussian_filter1d(lc.flux.value,
                                                sigma = sd, mode = 'nearest'))
            if p2p > 2*mad:
                j.results.loc[s, f'f_p2p_{m}'] = int(1)

            j.results.loc[s, f'snr_{m}'] = p2p/(2*mad)

    _safety(j)

def validator(j):
    """
    TODO: THIS DOCSTRING IS OUT OF DATE

    ## Flag values
    Overall flag values are:
    1 - No robust matches, CACF assumed best
    2 - No robust matches, SW assumed best
    4 - Only two out of three estimates agreed
    8 - One or more sectors disagree strongly across all estimates
    16 - The 'best overall' value does not clear the p2p boundary.
    """

    # Peak-to-peak validation
    validate_p2p(j)

    # Validate LombScargle
    validate_SLS(j)

    # Validate Wavelet
    validate_SW(j)

    # Validate Composite ACF
    validate_CACF(j)

    # Validate regular ACF
    validate_ACF(j)

    # Validate the three estimates
    validate_best(j)

    # Validate ACF vs the 'best' period
    # validate_best_vs_ACF(j)

    # Validate individual sectors
    if len(j.sectors) > 1:
        validate_sectors(j)

    # if j.samples is not None:
    #     validate_prior(j)


    _safety(j)
