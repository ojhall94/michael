import pandas as pd
import numpy as np
import warnings
from .utils import _safety

def longest_sector(j):
    if len(j.sectors) == 1:
        return j.sectors[0]

    diffs = np.zeros(len(j.sectors))
    for idx, s in enumerate(j.sectors):
        d = np.diff(np.array(s.split('-')).astype(int))
        if np.isfinite(d):
            diffs[idx] = d

    # Will pick the first longest
    if any(diffs > 0):
        return j.sectors[np.argmax(diffs)]
    else:
        return None

def validate_SLS(j):
    j.results['s_SLS'] = np.nan
    # Validate LombScargle
    longest = longest_sector(j)
    if longest is not None:
        # Consider the longest sector the default best
        j.results.loc['best', 'SLS'] = j.results.loc[longest, 'SLS']
        j.results.loc['best', 'e_SLS'] = j.results.loc[longest, 'e_SLS']
        j.results.loc['best', 'h_SLS'] = j.results.loc[longest, 'h_SLS']
        j.results.loc['best', 's_SLS'] = longest
        j.results.loc['best', 'f_SLS'] = j.results.loc[longest, 'f_SLS']

    else:
        # If onlys single-sector cases are available, pick the value with
        # the highest peak height on an unflagged value
        s = j.results['f_SLS'] == 0
        if len(j.results[s]) > 0:
            idx = np.array(j.results[s]['h_SLS'].idxmax())
        # It may be the case that there are only flagged values. In this
        # case, ignore the flags
        else:
            idx = np.array(j.results['h_SLS'].idxmax())

        j.results.loc['best', 'SLS'] = j.results.loc[idx, 'SLS']
        j.results.loc['best', 'e_SLS'] = j.results.loc[idx, 'e_SLS']
        j.results.loc['best', 'h_SLS'] = j.results.loc[idx, 'h_SLS']
        j.results.loc['best', 's_SLS'] = idx.astype(str)
        j.results.loc['best', 'f_SLS'] = j.results.loc[idx, 'f_SLS']
    _safety(j)

def validate_SW(j):
    j.results['s_SW'] = np.nan

    longest = longest_sector(j)
    if longest is not None:
        #  Consider the longest sector the default best
        j.results.loc['best', 'SW'] = j.results.loc[longest, 'SW']
        j.results.loc['best', 'e_SW'] = j.results.loc[longest, 'e_SW']
        j.results.loc['best', 'h_SW'] = j.results.loc[longest, 'h_SW']
        j.results.loc['best', 's_SW'] = longest

    else:
        # If onlys single-sector cases are available, pick the value with
        # the highest peak
        idx = np.array(j.results['h_SW'].idxmax())

        j.results.loc['best', 'SW'] = j.results.loc[idx, 'SW']
        j.results.loc['best', 'e_SW'] = j.results.loc[idx, 'e_SW']
        j.results.loc['best', 'h_SW'] = j.results.loc[idx, 'h_SW']
        j.results.loc['best', 's_SW'] = idx.astype(str)
    _safety(j)

def validate_CACF(j):
    j.results['s_CACF'] = np.nan

    longest = longest_sector(j)
    if longest is not None:
        #  Consider the longest sector the default best
        j.results.loc['best', 'CACF'] = j.results.loc[longest, 'CACF']
        j.results.loc['best', 'e_CACF'] = j.results.loc[longest, 'e_CACF']
        j.results.loc['best', 'h_CACF'] = j.results.loc[longest, 'h_CACF']
        j.results.loc['best', 's_CACF'] = longest
    #
    else:
        # If only single-sector cases are available, pick cases where double
        # peaks are occuring within 2sigma.
        flag = np.zeros(len(j.sectors), dtype=bool)

        for idx, sector in enumerate(j.sectors):
            lolim = j.results.loc[sector, 'CACF'] - 2*j.results.loc[sector, 'e_CACF']
            uplim = j.results.loc[sector, 'CACF'] + 2*j.results.loc[sector, 'e_CACF']

            altpeaks_x = []
            mask = np.ones(len(j.sectors), dtype=bool)
            mask[idx] = 0
            if len(j.sectors[mask]) > 1:
                for s in j.sectors[mask]:
                    for p in j.void[f'{s}_cpeaks']:
                        altpeaks_x.append(j.void[f'{s}_cacf'][p]['time'].value)
            else:
                s = j.sectors[mask][0]
                for p in j.void[f'{s}_cpeaks']:
                    altpeaks_x.append(j.void[f'{s}_cacf'][p]['time'].value)

            # Check if there are any peaks from other sectors present within 2 sigma
            flag[idx] = any(altpeaks_x - j.results.loc[sector, 'CACF'] < j.results.loc[sector, 'e_CACF'])


            # If flag 1 on one sector, then that's the "best"
            if len(flag[flag == 1]):
                idx = j.sectors[np.argmax(flag)]

            # If flag 1 on multiple sectors, then select "best" based on peak height
            elif len(flag[flag > 1]):
                s = j.sectors[flag]
                idx = np.array(j.results[s]['h_CACF'].idxmax())

            # If 0 or 1 on all, then select "best" based on peak height.
            elif np.min(flag) == np.max(flag):
                idx = np.array(j.results['h_CACF'].idxmax())

            # Otherwise, pick sector with lowest fractional uncertainty
            else:
                idx = np.array(j.results['e_CACF']/j.results['CACF']).idxmin()

            j.results.loc['best', 'CACF'] = j.results.loc[idx, 'CACF']
            j.results.loc['best', 'e_CACF'] = j.results.loc[idx, 'e_CACF']
            j.results.loc['best', 'h_CACF'] = j.results.loc[idx, 'h_CACF']
            j.results.loc['best', 's_CACF'] = idx.astype(str)
    _safety(j)

def validate_best(j):
    # Validate the best estimates against one another
    # Check to see if they agree closely with one another
    best = j.results.loc['best', ['SLS','SW','CACF']].dropna()
    ebest = j.results.loc['best', ['e_SLS','e_SW','e_CACF']].dropna()

    # If they agree, then pick the one with the best fractional uncertainty
    if np.abs(np.diff(best, len(best)-1)) < np.sqrt(np.sum(ebest**2)):
        frac = ebest.values /  best.values
        s = np.argmin(frac)
        j.results.loc['best', 'overall'] = best[s]
        j.results.loc['best', 'e_overall'] = ebest[s]
        j.results.loc['best', 'f_overall'] = 2**s

    # If they disagree, see if two of them are in agreement
    else:
        # We check in this order, as the priority is CACF -> SW -> SLS
        best = j.results.loc['best', ['SLS','SW','CACF']]
        ebest = j.results.loc['best', ['e_SLS','e_SW','e_CACF']]
        a = np.abs(np.diff(best[['SLS','SW']])) < np.sqrt(np.sum(ebest[['e_SLS', 'e_SW']]**2))
        b = np.abs(np.diff(best[['SLS','CACF']])) < np.sqrt(np.sum(ebest[['e_SLS', 'e_CACF']]**2))
        c = np.abs(np.diff(best[['SW','CACF']])) < np.sqrt(np.sum(ebest[['e_SW', 'e_CACF']]**2))

        # SW and CACF are in agreement
        if c:
            frac = ebest[['e_SW', 'e_CACF']].values /  best[['SW', 'CACF']].values
            s = np.argmin(frac)
            j.results.loc['best', 'overall'] = best[['SW', 'CACF']][s]
            j.results.loc['best', 'e_overall'] = ebest[['e_SW', 'e_CACF']][s]
            j.results.loc['best', 'f_overall'] = 2**(s+1) + 128

        # SLS and CACF are in agreement
        elif b:
            frac = ebest[['e_SLS','e_CACF']].values /  best[['SLS','CACF']].values
            s = np.argmin(frac)
            j.results.loc['best', 'overall'] = best[['SLS','CACF']][s]
            j.results.loc['best', 'e_overall'] = ebest[['e_SLS','e_CACF']][s]
            if s == 0:
                j.results.loc['best', 'f_overall'] = 1 + 128
            else:
                 j.results.loc['best', 'f_overall'] = 4 + 128

        # SLS and SW are in agreement
        elif a:
            frac = ebest[['e_SLS','e_SW']].values /  best[['SLS','SW']].values
            s = np.argmin(frac)
            j.results.loc['best', 'overall'] = best[['SLS','SW']][s]
            j.results.loc['best', 'e_overall'] = ebest[['e_SLS','e_SW']][s]
            j.results.loc['best', 'f_overall'] = 2**s + 128

        # There is no agreement whatsover
        else:
            if np.isfinite(best['CACF']):
                j.results.loc['best', 'overall'] = best['CACF']
                j.results.loc['best', 'e_overall'] = ebest['e_CACF']
                j.results.loc['best', 'f_overall'] = 4 + 256
                warnings.warn("No estimates could agree. Please inspect the results carefully yourself.")
            elif np.isfinite(best['SW']):
                frac = ebest[['e_SLS','e_SW']].values /  best[['SLS','SW']].values
                s = np.argmin(frac)
                j.results.loc['best', 'overall'] = best[['SLS','SW']][s]
                j.results.loc['best', 'e_overall'] = ebest[['e_SLS','e_SW']][s]
                j.results.loc['best', 'f_overall'] = 2**s + 256
                warnings.warn("No estimates could agree. Please inspect the results carefully yourself.")
    _safety(j)

def validate_best_vs_ACF(j):
    # Validate the ACF vs the best value
    if np.isfinite(j.results.loc['all', 'ACF']):
        # Flag if the ACF does not match the 'best' period within 2 sigma
        condition = np.abs(j.results.loc['all', 'ACF'] - j.results.loc['best', 'overall'])\
                    < 2*j.results.loc['best','e_overall']
        if not condition:
            j.results.loc['best', 'f_overall'] += 32

        # Flag if the ACF appears to be a harmonic of the 'best' period
        condition = (np.abs(0.5*j.results.loc['all', 'ACF'] - j.results.loc['best', 'overall'])\
                    < 2*j.results.loc['best', 'e_overall']) or\
                    (np.abs(2*j.results.loc['all', 'ACF'] - j.results.loc['best', 'overall'])\
                                < 2*j.results.loc['best', 'e_overall'])
        if condition:
            j.results.loc['best', 'f_overall'] += 64
    else:
        j.results.loc['best', 'f_overall'] += 16
    _safety(j)

def validate_sectors(j):
    # Check if any individual sectors are wildly out of line, possibly due to binaries
    res = j.results.loc[(j.results.index != 'best') & (j.results.index != 'all'), ['SLS','SW','CACF']]
    err = j.results.loc[(j.results.index != 'best') & (j.results.index != 'all'), ['e_SLS', 'e_SW', 'e_CACF']]

    a = np.abs(np.diff(res,axis=0, n = len(res)-1))
    b = np.sqrt(np.sum(err**2, axis=0)).values

    # Do any sectors disagree repeatedly over 1 sigma across all sectors?
    if all(list((a-b > 0)[0])):
        j.results.loc['best', 'f_overall'] += 512

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

def validator(j):
    """
    TODO: THIS DOCSTRING IS OUT OF DATE

    This function will validate the measured rotation rates and determine
    a value that it considers to be the best. It does this following a flow-
    chart, starting with the Lomb Scargle periodogram results.

    ## Validating the Simple Lomb Scargle (SLS) Period
    - If there is a SLS value for 'all' sectors, this is the 'best' value.
    - Otherwise, the SLS value with the lowest fractional uncertainty in
        and unflagged sector is deemed the 'best' value.
        - If all sectors have flags, the 'flag' condition is ignored.

    ## Validating the Simple Wavelet (SW) Period vs the SLS Period
    - There is only one Wavelet Period, which is the best by default.
    - If the 'best' SW and SLS periods agree within 1 sigma, the value
        with the smallest fractional uncertainty is chosen as the 'best
        overall' rotation period.
        - If there is no agreement within 1 sigma, we check whether the SW
            agrees within 1 sigma with any *unflagged* single-sector SLS
            periods.
        - If there are no matching *unflagged* single-sector SLS periods,
            no match is found. The wavelet is then assumed to be the 'best
            overall' rotation period, and the value is flagged.

    ## Validating the ACF Period vs the 'Best Overall' Period
    - TO DO
        Something like, verify the ACF has the same value as the best? if not,
        search for a corresponding peak and flag as maybe a harmonic?

    ## Validating the Gaussian Process (GP) period
    - As the GP is the most statistically intensive measurement of the
        rotation, it is automatically taken to be the 'best overall' period.

    ## Flag values
    Overall flag values are:
    1 - SLS-obtained value
    2 - SW-obtained value
    4 - CACF-obtained value
    8 - GP-obtained value
    16 - No ACF measured
    32 - ACF does not match 'best' period within 2 sigma
    64 - ACF indicates that 'best' period is a potential harmonic
    128 - Only two out of three estimates agreed
    256 - No robust matches, CACF assumed best
    512 - One or more sectors disagree strongly across all estimates
    1024 - The result disagrees with a prior value.
    2048 - The result is an integer multiple of the prior value (likely harmonic).
    """
    # Validate LombScargle
    validate_SLS(j)

    # Validate Wavelet
    validate_SW(j)

    # Validate Composite ACF
    validate_CACF(j)

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
