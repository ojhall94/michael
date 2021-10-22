import pandas as pd
import numpy as np
import warnings
from .utils import _safety

def validate_SLS(j):
    j.results['s_SLS'] = np.nan
    # Validate LombScargle
    if np.isfinite(j.results.loc['all', 'SLS']):
        # If there is a LS value for 'all', consider this the default best
        j.results.loc['best', 'SLS'] = j.results.loc['all', 'SLS']
        j.results.loc['best', 'e_SLS'] = j.results.loc['all', 'e_SLS']
        j.results.loc['best', 's_SLS'] = 'all'
        j.results.loc['best', 'f_SLS'] = j.results.loc['all', 'f_SLS']

    else:
        # If onlys single-sector cases are available, pick the value with
        # the lowest fractional uncertainty on an unflagged value
        s = j.results['f_SLS'] == 0
        if len(j.results[s]) > 0:
            sigfrac = j.results[s]['e_SLS'] / j.results[s]['SLS']

        # It may be the case that there are only flagged values. In this
        # case, ignore the flags
        else:
            sigfrac = j.results['e_SLS'] / j.results['SLS']

        idx = np.array(sigfrac.idxmin())
        if np.isfinite(idx):
            j.results.loc['best', 'SLS'] = j.results.loc[idx, 'SLS']
            j.results.loc['best', 'e_SLS'] = j.results.loc[idx, 'e_SLS']
            j.results.loc['best', 's_SLS'] = idx.astype(str)
            j.results.loc['best', 'f_SLS'] = j.results.loc[idx, 'f_SLS']
    _safety(j)

def validate_SW(j):
    j.results['s_SW'] = np.nan
    if np.isfinite(j.results.loc['all', 'SW']):
        # If there is a SW value for 'all', consider this the default best
        j.results.loc['best', 'SW'] = j.results.loc['all', 'SW']
        j.results.loc['best', 'e_SW'] = j.results.loc['all', 'e_SW']
        j.results.loc['best', 's_SW'] = 'all'

    else:
        # If onlys single-sector cases are available, pick the value with
        # the lowest fractional uncertainty
        sigfrac = j.results['e_SW'] / j.results['SW']
        idx = np.array(sigfrac.idxmin())

        if np.isfinite(idx):
            j.results.loc['best', 'SW'] = j.results.loc[idx, 'SW']
            j.results.loc['best', 'e_SW'] = j.results.loc[idx, 'e_SW']
            j.results.loc['best', 's_SW'] = idx.astype(str)
    _safety(j)

def validate_CACF(j):
    j.results['s_CACF'] = np.nan
    if np.isfinite(j.results.loc['all', 'CACF']):
        # If there is a CACF value for 'all', consider this the default best
        j.results.loc['best', 'CACF'] = j.results.loc['all', 'CACF']
        j.results.loc['best', 'e_CACF'] = j.results.loc['all', 'e_CACF']
        j.results.loc['best', 's_CACF'] = 'all'

    else:
        # If onlys single-sector cases are available, pick the value with
        # the lowest fractional uncertainty
        sigfrac = j.results['e_CACF'] / j.results['CACF']
        idx = np.array(sigfrac.idxmin())

        if np.isfinite(idx):
            j.results.loc['best', 'CACF'] = j.results.loc[idx, 'CACF']
            j.results.loc['best', 'e_CACF'] = j.results.loc[idx, 'e_CACF']
            j.results.loc['best', 's_CACF'] = idx.astype(str)
    _safety(j)

def validate_best(j):
    # Validate the best estimates against one another
    # Check to see if they agree closely with one another
    best = j.results.loc['best', ['SLS','SW','CACF']]
    ebest = j.results.loc['best', ['e_SLS','e_SW','e_CACF']]

    # If they agree, then pick the one with the best fractional uncertainty
    if np.abs(np.diff(best, 2)) < np.sqrt(np.sum(ebest**2)):
        frac = ebest.values /  best.values
        s = np.argmin(frac)
        j.results.loc['best', 'overall'] = best[s]
        j.results.loc['best', 'e_overall'] = ebest[s]
        j.results.loc['best', 'f_overall'] = 2**s

    # If they disagree, see if two of them are in agreement
    else:
        # We check in this order, as the priority is SLS -> SW -> CACF
        a = np.abs(np.diff(best[['SLS','SW']])) < np.sqrt(np.sum(ebest[['e_SLS', 'e_SW']]**2))
        b = np.abs(np.diff(best[['SLS','CACF']])) < np.sqrt(np.sum(ebest[['e_SLS', 'e_CACF']]**2))
        c = np.abs(np.diff(best[['SW','CACF']])) < np.sqrt(np.sum(ebest[['e_SW', 'e_CACF']]**2))

        # SLS and SW are in agreement
        if a:
            frac = ebest[['e_SLS','e_SW']].values /  best[['SLS','SW']].values
            s = np.argmin(frac)
            j.results.loc['best', 'overall'] = best[['SLS','SW']][s]
            j.results.loc['best', 'e_overall'] = ebest[['e_SLS','e_SW']][s]
            j.results.loc['best', 'f_overall'] = 2**s + 16

        # SLS and CACF are in agreement
        elif b:
            frac = ebest[['e_SLS','e_CACF']].values /  best[['SLS','CACF']].values
            s = np.argmin(frac)
            j.results.loc['best', 'overall'] = best[['SLS','CACF']][s]
            j.results.loc['best', 'e_overall'] = ebest[['e_SLS','e_CACF']][s]
            if s == 0:
                j.results.loc['best', 'f_overall'] = 1 + 16
            else:
                 j.results.loc['best', 'f_overall'] = 4 + 16

        # SW and CACF are in agreement
        elif c:
            frac = ebest[['e_SW', 'e_CACF']].values /  best[['SW', 'CACF']].values
            s = np.argmin(frac)
            j.results.loc['best', 'overall'] = best[['SW', 'CACF']][s]
            j.results.loc['best', 'e_overall'] = ebest[['e_SW', 'e_CACF']][s]
            j.results.loc['best', 'f_overall'] = 2**(s+1) + 16

        # There is no agreement whatsover
        else:
            j.results.loc['best', 'overall'] = best['CACF']
            j.results.loc['best', 'e_overall'] = ebest['e_CACF']
            j.results.loc['best', 'f_overall'] = 4 + 32

            warnings.warn("No estimates could agree. Please inspect the results carefully yourself.")
    _safety(j)

def validate_best_vs_ACF(j):
    # Validate the ACF vs the best value

    # Flag if the ACF does not match the 'best' period within 2 sigma
    condition = np.abs(j.results.loc['all', 'ACF'] - j.results.loc['best', 'overall'])\
                < 2*j.results.loc['best','e_overall']
    if not condition:
        j.results.loc['best', 'f_overall'] += 128

    # Flag if the ACF appears to be a harmonic of the 'best' period
    condition = (np.abs(0.5*j.results.loc['all', 'ACF'] - j.results.loc['best', 'overall'])\
                < 2*j.results.loc['best', 'e_overall']) or\
                (np.abs(2*j.results.loc['all', 'ACF'] - j.results.loc['best', 'overall'])\
                            < 2*j.results.loc['best', 'e_overall'])
    if condition:
        j.results.loc['best', 'f_overall'] += 256
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
    16 - Only two out of three estimates agreed
    32 - No robust matches, CACF assumed best
    64 - No ACF measured
    128 - ACF does not match 'best' period within 2 sigma
    256 - ACF indicates that 'best' period is a potential harmonic
    512 - One or more sectors disagree strongly across all estimates
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
    validate_best_vs_ACF(j)

    # Validate individual sectors
    if len(j.sectors) > 1:
        validate_sectors(j)

    _safety(j)
