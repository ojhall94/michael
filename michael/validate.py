import pandas as pd
import numpy as np
from .utils import _safety

def validate_SLS(j):
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

        idx = sigfrac.idxmin()
        j.results.loc['best', 'SLS'] = j.results.loc[idx, 'SLS']
        j.results.loc['best', 'e_SLS'] = j.results.loc[idx, 'e_SLS']
        j.results.loc['best', 's_SLS'] = str(int(idx))
        j.results.loc['best', 'f_SLS'] = j.results.loc[idx, 'f_SLS']
    _safety(j)

def validate_WS_vs_SLS(j):
    # Validate Wavelet vs LombScargle
    j.results.loc['best', 'SW'] = j.results.loc['all', 'SW']
    j.results.loc['best', 'e_SW'] = j.results.loc['all', 'e_SW']

    # Check to see if they agree within 1 sigma
    best = j.results.loc['best']

    # If they agree, then pick the one with the best fractional uncertainty
    if np.diff(best[['SLS', 'SW']]) < np.sum(best[['e_SLS', 'e_SW']]):
        frac = best[['e_SLS', 'e_SW']].values /  best[['SLS', 'SW']].values
        s = np.argmin(frac)
        j.results.loc['best', 'overall'] = best[['SLS', 'SW'][s]]
        j.results.loc['best', 'e_overall'] = best[['e_SLS', 'e_SW'][s]]
        j.results.loc['best', 'f_overall'] = s + 1

    # If they disagree, see if there are any matches with another sector
    else:
        if len(j.sectors) >= 2:
            sls = j.results.loc[j.sectors,['SLS', 'e_SLS', 'f_SLS']]
        else:
            sls = j.results.loc['all',['SLS', 'e_SLS', 'f_SLS']]

        swb = j.results.loc['best', 'SW']
        e_swb = j.results.loc['best', 'e_SW']

        # An agreement within 1 Sigma has been found
        if np.any(np.abs(sls.SLS - swb) - (e_swb + sls.e_SLS) < 0):
            match = sls[np.abs(sls.SLS - swb) - (e_swb + sls.e_SLS) < 0]
            frac = match.e_SLS / match.SLS
            bestmatch = frac.idxmin()

            # No matching results found without a flag, Wavelet assumed bests
            if sls.loc[bestmatch, 'f_SLS'] != 0:
                j.results.loc['best', 'overall'] = j.results.loc['best', 'SW']
                j.results.loc['best', 'e_overall'] = j.results.loc['best', 'e_SW']
                j.results.loc['best', 'f_overall'] = 34

            else:
                #See whether SW or SLS has the most well-constrained value
                vals = np.array([sls.loc[bestmatch, 'SLS'], swb])
                e_vals = np.array([sls.loc[bestmatch, 'e_SLS'], e_swb])
                frac = e_vals / vals
                s = np.argmin(frac)

                j.results.loc['best', 'overall'] = vals[s]
                j.results.loc['best', 'e_overall'] = e_vals[s]
                j.results.loc['best', 'f_overall'] = s + 1 + 16

        # No matching results found, Wavelet assumed best
        else:
            j.results.loc['best', 'overall'] = j.results.loc['best', 'SW']
            j.results.loc['best', 'e_overall'] = j.results.loc['best', 'e_SW']
            j.results.loc['best', 'f_overall'] = 34
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

def validator(j):
    """
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
    # Validate LombScargle
    validate_SLS(j)

    # Validate Wavelet VS Lomb Scargle
    validate_WS_vs_SLS(j)

    # Validate ACF vs the 'best' period
    validate_best_vs_ACF(j)

    _safety(j)
