"""
The `janet' class globally handles all of `michael''tasks. It calls all classes
in the package, e.g.:
- Downloading data using Eleanor
- Generating the light curve
- Estimating rotation periods
- Consolidating rotation periods
- Producing plots and saving data.

The exact steps used (e.g. which rotation period techniques to apply) can be
selected when the package is called. It also goes looking in APOGEE to find
temperatures and metallicities for your star, if you like.

Unlike the real Janet, this janet does not know the answers to everything in
the Universe (if only).
"""

import os, warnings
import pandas as pd
import numpy as np
import astropy.units as u

from .data import data_class
from .methods import simple_astropy_lombscargle, simple_wavelet
from .plotting import plot

class janet():
    """ Class managing all i/o for the `michael' package.

    Examples
    --------
    from michael import janet
    j = janet.boot(df).get_rotation()
    j.view()

    Parameters
    ----------


    Attributes
    ----------

    """

    def __init__(self, gaiaid, ra, dec, output_path=None, verbose=True):
            self.gaiaid = gaiaid
            self.ra = ra
            self.dec = dec
            self.results = pd.DataFrame()
            self.output_path = output_path
            self.verbose = verbose
            self.void = {}
            self.gaps = False

    def prepare_data(self):
        """
        This function calls the `data_class()`, which prepares the `eleanor`
        light curves.
        """
        self.data = data_class(self)
        self.data.check_eleanor_setup()
        self.data.build_eleanor_lc()

    def get_rotation(self, period_range = (0.2, 12.)):
        """
        This needs some polish to get multiple methods working.
        """
        sectorlist = list(self.sectors)
        if not self.gaps:
            sectorlist += ['all']

        # Loop over all sectors.
        if len(self.sectors) == 1:
            simple_astropy_lombscargle(self, sector='all', period_range = period_range)

        else:
            for sector in sectorlist:
                simple_astropy_lombscargle(self, sector = sector, period_range = period_range)

        simple_wavelet(self, period_range = period_range)

    def validate_rotation(self):
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
        """
        # Validate LombScargle
        if np.isfinite(self.results.loc['all', 'SLS']):
            # If there is a LS value for 'all', consider this the default best
            self.results.loc['best', 'SLS'] = self.results.loc['all', 'SLS']
            self.results.loc['best', 'e_SLS'] = self.results.loc['all', 'e_SLS']
            self.results.loc['best', 's_SLS'] = 'all'
            self.results.loc['best', 'f_SLS'] = self.results.loc['all', 'f_SLS']

        else:
            # If onlys single-sector cases are available, pick the value with
            # the lowest fractional uncertainty on an unflagged value
            s = self.results['f_SLS'] == 0
            if len(self.results[s]) > 0:
                sigfrac = self.results[s]['e_SLS'] / self.results[s]['SLS']

            # It may be the case that there are only flagged values. In this
            # case, ignore the flags
            else:
                sigfrac = self.results['e_SLS'] / self.results[s]['SLS']

            idx = sigfrac.idxmin()
            self.results.loc['best', 'SLS'] = self.results.loc[idx, 'SLS']
            self.results.loc['best', 'e_SLS'] = self.results.loc[idx, 'e_SLS']
            self.results.loc['best', 's_SLS'] = str(int(idx))
            self.results.loc['best', 'f_SLS'] = self.results.loc[idx, 'f_SLS']

        # Validate Wavelet vs LombScargle
        self.results.loc['best', 'SW'] = self.results.loc['all', 'SW']
        self.results.loc['best', 'e_SW'] = self.results.loc['all', 'e_SW']

        # Check to see if they agree within 1 sigma
        best = self.results.loc['best']

        # If they agree, then pick the one with the best fractional uncertainty
        if np.diff(best[['SLS', 'SW']]) < np.sum(best[['e_SLS', 'e_SW']]):
            frac = best[['e_SLS', 'e_SW']].values /  best[['SLS', 'SW']].values
            s = np.argmin(frac)
            self.results.loc['best', 'overall'] = best[['SLS', 'SW'][s]]
            self.results.loc['best', 'e_overall'] = best[['e_SLS', 'e_SW'][s]]
            self.results.loc['best', 'f_overall'] = s + 1

        # If they disagree, see if there are any matches with another sector
        else:
            if len(self.sectors) >= 2:
                sls = self.results.loc[self.sectors,['SLS', 'e_SLS', 'f_SLS']]
            else:
                sls = self.results.loc['all',['SLS', 'e_SLS', 'f_SLS']]

            swb = self.results.loc['best', 'SW']
            e_swb = self.results.loc['best', 'e_SW']

            # An agreement within 1 Sigma has been found
            if np.any(np.abs(sls.SLS - swb) - (e_swb + sls.e_SLS) < 0):
                match = sls[np.abs(sls.SLS - swb) - (e_swb + sls.e_SLS) < 0]
                frac = match.e_SLS / match.SLS
                bestmatch = frac.idxmin()

                # No matching results found without a flag, Wavelet assumed bests
                if sls.loc[bestmatch, 'f_SLS'] != 0:
                    self.results.loc['best', 'overall'] = self.results.loc['best', 'SW']
                    self.results.loc['best', 'e_overall'] = self.results.loc['best', 'e_SW']
                    self.results.loc['best', 'f_overall'] = 34

                else:
                    #See whether SW or SLS has the most well-constrained value
                    vals = np.array([sls.loc[bestmatch, 'SLS'], swb])
                    e_vals = np.array([sls.loc[bestmatch, 'e_SLS'], e_swb])
                    frac = e_vals / vals
                    s = np.argmin(frac)

                    self.results.loc['best', 'overall'] = vals[s]
                    self.results.loc['best', 'e_overall'] = e_vals[s]
                    self.results.loc['best', 'f_overall'] = s + 1 + 16

            # No matching results found, Wavelet assumed best
            else:
                self.results.loc['best', 'overall'] = self.results.loc['best', 'SW']
                self.results.loc['best', 'e_overall'] = self.results.loc['best', 'e_SW']
                self.results.loc['best', 'f_overall'] = 34


    def view(self):
        """
        Calls `michael`'s plotting functions.

        Eventually there will be some extra kwargs to add
        """
        plot(self)


    def run(self):
        self.prepare_data()
        self.get_rotation()
        self.validate_rotation()
        self.view()

    def __repr__(self):
        repr = "Hi there! I'm Janet 🌵\n"

        if len(self.void) == 0:
            repr += "I don't have any data or results in storage right now. Try running `janet.prepare_data()` to get started! ✨"
        return repr


    @staticmethod
    def boot(df, index=0):
        """
        Sets up Janet quickly.
        """
        return janet(
            gaiaid = df.loc[index, 'source_id'], ra = df.loc[index, 'ra'], dec = df.loc[index, 'dec'],
            output_path = '/Users/oliver hall/Research/unicorn/data', verbose=True
        )
