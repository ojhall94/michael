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

    def view(self):
        """
        Calls `michael`'s plotting functions.

        Eventually there will be some extra kwargs to add
        """
        plot(self)


    def run(self):
        self.prepare_data()
        self.get_rotation()
        #self.validate_rotation()
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
