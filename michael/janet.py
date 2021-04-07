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
from .SLS import simple_astropy_lombscargle

class janet():
    """ Class managing all i/o for the `michael' package.

    Examples
    --------



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
            self.verbose = verbose
            self.void = {}

    def prepare_data(self):
        """
        This function calls the `data_class()`, which prepares the `eleanor`
        light curves.
        """
        self.data = data_class(self)
        self.data.check_eleanor_setup()
        self.data.build_eleanor_lc(self)

    def get_rotation(self):
        """
        This needs some polish to get multiple methods working.
        """
        #
        # # Loop over all sectors.
        # # This runs even if a star has only a single sector
        # if sectors == 'complete':
        #     if len(list(self.sectors)) > 1:
        #         for sector in list(self.sectors) + ['all']:
        #             self.build_eleanor_lc(sector = sector)
        #             self.simple_astropy_lombscargle(sector = sector)
        #     else:
        #         self.build_eleanor_lc(sector = 'all')
        #         self.simple_astropy_lombscargle(sector = 'all')
        #
        # # One run only
        # else:
        #     self.build_eleanor_lc(sector = sectors)
        #     self.simple_astropy_lombscargle(sector = sectors)
        #
        # # Validate the results
        # self.validate_rotation()
        #
        # self.output()


    def run(self):
        self.prepare_data()

    @staticmethod
    def boot(df):
        """
        Sets up Janet quickly.
        """
        return Janet(
            gaiaid = df['source_id'], ra = df['ra'], dec = df['dec'], verbose=True
        )
