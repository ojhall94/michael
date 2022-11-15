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
import eleanor
import glob
import astropy.units as u

from .data import data_class
from .methods import *
from .validate import validator
from .plotting import plot
from .utils import _decode, _safety, longest_sectors
from .prior import priorclass

pipelines = {'eleanor' : 'c',
             'eleanor-raw' : 'raw',
             'eleanor-pca' : 'pca',
             'eleanor-corner' : 'corn',
             'unpopular' : 'cpm',
             'tess-sip': 'r'}

class janet():
    """ Class managing all i/o for the `michael' package.

    Pipeline options are:
    - eleanor
    - eleanor-raw
    - eleanor-pca
    - eleanor-corner
    - unpopular
    - tess-sip

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

    def __init__(self, gaiaid, ra = None, dec = None, output_path = None,
                pipeline = 'eleanor', update = False, verbose = True):
            self.gaiaid = gaiaid
            self.ra = ra
            self.dec = dec
            self.results = pd.DataFrame()
            self.output_path = output_path
            self.verbose = verbose
            self.void = {}
            self.update = update
            # self.override=False
            # self.use_prior = use_prior
            # self.obs = obs
            # self.prot_prior = np.array([np.nan, np.nan, np.nan])
            # self.samples = None
            if pipeline not in pipelines:
                raise ValueError("Requested pipeline not available, defaulting"+
                                " to eleanor pipeline")
                pipeline = 'eleanor'

            self.pipeline = pipeline
            self.pl = pipelines[pipeline]

            # if use_prior and obs is None:
            #     raise ValueError('When using the prior function you must provide '
            #                      'observables as input.')

    def prepare_data(self):
        """
        This function calls the `data_class()`, which prepares the `eleanor`
        light curves.
        """
        self.data = data_class(self)
        self.data.check_eleanor_setup()
        self.data.build_eleanor_lc()

        if self.pipeline == 'unpopular':
            self.data.build_unpopular_lc()

        if self.pipeline == 'tess-sip':
            self.data.build_tess_sip_lc()

        # self.data.build_stitched_all_lc()

    def reset_data(self):
        """
        Calling this function serves to delete all stored eleanor data
        associated with a given target.

        This is to be used in case there is a data corruption issue, or an
        inconsistency between machines on which data is available.
        """
        rastr = str(self.ra)
        step = len(rastr.split('.')[0])
        decstr = str(self.dec)
        step = len(decstr.split('.')[0])
        sfiles = np.sort(glob.glob(f'{os.path.expanduser("~")}/.eleanor/tesscut/*{rastr[:(6+step)]}*{decstr[:(6+step)]}*'))
        for s in sfiles:
            os.system("rm "+s)

    # def flux_override(self, time, flux):
    #     """
    #     Michael is intended for use with `eleanor` light curves only. However for
    #     testing purposes, this `flux_override()` command allows input of a custom
    #     light curve.
    #
    #     After calling this command, the user should call `get_rotation()`,
    #     `validate_rotation()` and `view()` manually.
    #
    #     This lone light curve is treated as if it's an 'all' sectors light curve.
    #
    #     Parameters
    #     ----------
    #     time: ndarray
    #         The time values in units of days.
    #
    #     flux: ndarray
    #         The flux values in any units.
    #     """
    #     self.sectors = np.array(['all'])
    #     self.gaps = False
    #     self.override = True
    #
    #     # Create matching data folders
    #     if not os.path.exists(f'{self.output_path}/{self.gaiaid}'):
    #         print(f'Making folder {self.output_path}/{self.gaiaid}/...')
    #         os.makedirs(f'{self.output_path}/{self.gaiaid}')
    #     else:
    #         pass
    #
    #     lc = lk.LightCurve(time = time, flux = flux)
    #     clc = lc.normalize().remove_nans().remove_outliers()
    #     self.void['datum_all'] = None
    #     self.void['clc_all'] = clc

    def get_rotation(self, period_range = (0.2, 27.)):
        """
        This needs some polish to get multiple methods working.
        """
        # Assert the period range is positive and not longer than your longest
        # sector. Raise a warning if it is over half your longest sector.
        if any(np.array(period_range) <= 0):
            raise ValueError("Please input a lower period limit > 0.")

        if period_range[1] <= period_range[0]:
            raise ValueError("It looks like you've got your period limits mixed up!")

        longest = longest_sectors(self)
        longest = longest[0]
        if len(longest.split('-')) == 1:
            maxlen = 27
        else:
            split = longest.split('-')
            maxlen = 27*(int(split[1]) - int(split[0])+1)

        if period_range[1] > maxlen/2:
            warnings.warn(UserWarning("Your upper period limit is longer than half your "+
                            "longest set of consecutive TESS sectors. You'll "+
                            "be more prone to harmonics."))

        if period_range[1] > maxlen:
            print(f'Error on {self.gaiaid}\n')
            raise ValueError("Your upper period limit is longer than your "+
                            "longest set of consecutive TESS sectors.")

        # Only look at consecutive sectors if using tess-sip
        if self.pipeline == 'tess-sip':
            lim = self.sectors[[len(a) > 2 for a in self.sectors]]
            self.sectors = self.sectors[self.sectors == lim]
            self.sectorlist = []
            for sector in self.sectors:
                split = sector.split('-')
                self.sectorlist += list(np.arange(int(split[0]), int(split[1])))

        sectorlist = list(self.sectors)

        # TO DO: Set period range based on longest baseline
        self.period_range = period_range

        # Loop over all sectors.
        for sector in sectorlist:
            simple_astropy_lombscargle(self, sector = sector, period_range = period_range)
            simple_wavelet(self, sector = sector, period_range = period_range)
            composite_ACF(self, sector = sector, period_range = period_range)
            simple_ACF(self, sector = sector, period_range = period_range)

    def validate_rotation(self):
        validator(self)

    def view(self):
        """
        Calls `michael`'s plotting functions.

        Eventually there will be some extra kwargs to add
        """
        plot(self)

    def decode(self, flag):
        """
        Converts the 'f_overall' flag into human-readable strings.

        This function takes the 'f_overall' integer value in the results table.
        This does not work for the flag associated with any of the individual
        rotation estimates (such as the 'f_SLS' flag).

        If `michael` is run using the `verbose=True` kwarg, the decoded flag
        will be printed at the end of the run.
        """
        if (type(flag) is not int) & (type(flag) is not float) & (type(flag) is not np.int64):
            raise ValueError("Please input an integer flag.")

        if flag >= 32:
            print("Our flags don't go this high. Please see the `validator()`"+
                    " function docstring for more information")
        else:
            print(f'\n------ Decoding Overall Period Flag {int(flag)} ------')
            print(_decode(flag))
            print('No other flags raised. \n')

    def run(self, period_range = (0.2, 27.)):
        self.prepare_data()

        self.get_rotation(period_range = period_range)
        self.validate_rotation()

        if self.verbose:
            self.view()

        if self.verbose:
            self.decode(self.results.loc['best', 'f_overall'].astype(int))

        _safety(self)

    def __repr__(self):
        repr = "Hi there! I'm Janet 🌵\n"

        if len(self.void) == 0:
            repr += "I don't have any data or results in storage right now. Try running `janet.prepare_data()` to get started! ✨"
        return repr


    @staticmethod
    def boot(df, index, output_path = '/Users/oliver hall/Research/unicorn/data/eleanor',
            pipeline = 'eleanor',update = False):
        """
        Sets up Janet quickly.
        """
        return janet(
            gaiaid = df.loc[index, 'source_id'], ra = df.loc[index, 'ra'], dec = df.loc[index, 'dec'],
            output_path = output_path, pipeline=pipeline, update= update, verbose=True
        )
