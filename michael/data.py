"""
The `data' class contains functions for `janet' tp call. This includes mainly
Eleanor light curves, but also things like auxiliary APOGEE data.
"""

import eleanor
import numpy as np
import glob
from astropy.coordinates import SkyCoord
from astropy import units as u
import lightkurve as lk
import eleanor

class data_class():
    def __init__(self, janet):
        self.j = janet

    def check_eleanor_setup(self):
        """ Check the Eleanor setup.
        This function checks that Eleanor data has already been prepared for the
        target star. If it has not, the function creates a directory and
        downloads and saves the data.
        """

        # Create matching data folders
        if not os.path.exists(f'{self.j.output_path}/{self.j.gaia_id}'):
            print(f'Making folder {self.j.output_path}/{self.j.gaia_id}/...')
            os.makedirs(f'{self.j.output_path}/{self.j.gaia_id}')
        else:
            pass

        # Check for existing data
        if len(glob.glob(f'{self.j.output_path}/{self.j.gaia_id}/*.fits')) > 0:
            print(f'Already have data loaded for Gaia ID {self.j.gaiaid}.')
            print(f'If you want to check for new data, run `janet.Update()`.')
        else:
            self.download_eleanor_data()

        # Check which sectors have been downloaded
        j.sfiles = glob.glob(f'{self.j.output_path}/{self.gaiaid}/*.fits')
        j.sectors = np.sort([int(f.split('_')[-1][:-5]) for f in self.sfiles])

    def download_eleanor_data(self):
        """ Download Eleanor data.
        Data may not always be available due to not being observed by TESS, or
        errors in the download (which Im still trying to solve).
        """

        coords = SkyCoord(ra = self.j.ra, dec = self.j.dec, unit = (u.deg, u.deg))
        star = eleanor.multi_sectors(coords = coords, sectors = 'all')

        for s in self.star:
            try:
                datum = eleanor.TargetData(s)
                datum.save(output_fn = f'lc_sector_{s.sector}.fits',
                        directory = f'{self.j.output_path}/{self.j.gaia_id}/')
            except TypeError:
                print(f'There is some kind of Eleanor error for Sector {s.sector}')
                print('Try running eleanor.Update(), or raise an issue on the '
                        'Eleanor GitHub!')
                print('Moving on the next sector... \n')

    def build_eleanor_lc(self):
        """
        This function constructs a `Lightkurve` object from downloaded `eleanor`
        light curves. It also stores the light curve for each object it reads in,
        a well as the full Eleanor Target Pixel File data.
        """

        datum = eleanor.TargetData(eleanor.Source(fn=f'lc_sector_{j.sectors[0]}.fits',
                                                 fn_dir=f'{self.j.output_path}/{self.j.gaiaid}/'))

        q = datum.quality == 0
        lc = lk.LightCurve(datum.time[q], datum.corr_flux[q])
        self.clc = lc.normalize().remove_nans().remove_outliers()

        # Store the datum and light curve
        self.j.void[f'datum_{sectors[0]}'] = datum
        self.j.void[f'clc_{sectors[0]}'] = self.clc

        # Looping and appending all sectors
        for s in j.sectors[1:]:
            datum = eleanor.TargetData(eleanor.Source(fn=f'lc_sector_{s}.fits',
                                                     fn_dir=f'{self.j.output_path}/{self.j.gaiaid}/'))
            q = datum.quality == 0
            lc = lk.LightCurve(datum.time[q], datum.corr_flux[q])
            self.clc = self.clc.append(lc.normalize().remove_nans().remove_outliers())

            # Store the datum and light curve
            self.j.void[f'datum_{s}'] = datum
            self.j.void[f'clc_{s}'] = lc.normalize().remove_nans().remove_outliers()

        self.j.void[f'clc_all'] = self.clc
