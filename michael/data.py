"""
The `data' class contains functions for `janet' tp call. This includes mainly
Eleanor light curves, but also things like auxiliary APOGEE data.
"""

import eleanor
import os
import numpy as np
import glob
from tess_sip import SIP
from astropy.coordinates import SkyCoord
from astropy import units as u
import lightkurve as lk
import eleanor
from astroquery.mast import Tesscut
from eleanor.utils import SearchError
import tess_cpm

class data_class():
    def __init__(self, janet):
        self.j = janet

    def check_local_setup(self):
        """
        Checks the local setup. If this is the first time downloading data
        using michael, it will set up the `~/.michael/tesscut/`` directories, with
        a subfolder for the target using its Gaia DR3 ID.
        """

        # Create michael folder
        if not os.path.exists(f'{os.path.expanduser("~")}/.michael'):
            if self.j.verbose:
                print(f'Making folder {os.path.expanduser("~")}/.michael/...')
            os.makedirs(f'{os.path.expanduser("~")}/.michael')

        # Create tesscut folder
        if not os.path.exists(f'{os.path.expanduser("~")}/.michael/tesscut'):
            if self.j.verbose:
                print(f'Making folder {os.path.expanduser("~")}/.michael/tesscut/...')
            os.makedirs(f'{os.path.expanduser("~")}/.michael/tesscut')

        # Create data folder for target
        if not os.path.exists(f'{os.path.expanduser("~")}/.michael/tesscut/{self.j.gaiaid}'):
            if self.j.verbose:
                print(f'Making folder {os.path.expanduser("~")}/.michael/tesscut/{self.j.gaiaid}/...')
            os.makedirs(f'{os.path.expanduser("~")}/.michael/tesscut/{self.j.gaiaid}')

    def download_tesscut(self):
        """
        Downloads a 50x50 TESS cutout around the targeted coordinates. This is
        stored in the `~/.michael/tesscut/{target_id}` directory.
        """
        coords = SkyCoord(ra = self.j.ra, dec = self.j.dec, unit = (u.deg, u.deg))

        # Check coordinates and sectors pulled up by tesscut.get_sectors
        reported_sectors = list(md['sector'])
        print(f'Target DR3 ID {self.j.gaiaid} has data available for sectors '.join(sector for sector in reported_sectors))

        hdulist = Tesscut.get_cutouts(coordiantes = coords, size = 50)

        # Download tesscut, store in local path
        ## At later stage allow setup of an alternative folder?

    def check_eleanor_setup(self):
        """
        This function is being deprecated/altered for apure eleanor focus.

        Check the Eleanor setup.
        This function checks that Eleanor data has already been prepared for the
        target star. If it has not, the function creates a directory and
        downloads and saves the data.
        """
        # Check for existing data, unless a full update is demanded
        if self.j.update:
            self.download_eleanor_data()

        else:
            if len(glob.glob(f'{self.j.output_path}/{self.j.gaiaid}/*.fits')) > 0:
                if self.j.verbose:
                    print(f'Already have data downloaded for Gaia ID {self.j.gaiaid}.')
            else:
                self.download_eleanor_data()

        # Check which sectors have been downloaded
        self.j.sfiles = glob.glob(f'{self.j.output_path}/{self.j.gaiaid}/*.fits')
        self.j.sectorlist = np.unique(np.sort([int(f.split('_')[-1][:-5]) for f in self.j.sfiles]))

        # Create sectorlabels that connect sectors together.
        sectors = []
        step = 0
        for start, sector in enumerate(self.j.sectorlist):
            if step > start:
                continue
            else:
                # Start the label
                label = f"{sector}"
                # Check all consecutive sectors if they follow this one
                for step in np.arange(start+1, len(self.j.sectorlist)):
                    diff = np.diff([self.j.sectorlist[step-1],self.j.sectorlist[step]])
                    # Next step is 1 ahead of the previous sector, continue
                    if diff == 1:
                        #Check whether this is the final entry
                        if step == len(self.j.sectorlist)-1:
                            label += f"-{self.j.sectorlist[step]}"
                            step += 1 # End the loop
                            break
                        else:
                            continue

                    # Next step is a leap ahead
                    elif step-1 != start:
                        label += f"-{self.j.sectorlist[step-1]}"
                        break

                    else:
                        break
                sectors.append(label)
        self.j.sectors = np.array(sectors)

    def download_eleanor_data(self):
        """ Download Eleanor data.
        Data may not always be available due to not being observed by TESS, or
        errors in the download (which Im still trying to solve).

        Eleanor will download the data as a tesscut of 50x50 pixels, which is
        required for the `unpopular` method. For other methods, this is reduced
        to a 13x13 postcard.
        """

        coords = SkyCoord(ra = self.j.ra, dec = self.j.dec, unit = (u.deg, u.deg))

        try:
            star = eleanor.multi_sectors(coords = coords, sectors = 'all',
                                        tc = True, tesscut_size=50)
        except SearchError:
            print(f'Eleanor thinks your target has not been observed by TESS')
            print('If you believe this to be in error, please get in touch.')
            print('Exiting `michael`.')
            raise

        for s in star:
            try:
                datum = eleanor.TargetData(s, do_pca = True)
                datum.save(output_fn = f'lc_sector_{s.sector}.fits',
                        directory = f'{self.j.output_path}/{self.j.gaiaid}/')
            except TypeError:
                print(f'There is some kind of Eleanor error for Sector {s.sector}')
                print('Try running eleanor.Update(), or raise an issue on the '
                        'Eleanor GitHub!')
                print('Moving on the next sector... \n')

            except ValueError:
                print(f'There may be an issue where eleanor is detecting multiple '
                        'instances of a single sector. Skipping this sector.' )

        # Eliminate the secondary duplicate sector from the TESSCuts
        rastr = str(self.j.ra)
        step = len(rastr.split('.')[0])
        decstr = str(self.j.dec)
        step = len(decstr.split('.')[0])
        sfiles = np.sort(glob.glob(f'{os.path.expanduser("~")}/.eleanor/tesscut/*_{rastr[:(6+step)]}*{decstr[:(6+step)]}_*'))

        slabels = []
        for idx in range(len(sfiles)):
            slabels.append(sfiles[idx].split('-')[1])
        keep = np.unique(slabels, return_index=True)[1]
        for idx, sfile in enumerate(sfiles):
            if idx not in keep:
                os.remove(sfile)

    def build_eleanor_lc(self):
        """
        This function constructs a `Lightkurve` object from downloaded `eleanor`
        light curves. It also stores the light curve for each object it reads in,
        a well as the full Eleanor Target Pixel File data.
        """
        # Looping and appending all sectors
        for s in self.j.sectorlist:
            datum = eleanor.TargetData(eleanor.Source(fn=f'lc_sector_{s}.fits',
                                                     fn_dir=f'{self.j.output_path}/{self.j.gaiaid}/'),
                                                     do_pca = True)
            q = datum.quality == 0
            lc = lk.LightCurve(time = datum.time[q], flux = datum.corr_flux[q])
            # self.clc = self.clc.append(lc.normalize().remove_nans().remove_outliers())

            # Store the datum and light curve
            self.j.void[f'datum_{s}'] = datum
            self.j.void[f'clc_{s}'] = lc.normalize().remove_nans().remove_outliers()

            # Save additional format light curves
            ## Raw light curve from eleanor aperture
            self.j.void[f'rawlc_{s}'] = lk.LightCurve(time = datum.time[q], flux = datum.raw_flux[q])
            self.j.void[f'rawlc_{s}'] = self.j.void[f'rawlc_{s}'].normalize().remove_nans().remove_outliers()

            ## PCA light curve from eleanor aperture
            self.j.void[f'pcalc_{s}'] = lk.LightCurve(time = datum.time[q], flux = datum.pca_flux[q])
            self.j.void[f'pcalc_{s}'] = self.j.void[f'pcalc_{s}'].normalize().remove_nans().remove_outliers()

            ## corner correction light curve from eleanor
            cf = eleanor.TargetData.corrected_flux(datum, flux=datum.raw_flux, regressors='corner')
            self.j.void[f'cornlc_{s}'] = lk.LightCurve(time = datum.time[q], flux = cf[q])
            self.j.void[f'cornlc_{s}'] = self.j.void[f'cornlc_{s}'].normalize().remove_nans().remove_outliers()


        # Combine consecutive lightcurves
        pls = ['c','raw','pca','corn']
        for pl in pls:
            for s in self.j.sectors:
                if len(s.split('-')) > 1:
                    sectors = np.arange(int(s.split('-')[0]), int(s.split('-')[-1])+1)

                    combo = self.j.void[f'{pl}lc_{sectors[0]}']

                    for i in sectors[1:]:
                        combo = combo.append(self.j.void[f'{pl}lc_{i}'])

                    self.j.void[f'{pl}lc_{s}'] = combo

    def build_tess_sip_lc(self, detrended=False):
        """
        This function constructs a `Lightkurve` object output from the
        `tess_sip` technique by Hedges et al. (2021).

        Note: this only works on consecutive sectors of data.

        If `detrended = True`, the `tess-sip` signal is smoothed with a long
        baseline filter.

        """
        rastr = str(self.j.ra)
        step = len(rastr.split('.')[0])
        decstr = str(self.j.dec)
        step = len(decstr.split('.')[0])

        sfiles = []
        for sector in self.j.sectors:
            split = sector.split('-')
            if len(split) > 1:
                sfiles = []
                for s in np.arange(int(split[0]), int(split[1])+1):
                    strlen = np.floor(np.log10(s)).astype(int)+1
                    secstr = 's0000'[:-strlen] + str(s)

                    sfile = glob.glob(f'{os.path.expanduser("~")}/.eleanor/tesscut/*-{secstr}-*{rastr[:(6+step)]}*{decstr[:(6+step)]}*')

                    if len(sfile) == 0:
                        sfile = glob.glob(f'{os.path.expanduser("~")}/.eleanor/tesscut/*{rastr[:(4+step)]}*{decstr[:(4+step)]}*')
                        if len(sfile) == 0:
                            raise ValueError("No tesscut files could be found for this target.")

                    sfiles.append(sfile[0])

                tpflist = [lk.TessTargetPixelFile(f).cutout([26,26],13) for f in sfiles]
                tpfs = lk.TargetPixelFileCollection(tpflist)
                r = SIP(tpfs)
                self.j.void[f'r_{sector}'] = r
                # tess-sip can sometimes introduce major peaks at the ends of
                # the light curve, so we remove these.
                self.j.void[f'rlc_{sector}'] = r['corr_lc'].remove_nans().remove_outliers()

                if detrended:
                    self.j.void[f'rdtlc_{sector}'] = \
                        r['corr_lc'].remove_nans().flatten(window_length = len(r['corr_lc'])).remove_outliers()

            else:
                continue

    def build_unpopular_lc(self):
        """
        This function constructs a `Lightkurve` object output from the
        `tess_cpm` (a.k.a. `unpopular`) technique by Hattori et al. (2021).

        Note that this method does not work particularly well for fainter stars.
        """
        rastr = str(self.j.ra)
        step = len(rastr.split('.')[0])
        decstr = str(self.j.dec)
        step = len(decstr.split('.')[0])
        sfiles = np.sort(glob.glob(f'{os.path.expanduser("~")}/.eleanor/tesscut/*{rastr[:(6+step)]}*{decstr[:(6+step)]}*'))
        coords = SkyCoord(ra = self.j.ra, dec = self.j.dec, unit = (u.deg, u.deg))

        if len(sfiles) == 0:
            sfiles = np.sort(glob.glob(f'{os.path.expanduser("~")}/.eleanor/tesscut/*{rastr[:(4+step)]}*{decstr[:(4+step)]}*'))
            if len(sfiles) == 0:
                raise ValueError("No tesscut files could be found for this target.")

        if len(sfiles) < len(self.j.sectorlist):
            raise ValueError("There are more sectors available than have been "+
                            "loaded into the sectorlist. Reset the data.")

        # Set up a standard aperture based on the `eleanor` aperture for a 50x50
        # postcard.
        for sfile, s in zip(sfiles, self.j.sectorlist):
            cpm = tess_cpm.Source(sfile, remove_bad=True)
            aperture = self.j.void[f'datum_{s}'].aperture
            rowlims = 20 + np.array([np.where(aperture)[0].min(), np.where(aperture)[0].max()])
            collims = 20 + np.array([np.where(aperture)[1].min(), np.where(aperture)[1].max()])
            cpm.set_aperture(rowlims = rowlims, collims = collims)

            # We use 200 predictors for a stamp of this size. This is a rough
            # guesstimate from trial-and-error, but seems to work well.
            cpm.add_cpm_model(exclusion_size=6, n=200,
                predictor_method = "similar_brightness")
            cpm.set_regs([0.1])
            cpm.holdout_fit_predict(k=100, verbose=False)

            # Save corrected flux as a lightcurve object for this sector
            flux = cpm.get_aperture_lc(data_type="cpm_subtracted_flux",
                                        weighting='median')
            self.j.void[f'cpmlc_{s}'] = lk.LightCurve(time = cpm.time, flux = flux).remove_nans().remove_outliers() + 1.
            self.j.void[f'cpm_{s}'] = cpm

        # And now we append the sectors that are consecutive
        for s in self.j.sectors:
            if len(s.split('-')) > 1:
                sectors = np.arange(int(s.split('-')[0]), int(s.split('-')[-1])+1)

                cpmlc = self.j.void[f'cpmlc_{sectors[0]}']

                for i in sectors[1:]:
                    cpmlc = cpmlc.append(self.j.void[f'cpmlc_{i}'])

                self.j.void[f'cpmlc_{s}'] = cpmlc
