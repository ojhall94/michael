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
from tess_stars2px import tess_stars2px_function_entry as tess_stars2px
from astroquery.mast import Tesscut
from eleanor.utils import SearchError
import tess_cpm

# This hack is required to make the code compatible with Lightkurve
np.float = np.float16

class data_class():
    def __init__(self, janet):
        self.j = janet
        
    def build_local_folders(self):
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

        # Create output folder for target
        if not os.path.exists(f'{self.j.output_path}/{self.j.gaiaid}'):
            if self.j.verbose:
                print(f'Making folder {self.j.output_path}/{self.j.gaiaid}/...')
            os.makedirs(f'{self.j.output_path}/{self.j.gaiaid}')
        else:
            pass

        self.path = f'{os.path.expanduser("~")}/.michael/tesscut/{self.j.gaiaid}/'

    def download_tesscut(self):
        """
        Downloads a 50x50 TESS cutout around the targeted coordinates. This is
        stored in the `~/.michael/tesscut/{target_id}` directory. The cut is done using the 
        `tesscut` Python package.
        """
        coords = SkyCoord(ra = self.j.ra, dec = self.j.dec, unit = (u.deg, u.deg))

        # Check coordinates and sectors pulled up by tesscut.get_sectors
        mdr = Tesscut.get_sectors(coordinates=coords)

        # In rare cases, Tesscut mistakenly identifies a target cut where the target
        # did not fall on silicone. To avoid this, we exclude it from the list here
        # (to avoid a second download), and delete it further down.
        on_silicone = tess_stars2px(0, coords.ra.deg, coords.dec.deg)[3]
        keep = np.ones(len(mdr['sector']))
        for idx, s in enumerate(mdr['sector']):
            if s not in on_silicone:
                keep[idx] = 0
        md = mdr[keep.astype(bool)]

        print(f'Target DR3 ID {self.j.gaiaid} has tesscut data available on MAST for Sectors ' +\
                 ', '.join(str(f) for f in list(md['sector'])))

        # Check whether data is already downloaded, download missing sectors
        if len(glob.glob(f'{os.path.expanduser("~")}/.michael/tesscut/{self.j.gaiaid}/*')) != 0:
            for sfile, sec in zip(md['sectorName'], md['sector']):
                if len(glob.glob(f'{os.path.expanduser("~")}/.michael/tesscut/{self.j.gaiaid}/{sfile}*')) == 1:
                    print(f"Data already downloaded for Sector {sec}.")
                else:
                    print(f"Downloading data for Sector {sec}.")
                    Tesscut.download_cutouts(coordinates = coords, size = 50, sector=sec,
                            path = f'{os.path.expanduser("~")}/.michael/tesscut/{self.j.gaiaid}/')        
        # Download all available sectors if nothing has been downloaded yet
        else:
            Tesscut.download_cutouts(coordinates = coords, size = 50,
                    path = f'{os.path.expanduser("~")}/.michael/tesscut/{self.j.gaiaid}/')

        # There are some instances of TESScut returning two observations in a single sector for a target
        # We delete the entry that does not appear in the `Tesscut.get_sectors` list.
        sfiles = np.sort(glob.glob(f'{self.path}/*'))
        sectornames = md['sectorName']

        for sfile in sfiles:
            if sfile.split('/')[-1].split('_')[0] not in sectornames:
                os.remove(sfile)

        # There are some rare cases of Tesscut downloading cutouts for targets that didnt' fall
        # on silicone. We'll delete those here:
        for s in mdr[~keep.astype(bool)]['sector']:
            strlen = np.floor(np.log10(s)).astype(int)+1
            secstr = 's0000'[:-strlen] + str(s)       
            sfile = glob.glob(f'{self.path}/tess-{secstr}*')
            if len(sfile) > 0:
                os.remove(glob.glob(f'{self.path}/tess-{secstr}*')[0])

    def setup_data(self):
        """
        This function ensures all the available data is downloaded into the expected file structure.
        """
        # Check local file structure.
        self.build_local_folders()

        # Download data (will skip sectors we have already)
        self.download_tesscut()

        # Check which sectors have been downloaded
        sfiles = glob.glob(f'{self.path}/*.fits')
        sort = np.argsort([f.split('/')[-1].split('-')[1][1:] for f in sfiles])  
        self.j.sfiles = np.array(sfiles)[sort].tolist() 

        self.j.nullsectors = np.unique(np.sort([f.split('/')[-1].split('-')[1] for f in self.j.sfiles]))
        self.j.sectorlist = np.sort([int(f[1:]) for f in self.j.nullsectors])

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

    def build_eleanor_lc(self):
        """
        This function constructs an eleanor datum object from a downloaded tesscut TPF.
        It also constructs light curves for raw data, PCA and corner corrected data,
        as well as the out-of-the-box eleanor corrected light curve.
        """
        coords = SkyCoord(ra = self.j.ra, dec = self.j.dec, unit = (u.deg, u.deg))

        stars = eleanor.multi_sectors(coords = coords,
                                sectors = self.j.sectorlist.tolist(),
                                tc = True,
                                post_dir = self.path,
                                tesscut_size = 50)

        # Looping and appending all sectors
        for s, star in zip(self.j.sectorlist, stars):
            datum = eleanor.TargetData(star,
                                        do_pca = True)

            q = datum.quality == 0
            lc = lk.LightCurve(time = datum.time[q], flux = datum.corr_flux[q])

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
        sfiles = []
        for sector in self.j.sectors:
            split = sector.split('-')
            if len(split) > 1:
                sfiles = []
                for s in np.arange(int(split[0]), int(split[1])+1):
                    strlen = np.floor(np.log10(s)).astype(int)+1
                    secstr = 's0000'[:-strlen] + str(s)

                    sfile = glob.glob(f'{self.path}'+
                                        f'*{secstr}*astrocut.fits')

                    if len(sfile) == 0:
                        raise ValueError("No tesscut files could be found for this target.")

                    sfiles.append(sfile[0])

                tpflist = [lk.TessTargetPixelFile(f).cutout([26,26],13) for f in sfiles]
                tpfs = lk.TargetPixelFileCollection(tpflist)

                self.j.void[f'tpfs_{sector}'] = tpfs

                r = SIP(tpfs)
                self.j.void[f'r_{sector}'] = r
                # tess-sip can sometimes introduce major peaks at the ends of
                # the light curve, so we remove these.
                self.j.void[f'rlc_{sector}'] = r['corr_lc'].remove_nans().remove_outliers()

                if detrended:
                    self.j.void[f'rdtlc_{sector}'] = \
                        r['corr_lc'].remove_nans().flatten(window_length = len(r['corr_lc'])).remove_outliers().remove_nans()

            else:
                continue

    def build_unpopular_lc(self):
        """
        This function constructs a `Lightkurve` object output from the
        `tess_cpm` (a.k.a. `unpopular`) technique by Hattori et al. (2021).

        Note that this method does not work particularly well for fainter stars.
        """

        # Set up a standard aperture based on the `eleanor` aperture for a 50x50
        # postcard.
        for sfile, s in zip(self.j.sfiles, self.j.sectorlist):
            cpm = tess_cpm.Source(sfile, remove_bad=True)
            aperture = self.j.void[f'datum_{s}'].aperture
            rowlims = 19 + np.array([np.where(aperture)[0].min(), np.where(aperture)[0].max()])
            collims = 19 + np.array([np.where(aperture)[1].min(), np.where(aperture)[1].max()])
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
