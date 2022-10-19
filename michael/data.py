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
import tess_cpm

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
        if not os.path.exists(f'{self.j.output_path}/{self.j.gaiaid}'):
            if self.j.verbose:
                print(f'Making folder {self.j.output_path}/{self.j.gaiaid}/...')
            os.makedirs(f'{self.j.output_path}/{self.j.gaiaid}')
        else:
            pass

        # Check for existing data
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
        star = eleanor.multi_sectors(coords = coords, sectors = 'all',
                                        tc = True, tesscut_size=50)

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
        sfiles = np.sort(glob.glob(f'{os.path.expanduser("~")}/.eleanor/tesscut/*_{rastr[:(7+step)]}*{decstr[:(7+step)]}_*'))

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

    # def build_stitched_all_lc(self):
    #     """
    #     Deprecated, data no longer needed.
    #
    #     Combine all available light curves of a given sector for the purposes of the ACF.
    #
    #     NOTE: We may want to change the ACF method to sector-by-sector due to changing spots.
    #     """
    #     pls = ['c','raw','pca','corn', 'cpm']
    #
    #     for pl in pls:
    #         all = self.j.void[f'{pl}lc_{self.j.sectorlist[0]}']
    #         for s in self.j.sectorlist[1:]:
    #             all = all.append(self.j.void[f'{pl}lc_{s}'])
    #         self.j.void[f'{pl}lc_all'] = all

    def build_tess_sip_lc(self):
        """
        This function constructs a `Lightkurve` object output from the
        `tess_sip` technique by Hedges et al. (2021).

        Note: this only works on consecutive sectors of data.
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


                    sfiles.append(glob.glob(
                    f'{os.path.expanduser("~")}/.eleanor/tesscut/*-{secstr}-*{rastr[:(6+step)]}*{decstr[:(6+step)]}*')[0])


                tpflist = [lk.TessTargetPixelFile(f).cutout([26,26],13) for f in sfiles]
                tpfs = lk.TargetPixelFileCollection(tpflist)
                r = SIP(tpfs)
                self.j.void[f'r_{sector}'] = r
                # tess-sip can sometimes introduce major peaks at the ends of
                # the light curve, so we remove these.
                self.j.void[f'rlc_{sector}'] = r['corr_lc'].remove_nans().remove_outliers()

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
