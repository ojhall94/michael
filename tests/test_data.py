import sys
import pytest
import os
import shutil

sys.path.append('../')

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from michael import janet
from michael.data import data_class
import astropy.units as u
import glob
import lightkurve

def test_eleanor_setup():
    # Using a known rotator from the Gaia catalogue
    gaiaid = 4984094970441940864
    ra = 20.457083
    dec = -42.022861

    j = janet(gaiaid, ra, dec, output_path = 'tests/data')
    data = data_class(j)

    # Perform initial clean pass
    shutil.rmtree(f'tests/data/{gaiaid}')

    # This call also tests download_eleanor_data().
    data.check_eleanor_setup()

    # Confirm the folders have been made
    assert os.path.exists(f'tests/data/{gaiaid}')

    # Assert files exist - at time of writing there are at least 4 sectors
    sfiles = glob.glob(f'tests/data/{gaiaid}/*.fits')
    assert len(sfiles) >= 4
    assert len(j.sfiles) >= 4
    assert len(j.sectorlist) >= 4
    assert len(j.sectors) >= 2

def test_eleanor_build():
    # Using a known rotator from the Gaia catalogue
    gaiaid = 4984094970441940864
    ra = 20.457083
    dec = -42.022861

    j = janet(gaiaid, ra, dec, output_path = 'tests/data')
    data = data_class(j)
    data.check_eleanor_setup()

    # Assert function creates relevant lightcurves as lightkurve objects
    data.build_eleanor_lc()
    for pl in ['c','raw','pca','corn']:
        for s in list(j.sectors) + list(j.sectorlist):
            assert f'{pl}lc_{s}' in list(j.void)
            assert type(j.void[f'{pl}lc_{s}']) == lightkurve.LightCurve

def test_tess_sip_build():
    # Using a known rotator from the Gaia catalogue
    gaiaid = 4984094970441940864
    ra = 20.457083
    dec = -42.022861

    j = janet(gaiaid, ra, dec, output_path = 'tests/data')
    data = data_class(j)
    data.check_eleanor_setup()

    # Assert tess-sip generates lightkurve LightCurve object
    data.build_tess_sip_lc()
    for s in list(j.sectors):
        assert f'r_{s}' in list(j.void)
        assert f'rlc_{s}' in list(j.void)
        assert type(j.void[f'rlc_{s}']) == lightkurve.lightcurve.TessLightCurve

def test_unpopular_build():
    # Using a known rotator from the Gaia catalogue
    gaiaid = 4984094970441940864
    ra = 20.457083
    dec = -42.022861

    j = janet(gaiaid, ra, dec, output_path = 'tests/data')
    data = data_class(j)
    data.check_eleanor_setup()
    data.build_eleanor_lc() # Unpopular relies on the eleanor aperture

    # Assert tess-sip generates lightkurve LightCurve object
    data.build_unpopular_lc()
    for s in j.sectorlist:
        assert f'cpm_{s}' in list(j.void)
    for s in list(j.sectors) + list(j.sectorlist):
        assert f'cpmlc_{s}' in list(j.void)
        assert type(j.void[f'cpmlc_{s}']) == lightkurve.LightCurve
