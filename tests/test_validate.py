import sys
import pytest
sys.path.append('../')

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from michael import janet
import astropy.units as u

from michael.validate import longest_sectors
from michael.validate import validate_SLS
from michael.validate import validate_SW
from michael.validate import validate_CACF
from michael.validate import validate_ACF
from michael.validate import validate_best
from michael.validate import validate_sectors
from michael.validate import validate_p2p

# Set up mock `janet` with synthetic pass case data
gj = janet('synthetic', 0., 0., output_path = 'tests/data')
gj.sectors = ['0', '1']
gj.sectorlist = ['0', '1']
gsfile = glob.glob('tests/data/synthetic*pass*')[0]
gprot = float(gsfile.split('_')[-2])
gsyn = np.genfromtxt(gsfile)
gj.void['clc_0']  = lightkurve.LightCurve(time = np.arange(0, 27., 0.02),
                                            flux = gsyn)
gj.void['clc_1'] = lightkurve.LightCurve(time = np.arange(0, 27., 0.02),
                                            flux = gsyn)
gj.void['clc_1'] = 1 + ((gj.void['clc_1'] - 1) * 3)
gj.get_rotation(period_range = (3, 6))

def test_longest_sectors():
    gj.sectors += ['2-4', '6-8']
    assert longest_sectors(gj) == ['2-4', '6-8']
    gj.sectors = ['0','1']

def test_validate_p2p():
    valdiate_p2p(gj)
    # Check finite output
    methods = ['SLS', 'SW', 'CACF','ACF']
    for m in methods:
        assert np.isfinite(gj.results.loc['0', f'p2p_{m}'])
        assert np.isfinite(gj.results.loc['0', f'snr_{m}'])
        assert np.isfinite(gj.results.loc['0', f'f_p2p_{m}'])

def test_validate_SLS():
    validate_SLS(gj)

    # Check all results have been saved
    for pref in ['','e_','h_','f_','p2p_']:
        assert not np.isnan(gj.results.loc['best',f'{pref}SLS'])
    assert type(gj.results.loc['best','s_SLS']) == np.ndarray

    # Check best value is for highest SNR

def test_validate_SW():
    return 0

def test_validate_CACF():
    return 0

def test_validate_ACF():
    return 0

def test_validate_best():
    return 0

def test_validate_sectors():
    return 0
