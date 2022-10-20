import sys
import pytest
sys.path.append('../')

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from michael import janet
import astropy.units as u
import glob
import lightkurve

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
    validate_p2p(gj)
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
    assert type(gj.results.loc['best','s_SLS']) == str

    # Check best value is for highest SNR, which is for sector 1
    assert gj.results.loc['best', 's_SLS'] == '1'

def test_validate_SW():
    validate_SW(gj)

    # Check all results have been saved
    for pref in ['','e_','h_','p2p_']:
        assert not np.isnan(gj.results.loc['best',f'{pref}SW'])
    assert type(gj.results.loc['best','s_SW']) == str

    # Check best value is for highest SNR, which is for sector 0
    assert gj.results.loc['best', 's_SW'] == '0'

def test_validate_CACF():
    validate_CACF(gj)

    # Check all results have been saved
    for pref in ['','e_','h_','p2p_']:
        assert not np.isnan(gj.results.loc['best',f'{pref}CACF'])
    assert type(gj.results.loc['best','s_CACF']) == str

    # Check best value is for highest SNR, which is for sector 0
    assert gj.results.loc['best', 's_CACF'] == '1'


def test_validate_ACF():
    validate_ACF(gj)

    # Check all results have been saved
    for pref in ['','p2p_']:
        assert not np.isnan(gj.results.loc['best',f'{pref}ACF'])
    assert type(gj.results.loc['best','s_ACF']) == str

    # Check best value is for highest SNR, which is for sector 0
    assert gj.results.loc['best', 's_ACF'] == '1'

def test_validate_sectors():
    validate_best(gj)

    # Make sure that a flag is added if all sectors disagree with one another
    methods = ['SLS', 'SW', 'CACF','ACF']
    for m in methods:
        gj.results.loc['1', m] *= 5
    validate_sectors(gj)

    assert gj.results.loc['best','f_overall'] >= 8
    gj.results.loc['1', m'] /= 5

def test_validate_best_():
    # Test passing case

    # Test failure modes
    ## Define values first
