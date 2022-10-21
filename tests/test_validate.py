import sys
import pytest
sys.path.append('../')

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from michael import janet
import astropy.units as u
import glob
import lightkurve

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
    for pref in ['','e_','h_','f_','p2p_', 'snr_']:
        assert not np.isnan(gj.results.loc['best',f'{pref}SLS'])
    assert type(gj.results.loc['best','s_SLS']) == str

    # Check best value is for highest SNR, which is for sector 1
    assert gj.results.loc['best', 's_SLS'] == '1'

def test_validate_SW():
    validate_SW(gj)

    # Check all results have been saved
    for pref in ['','e_','h_','p2p_', 'snr_']:
        assert not np.isnan(gj.results.loc['best',f'{pref}SW'])
    assert type(gj.results.loc['best','s_SW']) == str

    # Check best value is for highest SNR, which is for sector 0
    assert gj.results.loc['best', 's_SW'] == '0'

def test_validate_CACF():
    validate_CACF(gj)

    # Check all results have been saved
    for pref in ['','e_','h_','p2p_', 'snr_']:
        assert not np.isnan(gj.results.loc['best',f'{pref}CACF'])
    assert type(gj.results.loc['best','s_CACF']) == str

    # Check best value is for highest SNR, which is for sector 0
    assert gj.results.loc['best', 's_CACF'] == '1'


def test_validate_ACF():
    validate_ACF(gj)

    # Check all results have been saved
    for pref in ['','p2p_', 'snr_']:
        assert not np.isnan(gj.results.loc['best',f'{pref}ACF'])
    assert type(gj.results.loc['best','s_ACF']) == str

    # Check best value is for highest SNR, which is for sector 0
    assert gj.results.loc['best', 's_ACF'] == '1'

def test_validate_best_pass():
    validate_best(gj)

    assert np.isfinite(gj.results.loc['best', 'overall'])
    assert np.isfinite(gj.results.loc['best', 'e_overall'])
    assert type(gj.results.loc['best', 'method_overall']) == str
    assert np.isfinite(gj.results.loc['best', 'p2p_overall'])
    assert gj.results.loc['best','method_overall'] == 'SW'
    assert_almost_equal(gj.results.loc['best','overall'], gprot, decimal=2)

def test_validate_sectors():
    # Make sure that a flag is added if all sectors disagree with one another
    methods = ['SLS', 'SW', 'CACF','ACF']
    for m in methods:
        gj.results.loc['1', m] *= 5
    validate_sectors(gj)

    assert gj.results.loc['best','f_overall'] >= 8

def test_validate_best_fail():
    # Assert CACF compared to SW and CACF comes out top
    ## Define values first
    gj.results.loc['best', ['SLS','SW','CACF', 'ACF']] = \
                [1., 10., 10., 5.]
    gj.results.loc['best', ['e_SLS','e_SW','e_CACF', 'e_ACF']] = \
                [1., 1., 1., 1.]
    gj.results.loc['best', ['snr_SLS','snr_SW','snr_CACF', 'snr_ACF']] = \
                [50., 10., 20., 50.]
    validate_best(gj)
    assert gj.results.loc['best','method_overall'] == 'CACF'
    assert gj.results.loc['best','f_overall'] == 4

    # Assert CACF compared to SLS and SLS comes out top
    gj.results.loc['best', ['SLS','SW','CACF', 'ACF']] = \
                [10., 1., 10., 5.]
    gj.results.loc['best', ['e_SLS','e_SW','e_CACF', 'e_ACF']] = \
                [1., 1., 1., 1.]
    gj.results.loc['best', ['snr_SLS','snr_SW','snr_CACF', 'snr_ACF']] = \
                [20., 50., 10., 50.]
    validate_best(gj)
    assert gj.results.loc['best','method_overall'] == 'SLS'
    assert gj.results.loc['best','f_overall'] == 4

    # Assert CACF compared to ACF and ACF comes out top
    gj.results.loc['best', ['SLS','SW','CACF', 'ACF']] = \
                [1., 5., 10., 10.]
    gj.results.loc['best', ['e_SLS','e_SW','e_CACF', 'e_ACF']] = \
                [1., 1., 1., 1.]
    gj.results.loc['best', ['snr_SLS','snr_SW','snr_CACF', 'snr_ACF']] = \
                [50., 50., 10., 20.]
    validate_best(gj)
    assert gj.results.loc['best','method_overall'] == 'ACF'
    assert gj.results.loc['best','f_overall'] == 4

    # Assert SW compared to SLS and SW comes out top
    gj.results.loc['best', ['SLS','SW','CACF', 'ACF']] = \
                [10., 10., 1., 5.]
    gj.results.loc['best', ['e_SLS','e_SW','e_CACF', 'e_ACF']] = \
                [1., 1., 1., 1.]
    gj.results.loc['best', ['snr_SLS','snr_SW','snr_CACF', 'snr_ACF']] = \
                [10., 20., 50., 50.]
    validate_best(gj)
    assert gj.results.loc['best','method_overall'] == 'SW'
    assert gj.results.loc['best','f_overall'] == 4

    # Assert no agreement, CACF picked
    gj.results.loc['best', ['SLS','SW','CACF', 'ACF']] = \
                [10., 20., 1., 5.]
    gj.results.loc['best', ['e_SLS','e_SW','e_CACF', 'e_ACF']] = \
                [1., 1., 1., 1.]
    gj.results.loc['best', ['snr_SLS','snr_SW','snr_CACF', 'snr_ACF']] = \
                [10., 20., 50., 1.]
    validate_best(gj)
    assert gj.results.loc['best','method_overall'] == 'CACF'
    assert gj.results.loc['best','f_overall'] == 1
