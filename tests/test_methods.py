import sys
import pytest
sys.path.append('../')

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from michael import janet
import lightkurve
import glob
import astropy.units as u
import jazzhands

from michael.methods import simple_astropy_lombscargle
from michael.methods import simple_wavelet
from michael.methods import composite_ACF
from michael.methods import simple_ACF

def test_SLS_pass():
    # Set up mock `janet` with synthetic pass case data
    j = janet('synthetic', 0., 0., output_path = 'tests/data')
    j.sectors = ['0']
    j.sectorlist = ['0']
    sfile = glob.glob('tests/data/synthetic*pass*')[0]
    prot = float(sfile.split('_')[-2])
    syn = np.genfromtxt(sfile)
    j.void['clc_0']  = lightkurve.LightCurve(np.arange(0, 27., 0.02), syn)

    # Call function and assert outcomes
    simple_astropy_lombscargle(j, 0, period_range = (0.1, 27))

    # Check periodogram have been saved
    assert type(j.void['pg_0']) == lightkurve.periodogram.LombScarglePeriodogram
    assert type(j.void['p_0']) == np.ndarray
    assert type(j.void['P_0']) == np.ndarray
    assert type(j.void['popt_0']) == np.ndarray

    # Check results have been stored correctly
    assert len(j.results.loc[0]) == 4
    assert all(np.isfinite(j.results.loc[0]))

    # Assert the correct result has been recovered
    assert_almost_equal(j.results.loc[0,'SLS'], prot, decimal=1)

def test_SLS_fail():
    # Set up mock `janet` with synthetic pass case data
    j = janet('synthetic', 0., 0., output_path = 'tests/data')
    j.sectors = ['0']
    j.sectorlist = ['0']

    # Build failure mode data, designed to trigger if statements in methods
    np.random.seed(5)
    failure = np.random.randn(len(t))*0.1 + 1

    j.void['clc_0']  = lk.LightCurve(np.arange(0, 27., 0.02), failure)

    # Call function and assert outcomes
    simple_astropy_lombscargle(j, 0, period_range = (0.1, 27))

    # Check periodogram have been saved
    assert type(j.void['pg_0']) == lightkurve.periodogram.LombScarglePeriodogram
    assert type(j.void['p_0']) == np.ndarray
    assert type(j.void['P_0']) == np.ndarray
    assert type(j.void['popt_0']) == np.ndarray

    # Check results have been stored correctly
    assert len(j.results.loc[0]) == 4
    assert all(np.isfinite(j.results.loc[0]))
    assert j.results.loc[0, 'f_SLS'] == 3

def test_SW():
    # Set up mock `janet` with synthetic pass case data
    j = janet('synthetic', 0., 0., output_path = 'tests/data')
    j.sectors = ['0']
    j.sectorlist = ['0']
    sfile = glob.glob('tests/data/synthetic*pass*')[0]
    prot = float(sfile.split('_')[-2])
    syn = np.genfromtxt(sfile)
    j.void['clc_0']  = lightkurve.LightCurve(np.arange(0, 27., 0.02), syn)

    # Call function and assert outcomes
    simple_wavelet(j, 0, period_range = (2, 8))

    # Check wavelet have been saved
    assert type(j.void['0_wt']) == jazzhands.wavelets.WaveletTransformer
    assert type(j.void['0_wwz']) == np.ndarray
    assert type(j.void['0_wwa']) == np.ndarray
    assert type(j.void['0_wavelet_popt']) == np.ndarray

    # Check results have been stored correctly
    assert len(j.results.loc[0]) == 3
    assert all(np.isfinite(j.results.loc[0]))

    # Assert the correct result has been recovered
    assert_almost_equal(j.results.loc[0,'SW'], prot, decimal=1)

    # Check that SW doesn't fail on a ridiculous period_range
    j.void['clc_0']  = lightkurve.LightCurve(np.arange(0, 27., 0.02), syn)
    simple_wavelet(j, 0, period_range = (9, 10))
    assert len(j.results.loc[0]) == 3
    assert all(np.isfinite(j.results.loc[0]))

=
