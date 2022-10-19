import sys
import pytest
sys.path.append('../')

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from michael import janet
import lightkurve
import glob
import astropy.units as u

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

    j.void['clc_0']  = lk.LightCurve(t, failure)

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
