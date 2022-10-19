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
    j.void['clc_0']  = lightkurve.LightCurve(time = np.arange(0, 27., 0.02),
                                            flux = syn)

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
    t = np.arange(0, 27., 0.02)
    failure = np.random.randn(len(t))*0.1 + 1

    j.void['clc_0']  = lightkurve.LightCurve(time = t,
                                            flux = failure)

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
    j.void['clc_0']  = lightkurve.LightCurve(time = np.arange(0, 27., 0.02),
                                            flux = syn)

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

def test_CACF_pass():
    # Set up mock `janet` with synthetic pass case data
    j = janet('synthetic', 0., 0., output_path = 'tests/data')
    j.sectors = ['0']
    j.sectorlist = ['0']
    sfile = glob.glob('tests/data/synthetic*pass*')[0]
    prot = float(sfile.split('_')[-2])
    syn = np.genfromtxt(sfile)
    j.void['clc_0']  = lightkurve.LightCurve(time = np.arange(0, 27., 0.02),
                                            flux = syn)
    simple_wavelet(j, 0, period_range = (1, 13.5))

    # Call function and assert outcome
    composite_ACF(j, 0, period_range = (1, 13.5))

    # Check types of stored objects
    assert type(j.void['0_cacf_popt']) == np.ndarray
    assert type(j.void['0_vizacf']) == lightkurve.lightcurve.LightCurve
    assert type(j.void['0_acflc']) == lightkurve.lightcurve.LightCurve
    assert type(j.void['0_cacf']) == lightkurve.lightcurve.LightCurve
    assert type(j.void['0_cacfsmoo']) == np.ndarray
    assert type(j.void['0_cpeaks']) == np.ndarray

    # Check results have been stored correctly
    assert len(j.results.loc[0]) == 6
    assert all(np.isfinite(j.results.loc[0]))

    # Assert the correct result has been recovered
    assert_almost_equal(j.results.loc[0,'CACF'], prot, decimal=1)

def test_CACF_fail():
    # Set up mock `janet` with synthetic pass case data
    j = janet('synthetic', 0., 0., output_path = 'tests/data')
    j.sectors = ['0']
    j.sectorlist = ['0']

    # Build failure mode data, designed to trigger if statements in methods
    np.random.seed(5)
    t = np.arange(0, 27., 0.02)
    failure = np.random.randn(len(t))*1e-7 + 1

    j.void['clc_0']  = lightkurve.LightCurve(time = t,
                                            flux = failure)
    simple_wavelet(j, 0, period_range = (1, 13.5))

    # Call function and assert outcome
    composite_ACF(j, 0, period_range = (1, 13.5))

    # Check results are nans due to lack peaks
    assert all(np.isnan(j.results.loc[0, ['CACF','e_CACF','h_CACF']]))

def test_ACF_pass():
    # Set up mock `janet` with synthetic pass case data
    j = janet('synthetic', 0., 0., output_path = 'tests/data')
    j.sectors = ['0']
    j.sectorlist = ['0']
    sfile = glob.glob('tests/data/synthetic*pass*')[0]
    prot = float(sfile.split('_')[-2])
    syn = np.genfromtxt(sfile)
    j.void['clc_0']  = lightkurve.LightCurve(time = np.arange(0, 27., 0.02),
                                            flux = syn)

    simple_wavelet(j, 0, period_range = (2, 7))
    composite_ACF(j, 0, period_range = (2, 7))

    # Call function and assert outcome
    simple_ACF(j, 0, period_range = (2, 7))

    # Check types of stored objects
    assert type(j.void['0_acfsmoo']) == np.ndarray
    assert type(j.void['0_peaks']) == np.ndarray

    # Check results have been stored correctly
    assert len(j.results.loc[0]) == 8
    assert all(np.isfinite(j.results.loc[0, j.results.columns[:-1]]))

    # Assert the correct result has been recovered
    assert_almost_equal(j.results.loc[0,'ACF'], prot, decimal=1)

def test_ACF_fail():
    # Set up mock `janet` with synthetic pass case data
    j = janet('synthetic', 0., 0., output_path = 'tests/data')
    j.sectors = ['0']
    j.sectorlist = ['0']

    # Build failure mode data, designed to trigger if statements in methods
    np.random.seed(5)
    t = np.arange(0, 27., 0.02)
    failure = np.random.randn(len(t))*1e-7 + 1

    j.void['clc_0']  = lightkurve.LightCurve(time = t,
                                            flux = failure)
    simple_wavelet(j, 0, period_range = (1, 13.5))
    composite_ACF(j, 0, period_range = (1, 13.5))

    # minimise the scale of the ACF to assure on peaks are captured
    j.void['0_vizacf'] *= 1e-6
    simple_ACF(j, 0, period_range = (1, 13.5))

    # Check results are nans due to lack of peaks
    assert all(np.isnan(j.results.loc[0, ['ACF', 'e_ACF']]))
