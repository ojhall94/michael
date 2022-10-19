import sys
import pytest
sys.path.append('../')

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from michael import janet
from michael.data import data_class
import astropy.units as u

"""
Note: janet.view(), janet.run(), janet.prepare_data() and
janet.validate_rotation() just call functions that are tested separately.
"""

def test_init():
    # Check initiation of correct properties
    gaiaid = 'test'
    ra , dec = 0.0*u.deg, 0.0*u.deg
    output_path = 'data/'
    pipeline = 'eleanor'

    j = janet(gaiaid, ra, dec, output_path, pipeline, verbose=True)

    assert j.gaiaid == gaiaid
    assert j.ra == ra
    assert j.dec == dec
    assert j.output_path == output_path
    assert j.pipeline == pipeline
    assert j.verbose
    assert type(j.void) == dict
    assert j.pl == 'c'

    with pytest.raises(ValueError) as err:
        j = janet(gaiaid, ra, dec, output_path, pipeline = 'not real')

def test_decode():
    # Just run all the decode calls
    gaiaid = 'test'
    ra , dec = 0.0*u.deg, 0.0*u.deg
    output_path = 'data/'
    pipeline = 'eleanor'

    j = janet(gaiaid, ra, dec, output_path, pipeline, verbose=True)

    j.decode(293587)
    j.decode(5)

    with pytest.raises(ValueError) as err:
        j.decode('hello')

def test_get_rotation():
    # Assert `get_rotation` calls the correct functions under different
    # pipelines.

    # Test period range is checked appropriately
    # Just run all the decode calls
    gaiaid = 'test'
    ra , dec = 0.0*u.deg, 0.0*u.deg
    output_path = 'data/'
    pipeline = 'eleanor'
    j = janet(gaiaid, ra, dec, output_path, pipeline, verbose=True)

    with pytest.raises(ValueError) as err:
        j.get_rotation(period_range = (-5, 27))

    with pytest.raises(ValueError) as err:
        j.get_rotation(period_range = (27, 5))

    j.sectors = ['45-46']
    with pytest.raises(ValueError) as err:
        j.get_rotation(period_range = (2, 100))

    # Run through get_rotation() and assert outcomes.
    # Use a tess-sip run to hit all lines of code.
    # Using a known rotator from the Gaia catalogue
    gaiaid = 38329666836450304
    ra = 59.560442
    dec = 12.627969

    j = janet(gaiaid, ra, dec, pipeline = 'tess-sip', output_path = 'tests/data')
    data = data_class(j)
    data.check_eleanor_setup()
    data.build_tess_sip_lc()
    j.get_rotation(period_range = (2, 8))

    # Assert output
    assert len(j.void) == 18
    assert all(list(j.results.index) == j.sectors)
    assert all(np.diff(j.sectorlist) == 1)

def test_view():
    # Using a known rotator from the Gaia catalogue
    gaiaid = 38329666836450304
    ra = 59.560442
    dec = 12.627969

    j = janet(gaiaid, ra, dec, pipeline = 'unpopular', output_path = 'tests/data')
    data = data_class(j)
    data.check_eleanor_setup()
    data.build_eleanor_lc()
    data.build_unpopular_lc()
    j.get_rotation(period_range = (2, 8))
    j.validate_rotation()
    j.view()
