import sys
import pytest
sys.path.append('../')

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from michael.utils import _gaussian_fn, _decode

def test_gaussian():
    # Make sure the thing returned is a Gaussian
    mu = 0,
    sigma = 1
    A = 1
    x = np.linspace(-5, 5, 1000)
    y = _gaussian_fn(x, mu, sigma, A)

    assert_almost_equal(mu, x[np.argmax(y)], decimal=2)
    assert_almost_equal(np.max(y), A, decimal=2)

def test_decode():
    # Make sure the decoder returns a string for a sensible value
    message = _decode(17)
    assert type(message) == str
