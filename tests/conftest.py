import os
import pytest

def pytest_runtest_setup(item):
    r"""Enforce the use of matplotlib's Agg backend, because it
    does not require a graphical display.
    """
    import matplotlib

    matplotlib.use("Agg")
