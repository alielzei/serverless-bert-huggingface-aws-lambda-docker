"""
Pytest configuration and fixtures for the Numpy test suite.
"""

import os
import tempfile
import hypothesis
import pytest
import numpy
from numpy.core._multiarray_tests import get_fpu_mode
_old_fpu_mode = None
_collect_results = {}
hypothesis.configuration.set_hypothesis_home_dir(os.path.join(tempfile.gettempdir(), '.hypothesis'))
hypothesis.settings.register_profile(name='numpy-profile', deadline=None, print_blob=True)
hypothesis.settings.register_profile(name='np.test() profile', deadline=None, print_blob=True, database=None, derandomize=True, suppress_health_check=list(hypothesis.HealthCheck))
_pytest_ini = os.path.join(os.path.dirname(__file__), '..', 'pytest.ini')
hypothesis.settings.load_profile(('numpy-profile' if os.path.isfile(_pytest_ini) else 'np.test() profile'))

def pytest_configure(config):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.conftest.pytest_configure', 'pytest_configure(config)', {'config': config}, 0)

def pytest_addoption(parser):
    parser.addoption('--available-memory', action='store', default=None, help='Set amount of memory available for running the test suite. This can result to tests requiring especially large amounts of memory to be skipped. Equivalent to setting environment variable NPY_AVAILABLE_MEM. Default: determinedautomatically.')

def pytest_sessionstart(session):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.conftest.pytest_sessionstart', 'pytest_sessionstart(session)', {'os': os, 'session': session}, 0)

@pytest.hookimpl()
def pytest_itemcollected(item):
    """
    Check FPU precision mode was not changed during test collection.

    The clumsy way we do it here is mainly necessary because numpy
    still uses yield tests, which can execute code at test collection
    time.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.conftest.pytest_itemcollected', 'pytest_itemcollected(item)', {'get_fpu_mode': get_fpu_mode, '_collect_results': _collect_results, 'pytest': pytest, 'item': item}, 0)

@pytest.fixture(scope='function', autouse=True)
def check_fpu_mode(request):
    """
    Check FPU precision mode was not changed during the test.
    """
    old_mode = get_fpu_mode()
    yield
    new_mode = get_fpu_mode()
    if old_mode != new_mode:
        raise AssertionError('FPU precision mode changed from {0:#x} to {1:#x} during the test'.format(old_mode, new_mode))
    collect_result = _collect_results.get(request.node)
    if collect_result is not None:
        (old_mode, new_mode) = collect_result
        raise AssertionError('FPU precision mode changed from {0:#x} to {1:#x} when collecting the test'.format(old_mode, new_mode))

@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace['np'] = numpy

@pytest.fixture(autouse=True)
def env_setup(monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', '0')

@pytest.fixture(params=[True, False])
def weak_promotion(request):
    """
    Fixture to ensure "legacy" promotion state or change it to use the new
    weak promotion (plus warning).  `old_promotion` should be used as a
    parameter in the function.
    """
    state = numpy._get_promotion_state()
    if request.param:
        numpy._set_promotion_state('weak_and_warn')
    else:
        numpy._set_promotion_state('legacy')
    yield request.param
    numpy._set_promotion_state(state)

