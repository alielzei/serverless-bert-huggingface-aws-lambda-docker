import os
from joblib.parallel import parallel_config
from joblib.parallel import parallel_backend
from joblib.parallel import Parallel, delayed
from joblib.parallel import BACKENDS
from joblib.parallel import DEFAULT_BACKEND
from joblib.parallel import EXTERNAL_BACKENDS
from joblib._parallel_backends import LokyBackend
from joblib._parallel_backends import ThreadingBackend
from joblib._parallel_backends import MultiprocessingBackend
from joblib.testing import parametrize, raises
from joblib.test.common import np, with_numpy
from joblib.test.common import with_multiprocessing
from joblib.test.test_parallel import check_memmap

@parametrize('context', [parallel_config, parallel_backend])
def test_global_parallel_backend(context):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_config.test_global_parallel_backend', 'test_global_parallel_backend(context)', {'Parallel': Parallel, 'ThreadingBackend': ThreadingBackend, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context}, 0)

@parametrize('context', [parallel_config, parallel_backend])
def test_external_backends(context):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_config.test_external_backends', 'test_external_backends(context)', {'BACKENDS': BACKENDS, 'ThreadingBackend': ThreadingBackend, 'EXTERNAL_BACKENDS': EXTERNAL_BACKENDS, 'Parallel': Parallel, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context}, 0)

@with_numpy
@with_multiprocessing
def test_parallel_config_no_backend(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_config.test_parallel_config_no_backend', 'test_parallel_config_no_backend(tmpdir)', {'parallel_config': parallel_config, 'Parallel': Parallel, 'LokyBackend': LokyBackend, 'delayed': delayed, 'check_memmap': check_memmap, 'np': np, 'os': os, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'tmpdir': tmpdir}, 0)

@with_numpy
@with_multiprocessing
def test_parallel_config_params_explicit_set(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_config.test_parallel_config_params_explicit_set', 'test_parallel_config_params_explicit_set(tmpdir)', {'parallel_config': parallel_config, 'Parallel': Parallel, 'LokyBackend': LokyBackend, 'raises': raises, 'delayed': delayed, 'check_memmap': check_memmap, 'np': np, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'tmpdir': tmpdir}, 0)

@parametrize('param', ['prefer', 'require'])
def test_parallel_config_bad_params(param):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_config.test_parallel_config_bad_params', 'test_parallel_config_bad_params(param)', {'raises': raises, 'parallel_config': parallel_config, 'Parallel': Parallel, 'parametrize': parametrize, 'param': param}, 0)

def test_parallel_config_constructor_params():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_config.test_parallel_config_constructor_params', 'test_parallel_config_constructor_params()', {'raises': raises, 'parallel_config': parallel_config}, 0)

def test_parallel_config_nested():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_config.test_parallel_config_nested', 'test_parallel_config_nested()', {'parallel_config': parallel_config, 'Parallel': Parallel, 'BACKENDS': BACKENDS, 'DEFAULT_BACKEND': DEFAULT_BACKEND, 'ThreadingBackend': ThreadingBackend}, 0)

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'threading', MultiprocessingBackend(), ThreadingBackend()])
@parametrize('context', [parallel_config, parallel_backend])
def test_threadpool_limitation_in_child_context_error(context, backend):
    with raises(AssertionError, match='does not acc.*inner_max_num_threads'):
        context(backend, inner_max_num_threads=1)

@parametrize('context', [parallel_config, parallel_backend])
def test_parallel_n_jobs_none(context):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_config.test_parallel_n_jobs_none', 'test_parallel_n_jobs_none(context)', {'Parallel': Parallel, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context}, 0)

@parametrize('context', [parallel_config, parallel_backend])
def test_parallel_config_n_jobs_none(context):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_config.test_parallel_config_n_jobs_none', 'test_parallel_config_n_jobs_none(context)', {'Parallel': Parallel, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context}, 0)

