"""
Test the parallel module.
"""

import os
import sys
import time
import mmap
import weakref
import warnings
import threading
from traceback import format_exception
from math import sqrt
from time import sleep
from pickle import PicklingError
from contextlib import nullcontext
from multiprocessing import TimeoutError
import pytest
import joblib
from joblib import parallel
from joblib import dump, load
from joblib._multiprocessing_helpers import mp
from joblib.test.common import np, with_numpy
from joblib.test.common import with_multiprocessing
from joblib.test.common import IS_PYPY, force_gc_pypy
from joblib.testing import parametrize, raises, check_subprocess_call, skipif, warns
if mp is not None:
    from joblib.externals.loky import get_reusable_executor
from queue import Queue
try:
    import posix
except ImportError:
    posix = None
try:
    from ._openmp_test_helper.parallel_sum import parallel_sum
except ImportError:
    parallel_sum = None
try:
    import distributed
except ImportError:
    distributed = None
from joblib._parallel_backends import SequentialBackend
from joblib._parallel_backends import ThreadingBackend
from joblib._parallel_backends import MultiprocessingBackend
from joblib._parallel_backends import ParallelBackendBase
from joblib._parallel_backends import LokyBackend
from joblib.parallel import Parallel, delayed
from joblib.parallel import parallel_config
from joblib.parallel import parallel_backend
from joblib.parallel import register_parallel_backend
from joblib.parallel import effective_n_jobs, cpu_count
from joblib.parallel import mp, BACKENDS, DEFAULT_BACKEND
RETURN_GENERATOR_BACKENDS = BACKENDS.copy()
RETURN_GENERATOR_BACKENDS.pop('multiprocessing', None)
ALL_VALID_BACKENDS = [None] + sorted(BACKENDS.keys())
ALL_VALID_BACKENDS += [BACKENDS[backend_str]() for backend_str in BACKENDS]
if mp is None:
    PROCESS_BACKENDS = []
else:
    PROCESS_BACKENDS = ['multiprocessing', 'loky']
PARALLEL_BACKENDS = PROCESS_BACKENDS + ['threading']
if hasattr(mp, 'get_context'):
    ALL_VALID_BACKENDS.append(mp.get_context('spawn'))
DefaultBackend = BACKENDS[DEFAULT_BACKEND]

def get_workers(backend):
    return getattr(backend, '_pool', getattr(backend, '_workers', None))

def division(x, y):
    return x / y

def square(x):
    return x**2


class MyExceptionWithFinickyInit(Exception):
    """An exception class with non trivial __init__
    """
    
    def __init__(self, a, b, c, d):
        pass


def exception_raiser(x, custom_exception=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel.exception_raiser', 'exception_raiser(x, custom_exception=False)', {'MyExceptionWithFinickyInit': MyExceptionWithFinickyInit, 'x': x, 'custom_exception': custom_exception}, 1)

def interrupt_raiser(x):
    time.sleep(0.05)
    raise KeyboardInterrupt

def f(x, y=0, z=0):
    """ A module-level function so that it can be spawn with
    multiprocessing.
    """
    return x**2 + y + z

def _active_backend_type():
    return type(parallel.get_active_backend()[0])

def parallel_func(inner_n_jobs, backend):
    return Parallel(n_jobs=inner_n_jobs, backend=backend)((delayed(square)(i) for i in range(3)))

def test_cpu_count():
    assert cpu_count() > 0

def test_effective_n_jobs():
    assert effective_n_jobs() > 0

@parametrize('context', [parallel_config, parallel_backend])
@pytest.mark.parametrize('backend_n_jobs, expected_n_jobs', [(3, 3), (-1, effective_n_jobs(n_jobs=-1)), (None, 1)], ids=['positive-int', 'negative-int', 'None'])
@with_multiprocessing
def test_effective_n_jobs_None(context, backend_n_jobs, expected_n_jobs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_effective_n_jobs_None', 'test_effective_n_jobs_None(context, backend_n_jobs, expected_n_jobs)', {'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'pytest': pytest, 'effective_n_jobs': effective_n_jobs, 'with_multiprocessing': with_multiprocessing, 'context': context, 'backend_n_jobs': backend_n_jobs, 'expected_n_jobs': expected_n_jobs}, 0)

@parametrize('backend', ALL_VALID_BACKENDS)
@parametrize('n_jobs', [1, 2, -1, -2])
@parametrize('verbose', [2, 11, 100])
def test_simple_parallel(backend, n_jobs, verbose):
    assert [square(x) for x in range(5)] == Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose)((delayed(square)(x) for x in range(5)))

@parametrize('backend', ALL_VALID_BACKENDS)
def test_main_thread_renamed_no_warning(backend, monkeypatch):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_main_thread_renamed_no_warning', 'test_main_thread_renamed_no_warning(backend, monkeypatch)', {'threading': threading, 'warnings': warnings, 'Parallel': Parallel, 'delayed': delayed, 'square': square, 'parametrize': parametrize, 'ALL_VALID_BACKENDS': ALL_VALID_BACKENDS, 'backend': backend, 'monkeypatch': monkeypatch}, 0)

def _assert_warning_nested(backend, inner_n_jobs, expected):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel._assert_warning_nested', '_assert_warning_nested(backend, inner_n_jobs, expected)', {'warnings': warnings, 'parallel_func': parallel_func, 'sys': sys, 'backend': backend, 'inner_n_jobs': inner_n_jobs, 'expected': expected}, 1)

@with_multiprocessing
@parametrize('parent_backend,child_backend,expected', [('loky', 'multiprocessing', True), ('loky', 'loky', False), ('multiprocessing', 'multiprocessing', True), ('multiprocessing', 'loky', True), ('threading', 'multiprocessing', True), ('threading', 'loky', True)])
def test_nested_parallel_warnings(parent_backend, child_backend, expected):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_nested_parallel_warnings', 'test_nested_parallel_warnings(parent_backend, child_backend, expected)', {'Parallel': Parallel, 'delayed': delayed, '_assert_warning_nested': _assert_warning_nested, 'IS_PYPY': IS_PYPY, 'pytest': pytest, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'parent_backend': parent_backend, 'child_backend': child_backend, 'expected': expected}, 0)

@with_multiprocessing
@parametrize('backend', ['loky', 'multiprocessing', 'threading'])
def test_background_thread_parallelism(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_background_thread_parallelism', 'test_background_thread_parallelism(backend)', {'warnings': warnings, 'Parallel': Parallel, 'delayed': delayed, 'sleep': sleep, 'threading': threading, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'backend': backend}, 0)

def nested_loop(backend):
    Parallel(n_jobs=2, backend=backend)((delayed(square)(0.01) for _ in range(2)))

@parametrize('child_backend', BACKENDS)
@parametrize('parent_backend', BACKENDS)
def test_nested_loop(parent_backend, child_backend):
    Parallel(n_jobs=2, backend=parent_backend)((delayed(nested_loop)(child_backend) for _ in range(2)))

def raise_exception(backend):
    raise ValueError

@with_multiprocessing
def test_nested_loop_with_exception_with_loky():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_nested_loop_with_exception_with_loky', 'test_nested_loop_with_exception_with_loky()', {'raises': raises, 'Parallel': Parallel, 'delayed': delayed, 'nested_loop': nested_loop, 'raise_exception': raise_exception, 'with_multiprocessing': with_multiprocessing}, 0)

def test_mutate_input_with_threads():
    """Input is mutable when using the threading backend"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_mutate_input_with_threads', 'test_mutate_input_with_threads()', {'Queue': Queue, 'Parallel': Parallel, 'delayed': delayed}, 0)

@parametrize('n_jobs', [1, 2, 3])
def test_parallel_kwargs(n_jobs):
    """Check the keyword argument processing of pmap."""
    lst = range(10)
    assert [f(x, y=1) for x in lst] == Parallel(n_jobs=n_jobs)((delayed(f)(x, y=1) for x in lst))

@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_as_context_manager(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_parallel_as_context_manager', 'test_parallel_as_context_manager(backend)', {'f': f, 'Parallel': Parallel, 'delayed': delayed, 'mp': mp, 'get_workers': get_workers, 'parametrize': parametrize, 'PARALLEL_BACKENDS': PARALLEL_BACKENDS, 'backend': backend}, 0)

@with_multiprocessing
def test_parallel_pickling():
    """ Check that pmap captures the errors when it is passed an object
        that cannot be pickled.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_parallel_pickling', 'test_parallel_pickling()', {'raises': raises, 'PicklingError': PicklingError, 'Parallel': Parallel, 'delayed': delayed, 'with_multiprocessing': with_multiprocessing}, 0)

@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_timeout_success(backend):
    assert len(Parallel(n_jobs=2, backend=backend, timeout=30)((delayed(sleep)(0.001) for x in range(10)))) == 10

@with_multiprocessing
@parametrize('backend', PARALLEL_BACKENDS)
def test_parallel_timeout_fail(backend):
    with raises(TimeoutError):
        Parallel(n_jobs=2, backend=backend, timeout=0.01)((delayed(sleep)(10) for x in range(10)))

@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_error_capture(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_error_capture', 'test_error_capture(backend)', {'mp': mp, 'raises': raises, 'Parallel': Parallel, 'delayed': delayed, 'division': division, 'interrupt_raiser': interrupt_raiser, 'get_workers': get_workers, 'f': f, 'MyExceptionWithFinickyInit': MyExceptionWithFinickyInit, 'exception_raiser': exception_raiser, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PROCESS_BACKENDS': PROCESS_BACKENDS, 'backend': backend}, 0)

def consumer(queue, item):
    queue.append('Consumed %s' % item)

@parametrize('backend', BACKENDS)
@parametrize('batch_size, expected_queue', [(1, ['Produced 0', 'Consumed 0', 'Produced 1', 'Consumed 1', 'Produced 2', 'Consumed 2', 'Produced 3', 'Consumed 3', 'Produced 4', 'Consumed 4', 'Produced 5', 'Consumed 5']), (4, ['Produced 0', 'Produced 1', 'Produced 2', 'Produced 3', 'Consumed 0', 'Consumed 1', 'Consumed 2', 'Consumed 3', 'Produced 4', 'Produced 5', 'Consumed 4', 'Consumed 5'])])
def test_dispatch_one_job(backend, batch_size, expected_queue):
    """ Test that with only one job, Parallel does act as a iterator.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_dispatch_one_job', 'test_dispatch_one_job(backend, batch_size, expected_queue)', {'Parallel': Parallel, 'delayed': delayed, 'consumer': consumer, 'parametrize': parametrize, 'BACKENDS': BACKENDS, 'backend': backend, 'batch_size': batch_size, 'expected_queue': expected_queue}, 0)

@with_multiprocessing
@parametrize('backend', PARALLEL_BACKENDS)
def test_dispatch_multiprocessing(backend):
    """ Check that using pre_dispatch Parallel does indeed dispatch items
        lazily.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_dispatch_multiprocessing', 'test_dispatch_multiprocessing(backend)', {'mp': mp, 'Parallel': Parallel, 'delayed': delayed, 'consumer': consumer, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PARALLEL_BACKENDS': PARALLEL_BACKENDS, 'backend': backend}, 0)

def test_batching_auto_threading():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_batching_auto_threading', 'test_batching_auto_threading()', {'Parallel': Parallel, 'delayed': delayed}, 0)

@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_batching_auto_subprocesses(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_batching_auto_subprocesses', 'test_batching_auto_subprocesses(backend)', {'Parallel': Parallel, 'delayed': delayed, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PROCESS_BACKENDS': PROCESS_BACKENDS, 'backend': backend}, 0)

def test_exception_dispatch():
    """Make sure that exception raised during dispatch are indeed captured"""
    with raises(ValueError):
        Parallel(n_jobs=2, pre_dispatch=16, verbose=0)((delayed(exception_raiser)(i) for i in range(30)))

def nested_function_inner(i):
    Parallel(n_jobs=2)((delayed(exception_raiser)(j) for j in range(30)))

def nested_function_outer(i):
    Parallel(n_jobs=2)((delayed(nested_function_inner)(j) for j in range(30)))

@with_multiprocessing
@parametrize('backend', PARALLEL_BACKENDS)
@pytest.mark.xfail(reason='https://github.com/joblib/loky/pull/255')
def test_nested_exception_dispatch(backend):
    """Ensure errors for nested joblib cases gets propagated

    We rely on the Python 3 built-in __cause__ system that already
    report this kind of information to the user.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_nested_exception_dispatch', 'test_nested_exception_dispatch(backend)', {'raises': raises, 'Parallel': Parallel, 'delayed': delayed, 'nested_function_outer': nested_function_outer, 'format_exception': format_exception, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PARALLEL_BACKENDS': PARALLEL_BACKENDS, 'pytest': pytest, 'backend': backend}, 0)


class FakeParallelBackend(SequentialBackend):
    """Pretends to run concurrently while running sequentially."""
    
    def configure(self, n_jobs=1, parallel=None, **backend_args):
        self.n_jobs = self.effective_n_jobs(n_jobs)
        self.parallel = parallel
        return n_jobs
    
    def effective_n_jobs(self, n_jobs=1):
        if n_jobs < 0:
            n_jobs = max(mp.cpu_count() + 1 + n_jobs, 1)
        return n_jobs


def test_invalid_backend():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_invalid_backend', 'test_invalid_backend()', {'raises': raises, 'Parallel': Parallel, 'parallel_config': parallel_config}, 0)

@parametrize('backend', ALL_VALID_BACKENDS)
def test_invalid_njobs(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_invalid_njobs', 'test_invalid_njobs(backend)', {'raises': raises, 'Parallel': Parallel, 'parametrize': parametrize, 'ALL_VALID_BACKENDS': ALL_VALID_BACKENDS, 'backend': backend}, 0)

def test_register_parallel_backend():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_register_parallel_backend', 'test_register_parallel_backend()', {'register_parallel_backend': register_parallel_backend, 'FakeParallelBackend': FakeParallelBackend, 'BACKENDS': BACKENDS}, 0)

def test_overwrite_default_backend():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_overwrite_default_backend', 'test_overwrite_default_backend()', {'_active_backend_type': _active_backend_type, 'DefaultBackend': DefaultBackend, 'register_parallel_backend': register_parallel_backend, 'BACKENDS': BACKENDS, 'ThreadingBackend': ThreadingBackend, 'parallel': parallel, 'DEFAULT_BACKEND': DEFAULT_BACKEND}, 0)

@skipif(mp is not None, reason='Only without multiprocessing')
def test_backend_no_multiprocessing():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_backend_no_multiprocessing', 'test_backend_no_multiprocessing()', {'warns': warns, 'Parallel': Parallel, 'delayed': delayed, 'square': square, 'parallel_config': parallel_config, 'skipif': skipif, 'mp': mp}, 0)

def check_backend_context_manager(context, backend_name):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.check_backend_context_manager', 'check_backend_context_manager(context, backend_name)', {'parallel': parallel, 'effective_n_jobs': effective_n_jobs, 'Parallel': Parallel, 'MultiprocessingBackend': MultiprocessingBackend, 'LokyBackend': LokyBackend, 'ThreadingBackend': ThreadingBackend, 'FakeParallelBackend': FakeParallelBackend, 'context': context, 'backend_name': backend_name}, 0)
all_backends_for_context_manager = PARALLEL_BACKENDS[:]
all_backends_for_context_manager.extend(['test_backend_%d' % i for i in range(3)])

@with_multiprocessing
@parametrize('backend', all_backends_for_context_manager)
@parametrize('context', [parallel_backend, parallel_config])
def test_backend_context_manager(monkeypatch, backend, context):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_backend_context_manager', 'test_backend_context_manager(monkeypatch, backend, context)', {'BACKENDS': BACKENDS, 'FakeParallelBackend': FakeParallelBackend, '_active_backend_type': _active_backend_type, 'DefaultBackend': DefaultBackend, 'check_backend_context_manager': check_backend_context_manager, 'Parallel': Parallel, 'delayed': delayed, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'all_backends_for_context_manager': all_backends_for_context_manager, 'parallel_backend': parallel_backend, 'parallel_config': parallel_config, 'monkeypatch': monkeypatch, 'backend': backend, 'context': context}, 0)


class ParameterizedParallelBackend(SequentialBackend):
    """Pretends to run conncurrently while running sequentially."""
    
    def __init__(self, param=None):
        if param is None:
            raise ValueError('param should not be None')
        self.param = param


@parametrize('context', [parallel_config, parallel_backend])
def test_parameterized_backend_context_manager(monkeypatch, context):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_parameterized_backend_context_manager', 'test_parameterized_backend_context_manager(monkeypatch, context)', {'BACKENDS': BACKENDS, 'ParameterizedParallelBackend': ParameterizedParallelBackend, '_active_backend_type': _active_backend_type, 'DefaultBackend': DefaultBackend, 'parallel': parallel, 'Parallel': Parallel, 'delayed': delayed, 'sqrt': sqrt, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'monkeypatch': monkeypatch, 'context': context}, 0)

@parametrize('context', [parallel_config, parallel_backend])
def test_directly_parameterized_backend_context_manager(context):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_directly_parameterized_backend_context_manager', 'test_directly_parameterized_backend_context_manager(context)', {'_active_backend_type': _active_backend_type, 'DefaultBackend': DefaultBackend, 'ParameterizedParallelBackend': ParameterizedParallelBackend, 'parallel': parallel, 'Parallel': Parallel, 'delayed': delayed, 'sqrt': sqrt, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context}, 0)

def sleep_and_return_pid():
    sleep(0.1)
    return os.getpid()

def get_nested_pids():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel.get_nested_pids', 'get_nested_pids()', {'_active_backend_type': _active_backend_type, 'ThreadingBackend': ThreadingBackend, 'Parallel': Parallel, 'delayed': delayed, 'sleep_and_return_pid': sleep_and_return_pid}, 1)


class MyBackend(joblib._parallel_backends.LokyBackend):
    """Backend to test backward compatibility with older backends"""
    
    def get_nested_backend(self):
        return super(MyBackend, self).get_nested_backend()[0]

register_parallel_backend('back_compat_backend', MyBackend)

@with_multiprocessing
@parametrize('backend', ['threading', 'loky', 'multiprocessing', 'back_compat_backend'])
@parametrize('context', [parallel_config, parallel_backend])
def test_nested_backend_context_manager(context, backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_nested_backend_context_manager', 'test_nested_backend_context_manager(context, backend)', {'Parallel': Parallel, 'delayed': delayed, 'get_nested_pids': get_nested_pids, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context, 'backend': backend}, 0)

@with_multiprocessing
@parametrize('n_jobs', [2, -1, None])
@parametrize('backend', PARALLEL_BACKENDS)
@parametrize('context', [parallel_config, parallel_backend])
def test_nested_backend_in_sequential(backend, n_jobs, context):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_nested_backend_in_sequential', 'test_nested_backend_in_sequential(backend, n_jobs, context)', {'_active_backend_type': _active_backend_type, 'BACKENDS': BACKENDS, 'effective_n_jobs': effective_n_jobs, 'Parallel': Parallel, 'delayed': delayed, 'DEFAULT_BACKEND': DEFAULT_BACKEND, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PARALLEL_BACKENDS': PARALLEL_BACKENDS, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'backend': backend, 'n_jobs': n_jobs, 'context': context}, 0)

def check_nesting_level(context, inner_backend, expected_level):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.check_nesting_level', 'check_nesting_level(context, inner_backend, expected_level)', {'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context, 'inner_backend': inner_backend, 'expected_level': expected_level}, 0)

@with_multiprocessing
@parametrize('outer_backend', PARALLEL_BACKENDS)
@parametrize('inner_backend', PARALLEL_BACKENDS)
@parametrize('context', [parallel_config, parallel_backend])
def test_backend_nesting_level(context, outer_backend, inner_backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_backend_nesting_level', 'test_backend_nesting_level(context, outer_backend, inner_backend)', {'check_nesting_level': check_nesting_level, 'Parallel': Parallel, 'delayed': delayed, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PARALLEL_BACKENDS': PARALLEL_BACKENDS, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context, 'outer_backend': outer_backend, 'inner_backend': inner_backend}, 0)

@with_multiprocessing
@parametrize('context', [parallel_config, parallel_backend])
@parametrize('with_retrieve_callback', [True, False])
def test_retrieval_context(context, with_retrieve_callback):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_retrieval_context', 'test_retrieval_context(context, with_retrieve_callback)', {'ThreadingBackend': ThreadingBackend, 'register_parallel_backend': register_parallel_backend, 'Parallel': Parallel, 'delayed': delayed, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context, 'with_retrieve_callback': with_retrieve_callback}, 1)

@parametrize('batch_size', [0, -1, 1.42])
def test_invalid_batch_size(batch_size):
    with raises(ValueError):
        Parallel(batch_size=batch_size)

@parametrize('n_tasks, n_jobs, pre_dispatch, batch_size', [(2, 2, 'all', 'auto'), (2, 2, 'n_jobs', 'auto'), (10, 2, 'n_jobs', 'auto'), (517, 2, 'n_jobs', 'auto'), (10, 2, 'n_jobs', 'auto'), (10, 4, 'n_jobs', 'auto'), (200, 12, 'n_jobs', 'auto'), (25, 12, '2 * n_jobs', 1), (250, 12, 'all', 1), (250, 12, '2 * n_jobs', 7), (200, 12, '2 * n_jobs', 'auto')])
def test_dispatch_race_condition(n_tasks, n_jobs, pre_dispatch, batch_size):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_dispatch_race_condition', 'test_dispatch_race_condition(n_tasks, n_jobs, pre_dispatch, batch_size)', {'square': square, 'Parallel': Parallel, 'delayed': delayed, 'parametrize': parametrize, 'n_tasks': n_tasks, 'n_jobs': n_jobs, 'pre_dispatch': pre_dispatch, 'batch_size': batch_size}, 0)

@with_multiprocessing
def test_default_mp_context():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_default_mp_context', 'test_default_mp_context()', {'mp': mp, 'Parallel': Parallel, 'with_multiprocessing': with_multiprocessing}, 0)

@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_no_blas_crash_or_freeze_with_subprocesses(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_no_blas_crash_or_freeze_with_subprocesses', 'test_no_blas_crash_or_freeze_with_subprocesses(backend)', {'mp': mp, 'np': np, 'Parallel': Parallel, 'delayed': delayed, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PROCESS_BACKENDS': PROCESS_BACKENDS, 'backend': backend}, 0)
UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_NO_MAIN = 'from joblib import Parallel, delayed\n\ndef square(x):\n    return x ** 2\n\nbackend = "{}"\nif backend == "spawn":\n    from multiprocessing import get_context\n    backend = get_context(backend)\n\nprint(Parallel(n_jobs=2, backend=backend)(\n      delayed(square)(i) for i in range(5)))\n'

@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_parallel_with_interactively_defined_functions(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_parallel_with_interactively_defined_functions', 'test_parallel_with_interactively_defined_functions(backend)', {'mp': mp, 'pytest': pytest, 'UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_NO_MAIN': UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_NO_MAIN, 'check_subprocess_call': check_subprocess_call, 'sys': sys, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PROCESS_BACKENDS': PROCESS_BACKENDS, 'backend': backend}, 0)
UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_MAIN = 'import sys\n# Make sure that joblib is importable in the subprocess launching this\n# script. This is needed in case we run the tests from the joblib root\n# folder without having installed joblib\nsys.path.insert(0, {joblib_root_folder!r})\n\nfrom joblib import Parallel, delayed\n\ndef run(f, x):\n    return f(x)\n\n{define_func}\n\nif __name__ == "__main__":\n    backend = "{backend}"\n    if backend == "spawn":\n        from multiprocessing import get_context\n        backend = get_context(backend)\n\n    callable_position = "{callable_position}"\n    if callable_position == "delayed":\n        print(Parallel(n_jobs=2, backend=backend)(\n                delayed(square)(i) for i in range(5)))\n    elif callable_position == "args":\n        print(Parallel(n_jobs=2, backend=backend)(\n                delayed(run)(square, i) for i in range(5)))\n    else:\n        print(Parallel(n_jobs=2, backend=backend)(\n                delayed(run)(f=square, x=i) for i in range(5)))\n'
SQUARE_MAIN = 'def square(x):\n    return x ** 2\n'
SQUARE_LOCAL = 'def gen_square():\n    def square(x):\n        return x ** 2\n    return square\nsquare = gen_square()\n'
SQUARE_LAMBDA = 'square = lambda x: x ** 2\n'

@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS + (([] if mp is None else ['spawn'])))
@parametrize('define_func', [SQUARE_MAIN, SQUARE_LOCAL, SQUARE_LAMBDA])
@parametrize('callable_position', ['delayed', 'args', 'kwargs'])
def test_parallel_with_unpicklable_functions_in_args(backend, define_func, callable_position, tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_parallel_with_unpicklable_functions_in_args', 'test_parallel_with_unpicklable_functions_in_args(backend, define_func, callable_position, tmpdir)', {'sys': sys, 'pytest': pytest, 'UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_MAIN': UNPICKLABLE_CALLABLE_SCRIPT_TEMPLATE_MAIN, 'os': os, 'joblib': joblib, 'check_subprocess_call': check_subprocess_call, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PROCESS_BACKENDS': PROCESS_BACKENDS, 'mp': mp, 'SQUARE_MAIN': SQUARE_MAIN, 'SQUARE_LOCAL': SQUARE_LOCAL, 'SQUARE_LAMBDA': SQUARE_LAMBDA, 'backend': backend, 'define_func': define_func, 'callable_position': callable_position, 'tmpdir': tmpdir}, 0)
INTERACTIVE_DEFINED_FUNCTION_AND_CLASS_SCRIPT_CONTENT = 'import sys\nimport faulthandler\n# Make sure that joblib is importable in the subprocess launching this\n# script. This is needed in case we run the tests from the joblib root\n# folder without having installed joblib\nsys.path.insert(0, {joblib_root_folder!r})\n\nfrom joblib import Parallel, delayed\nfrom functools import partial\n\nclass MyClass:\n    \'\'\'Class defined in the __main__ namespace\'\'\'\n    def __init__(self, value):\n        self.value = value\n\n\ndef square(x, ignored=None, ignored2=None):\n    \'\'\'Function defined in the __main__ namespace\'\'\'\n    return x.value ** 2\n\n\nsquare2 = partial(square, ignored2=\'something\')\n\n# Here, we do not need the `if __name__ == "__main__":` safeguard when\n# using the default `loky` backend (even on Windows).\n\n# To make debugging easier\nfaulthandler.dump_traceback_later(30, exit=True)\n\n# The following baroque function call is meant to check that joblib\n# introspection rightfully uses cloudpickle instead of the (faster) pickle\n# module of the standard library when necessary. In particular cloudpickle is\n# necessary for functions and instances of classes interactively defined in the\n# __main__ module.\n\nprint(Parallel(backend="loky", n_jobs=2)(\n    delayed(square2)(MyClass(i), ignored=[dict(a=MyClass(1))])\n    for i in range(5)\n))\n'.format(joblib_root_folder=os.path.dirname(os.path.dirname(joblib.__file__)))

@with_multiprocessing
def test_parallel_with_interactively_defined_functions_loky(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_parallel_with_interactively_defined_functions_loky', 'test_parallel_with_interactively_defined_functions_loky(tmpdir)', {'INTERACTIVE_DEFINED_FUNCTION_AND_CLASS_SCRIPT_CONTENT': INTERACTIVE_DEFINED_FUNCTION_AND_CLASS_SCRIPT_CONTENT, 'check_subprocess_call': check_subprocess_call, 'sys': sys, 'with_multiprocessing': with_multiprocessing, 'tmpdir': tmpdir}, 0)
INTERACTIVELY_DEFINED_SUBCLASS_WITH_METHOD_SCRIPT_CONTENT = 'import sys\n# Make sure that joblib is importable in the subprocess launching this\n# script. This is needed in case we run the tests from the joblib root\n# folder without having installed joblib\nsys.path.insert(0, {joblib_root_folder!r})\n\nfrom joblib import Parallel, delayed, hash\nimport multiprocessing as mp\nmp.util.log_to_stderr(5)\n\nclass MyList(list):\n    \'\'\'MyList is interactively defined by MyList.append is a built-in\'\'\'\n    def __hash__(self):\n        # XXX: workaround limitation in cloudpickle\n        return hash(self).__hash__()\n\nl = MyList()\n\nprint(Parallel(backend="loky", n_jobs=2)(\n    delayed(l.append)(i) for i in range(3)\n))\n'.format(joblib_root_folder=os.path.dirname(os.path.dirname(joblib.__file__)))

@with_multiprocessing
def test_parallel_with_interactively_defined_bound_method_loky(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_parallel_with_interactively_defined_bound_method_loky', 'test_parallel_with_interactively_defined_bound_method_loky(tmpdir)', {'INTERACTIVELY_DEFINED_SUBCLASS_WITH_METHOD_SCRIPT_CONTENT': INTERACTIVELY_DEFINED_SUBCLASS_WITH_METHOD_SCRIPT_CONTENT, 'check_subprocess_call': check_subprocess_call, 'sys': sys, 'with_multiprocessing': with_multiprocessing, 'tmpdir': tmpdir}, 0)

def test_parallel_with_exhausted_iterator():
    exhausted_iterator = iter([])
    assert Parallel(n_jobs=2)(exhausted_iterator) == []

def _cleanup_worker():
    """Helper function to force gc in each worker."""
    force_gc_pypy()
    time.sleep(0.1)

def check_memmap(a):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel.check_memmap', 'check_memmap(a)', {'np': np, 'a': a}, 1)

@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_auto_memmap_on_arrays_from_generator(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_auto_memmap_on_arrays_from_generator', 'test_auto_memmap_on_arrays_from_generator(backend)', {'np': np, 'Parallel': Parallel, 'delayed': delayed, 'check_memmap': check_memmap, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PROCESS_BACKENDS': PROCESS_BACKENDS, 'backend': backend}, 0)

def identity(arg):
    return arg

@with_numpy
@with_multiprocessing
def test_memmap_with_big_offset(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_memmap_with_big_offset', 'test_memmap_with_big_offset(tmpdir)', {'mmap': mmap, 'np': np, 'dump': dump, 'load': load, 'Parallel': Parallel, 'delayed': delayed, 'identity': identity, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'tmpdir': tmpdir}, 0)

def test_warning_about_timeout_not_supported_by_backend():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_warning_about_timeout_not_supported_by_backend', 'test_warning_about_timeout_not_supported_by_backend()', {'warnings': warnings, 'Parallel': Parallel, 'delayed': delayed, 'square': square}, 0)

def set_list_value(input_list, index, value):
    input_list[index] = value
    return value

@pytest.mark.parametrize('n_jobs', [1, 2, 4])
def test_parallel_return_order_with_return_as_generator_parameter(n_jobs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_parallel_return_order_with_return_as_generator_parameter', 'test_parallel_return_order_with_return_as_generator_parameter(n_jobs)', {'Parallel': Parallel, 'delayed': delayed, 'set_list_value': set_list_value, 'pytest': pytest, 'n_jobs': n_jobs}, 0)

@parametrize('backend', ALL_VALID_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_abort_backend(n_jobs, backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_abort_backend', 'test_abort_backend(n_jobs, backend)', {'raises': raises, 'time': time, 'Parallel': Parallel, 'delayed': delayed, 'parametrize': parametrize, 'ALL_VALID_BACKENDS': ALL_VALID_BACKENDS, 'n_jobs': n_jobs, 'backend': backend}, 0)

def get_large_object(arg):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel.get_large_object', 'get_large_object(arg)', {'np': np, 'arg': arg}, 1)

@with_numpy
@parametrize('backend', RETURN_GENERATOR_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_deadlock_with_generator(backend, n_jobs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_deadlock_with_generator', 'test_deadlock_with_generator(backend, n_jobs)', {'Parallel': Parallel, 'delayed': delayed, 'get_large_object': get_large_object, 'force_gc_pypy': force_gc_pypy, 'with_numpy': with_numpy, 'parametrize': parametrize, 'RETURN_GENERATOR_BACKENDS': RETURN_GENERATOR_BACKENDS, 'backend': backend, 'n_jobs': n_jobs}, 0)

@parametrize('backend', RETURN_GENERATOR_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_multiple_generator_call(backend, n_jobs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_multiple_generator_call', 'test_multiple_generator_call(backend, n_jobs)', {'raises': raises, 'Parallel': Parallel, 'delayed': delayed, 'sleep': sleep, 'time': time, 'force_gc_pypy': force_gc_pypy, 'parametrize': parametrize, 'RETURN_GENERATOR_BACKENDS': RETURN_GENERATOR_BACKENDS, 'backend': backend, 'n_jobs': n_jobs}, 0)

@parametrize('backend', RETURN_GENERATOR_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_multiple_generator_call_managed(backend, n_jobs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_multiple_generator_call_managed', 'test_multiple_generator_call_managed(backend, n_jobs)', {'Parallel': Parallel, 'delayed': delayed, 'sleep': sleep, 'time': time, 'raises': raises, 'force_gc_pypy': force_gc_pypy, 'parametrize': parametrize, 'RETURN_GENERATOR_BACKENDS': RETURN_GENERATOR_BACKENDS, 'backend': backend, 'n_jobs': n_jobs}, 0)

@parametrize('backend', RETURN_GENERATOR_BACKENDS)
@parametrize('n_jobs', [1, 2, -2, -1])
def test_multiple_generator_call_separated(backend, n_jobs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_multiple_generator_call_separated', 'test_multiple_generator_call_separated(backend, n_jobs)', {'Parallel': Parallel, 'delayed': delayed, 'sqrt': sqrt, 'parametrize': parametrize, 'RETURN_GENERATOR_BACKENDS': RETURN_GENERATOR_BACKENDS, 'backend': backend, 'n_jobs': n_jobs}, 0)

@parametrize('backend, error', [('loky', True), ('threading', False), ('sequential', False)])
def test_multiple_generator_call_separated_gc(backend, error):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_multiple_generator_call_separated_gc', 'test_multiple_generator_call_separated_gc(backend, error)', {'mp': mp, 'pytest': pytest, 'Parallel': Parallel, 'delayed': delayed, 'sleep': sleep, 'weakref': weakref, 'raises': raises, 'nullcontext': nullcontext, 'time': time, 'sqrt': sqrt, 'force_gc_pypy': force_gc_pypy, 'parametrize': parametrize, 'backend': backend, 'error': error}, 0)

@with_numpy
@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_memmapping_leaks(backend, tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_memmapping_leaks', 'test_memmapping_leaks(backend, tmpdir)', {'Parallel': Parallel, 'delayed': delayed, 'check_memmap': check_memmap, 'np': np, 'os': os, '_cleanup_worker': _cleanup_worker, 'sleep': sleep, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PROCESS_BACKENDS': PROCESS_BACKENDS, 'backend': backend, 'tmpdir': tmpdir}, 0)

@parametrize('backend', ([None, 'threading'] if mp is None else [None, 'loky', 'threading']))
def test_lambda_expression(backend):
    results = Parallel(n_jobs=2, backend=backend)((delayed(lambda x: x**2)(i) for i in range(10)))
    assert results == [i**2 for i in range(10)]

@with_multiprocessing
@parametrize('backend', PROCESS_BACKENDS)
def test_backend_batch_statistics_reset(backend):
    """Test that a parallel backend correctly resets its batch statistics."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_backend_batch_statistics_reset', 'test_backend_batch_statistics_reset(backend)', {'Parallel': Parallel, 'delayed': delayed, 'time': time, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'PROCESS_BACKENDS': PROCESS_BACKENDS, 'backend': backend}, 0)

@with_multiprocessing
@parametrize('context', [parallel_config, parallel_backend])
def test_backend_hinting_and_constraints(context):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_backend_hinting_and_constraints', 'test_backend_hinting_and_constraints(context)', {'Parallel': Parallel, 'DefaultBackend': DefaultBackend, 'ThreadingBackend': ThreadingBackend, 'LokyBackend': LokyBackend, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context}, 0)

@parametrize('context', [parallel_config, parallel_backend])
def test_backend_hinting_and_constraints_with_custom_backends(capsys, context):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_backend_hinting_and_constraints_with_custom_backends', 'test_backend_hinting_and_constraints_with_custom_backends(capsys, context)', {'ParallelBackendBase': ParallelBackendBase, 'Parallel': Parallel, 'ThreadingBackend': ThreadingBackend, 'raises': raises, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'capsys': capsys, 'context': context}, 1)

def test_invalid_backend_hinting_and_constraints():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_invalid_backend_hinting_and_constraints', 'test_invalid_backend_hinting_and_constraints()', {'raises': raises, 'Parallel': Parallel, 'mp': mp}, 0)

def _recursive_backend_info(limit=3, **kwargs):
    """Perform nested parallel calls and introspect the backend on the way"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel._recursive_backend_info', '_recursive_backend_info(limit=3, **kwargs)', {'Parallel': Parallel, 'delayed': delayed, '_recursive_backend_info': _recursive_backend_info, 'limit': limit, 'kwargs': kwargs}, 1)

@with_multiprocessing
@parametrize('backend', ['loky', 'threading'])
@parametrize('context', [parallel_config, parallel_backend])
def test_nested_parallelism_limit(context, backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_nested_parallelism_limit', 'test_nested_parallelism_limit(context, backend)', {'_recursive_backend_info': _recursive_backend_info, 'cpu_count': cpu_count, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context, 'backend': backend}, 0)

@with_numpy
@skipif(distributed is None, reason='This test requires dask')
@parametrize('context', [parallel_config, parallel_backend])
def test_nested_parallelism_with_dask(context):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_nested_parallelism_with_dask', 'test_nested_parallelism_with_dask(context)', {'np': np, '_recursive_backend_info': _recursive_backend_info, 'with_numpy': with_numpy, 'skipif': skipif, 'distributed': distributed, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context}, 0)

def _recursive_parallel(nesting_limit=None):
    """A horrible function that does recursive parallel calls"""
    return Parallel()((delayed(_recursive_parallel)() for i in range(2)))

@pytest.mark.no_cover
@parametrize('context', [parallel_config, parallel_backend])
@parametrize('backend', (['threading'] if mp is None else ['loky', 'threading']))
def test_thread_bomb_mitigation(context, backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_thread_bomb_mitigation', 'test_thread_bomb_mitigation(context, backend)', {'raises': raises, '_recursive_parallel': _recursive_parallel, 'PicklingError': PicklingError, 'pytest': pytest, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'mp': mp, 'context': context, 'backend': backend}, 0)

def _run_parallel_sum():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel._run_parallel_sum', '_run_parallel_sum()', {'os': os, 'parallel_sum': parallel_sum}, 2)

@parametrize('backend', ([None, 'loky'] if mp is not None else [None]))
@skipif(parallel_sum is None, reason='Need OpenMP helper compiled')
def test_parallel_thread_limit(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_parallel_thread_limit', 'test_parallel_thread_limit(backend)', {'Parallel': Parallel, 'delayed': delayed, '_run_parallel_sum': _run_parallel_sum, 'cpu_count': cpu_count, 'parametrize': parametrize, 'mp': mp, 'skipif': skipif, 'parallel_sum': parallel_sum, 'backend': backend}, 0)

@skipif(distributed is not None, reason='This test requires dask NOT installed')
@parametrize('context', [parallel_config, parallel_backend])
def test_dask_backend_when_dask_not_installed(context):
    with raises(ValueError, match='Please install dask'):
        context('dask')

@parametrize('context', [parallel_config, parallel_backend])
def test_zero_worker_backend(context):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_zero_worker_backend', 'test_zero_worker_backend(context)', {'ThreadingBackend': ThreadingBackend, 'pytest': pytest, 'Parallel': Parallel, 'delayed': delayed, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context}, 1)

def test_globals_update_at_each_parallel_call():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_globals_update_at_each_parallel_call', 'test_globals_update_at_each_parallel_call()', {'Parallel': Parallel, 'delayed': delayed}, 1)

def _check_numpy_threadpool_limits():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel._check_numpy_threadpool_limits', '_check_numpy_threadpool_limits()', {}, 1)

def _parent_max_num_threads_for(child_module, parent_info):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel._parent_max_num_threads_for', '_parent_max_num_threads_for(child_module, parent_info)', {'child_module': child_module, 'parent_info': parent_info}, 1)

def check_child_num_threads(workers_info, parent_info, num_threads):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.check_child_num_threads', 'check_child_num_threads(workers_info, parent_info, num_threads)', {'_parent_max_num_threads_for': _parent_max_num_threads_for, 'workers_info': workers_info, 'parent_info': parent_info, 'num_threads': num_threads}, 0)

@with_numpy
@with_multiprocessing
@parametrize('n_jobs', [2, 4, -2, -1])
def test_threadpool_limitation_in_child_loky(n_jobs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_threadpool_limitation_in_child_loky', 'test_threadpool_limitation_in_child_loky(n_jobs)', {'_check_numpy_threadpool_limits': _check_numpy_threadpool_limits, 'pytest': pytest, 'Parallel': Parallel, 'delayed': delayed, 'effective_n_jobs': effective_n_jobs, 'cpu_count': cpu_count, 'check_child_num_threads': check_child_num_threads, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'n_jobs': n_jobs}, 0)

@with_numpy
@with_multiprocessing
@parametrize('inner_max_num_threads', [1, 2, 4, None])
@parametrize('n_jobs', [2, -1])
@parametrize('context', [parallel_config, parallel_backend])
def test_threadpool_limitation_in_child_context(context, n_jobs, inner_max_num_threads):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_threadpool_limitation_in_child_context', 'test_threadpool_limitation_in_child_context(context, n_jobs, inner_max_num_threads)', {'_check_numpy_threadpool_limits': _check_numpy_threadpool_limits, 'pytest': pytest, 'Parallel': Parallel, 'delayed': delayed, 'effective_n_jobs': effective_n_jobs, 'cpu_count': cpu_count, 'check_child_num_threads': check_child_num_threads, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context, 'n_jobs': n_jobs, 'inner_max_num_threads': inner_max_num_threads}, 0)

@with_multiprocessing
@parametrize('n_jobs', [2, -1])
@parametrize('var_name', ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'])
@parametrize('context', [parallel_config, parallel_backend])
def test_threadpool_limitation_in_child_override(context, n_jobs, var_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_threadpool_limitation_in_child_override', 'test_threadpool_limitation_in_child_override(context, n_jobs, var_name)', {'get_reusable_executor': get_reusable_executor, 'os': os, 'Parallel': Parallel, 'delayed': delayed, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'parallel_config': parallel_config, 'parallel_backend': parallel_backend, 'context': context, 'n_jobs': n_jobs, 'var_name': var_name}, 1)

@with_multiprocessing
@parametrize('n_jobs', [2, 4, -1])
def test_loky_reuse_workers(n_jobs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_parallel.test_loky_reuse_workers', 'test_loky_reuse_workers(n_jobs)', {'Parallel': Parallel, 'delayed': delayed, 'get_reusable_executor': get_reusable_executor, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'n_jobs': n_jobs}, 0)

