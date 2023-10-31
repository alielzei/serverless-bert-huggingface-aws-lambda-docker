import os
import mmap
import sys
import platform
import gc
import pickle
import itertools
from time import sleep
import subprocess
import threading
import faulthandler
import pytest
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.test.common import with_dev_shm
from joblib.testing import raises, parametrize, skipif
from joblib.backports import make_memmap
from joblib.parallel import Parallel, delayed
from joblib.pool import MemmappingPool
from joblib.executor import _TestingMemmappingExecutor as TestExecutor
from joblib._memmapping_reducer import has_shareable_memory
from joblib._memmapping_reducer import ArrayMemmapForwardReducer
from joblib._memmapping_reducer import _strided_from_memmap
from joblib._memmapping_reducer import _get_temp_dir
from joblib._memmapping_reducer import _WeakArrayKeyMap
from joblib._memmapping_reducer import _get_backing_memmap
import joblib._memmapping_reducer as jmr

def setup_module():
    faulthandler.dump_traceback_later(timeout=300, exit=True)

def teardown_module():
    faulthandler.cancel_dump_traceback_later()

def check_memmap_and_send_back(array):
    assert _get_backing_memmap(array) is not None
    return array

def check_array(args):
    """Dummy helper function to be executed in subprocesses

    Check that the provided array has the expected values in the provided
    range.

    """
    (data, position, expected) = args
    np.testing.assert_array_equal(data[position], expected)

def inplace_double(args):
    """Dummy helper function to be executed in subprocesses


    Check that the input array has the right values in the provided range
    and perform an inplace modification to double the values in the range by
    two.

    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.inplace_double', 'inplace_double(args)', {'np': np, 'args': args}, 0)

@with_numpy
@with_multiprocessing
def test_memmap_based_array_reducing(tmpdir):
    """Check that it is possible to reduce a memmap backed array"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_memmap_based_array_reducing', 'test_memmap_based_array_reducing(tmpdir)', {'np': np, 'ArrayMemmapForwardReducer': ArrayMemmapForwardReducer, 'has_shareable_memory': has_shareable_memory, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'tmpdir': tmpdir}, 1)

@with_multiprocessing
@skipif((sys.platform != 'win32' or ()), reason='PermissionError only easily triggerable on Windows')
def test_resource_tracker_retries_when_permissionerror(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_resource_tracker_retries_when_permissionerror', 'test_resource_tracker_retries_when_permissionerror(tmpdir)', {'subprocess': subprocess, 'with_multiprocessing': with_multiprocessing, 'skipif': skipif, 'sys': sys, 'tmpdir': tmpdir}, 0)

@with_numpy
@with_multiprocessing
def test_high_dimension_memmap_array_reducing(tmpdir):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_high_dimension_memmap_array_reducing', 'test_high_dimension_memmap_array_reducing(tmpdir)', {'np': np, 'ArrayMemmapForwardReducer': ArrayMemmapForwardReducer, 'has_shareable_memory': has_shareable_memory, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'tmpdir': tmpdir}, 1)

@with_numpy
def test__strided_from_memmap(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test__strided_from_memmap', 'test__strided_from_memmap(tmpdir)', {'mmap': mmap, 'np': np, '_strided_from_memmap': _strided_from_memmap, '_get_backing_memmap': _get_backing_memmap, 'with_numpy': with_numpy, 'tmpdir': tmpdir}, 0)

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_pool_with_memmap(factory, tmpdir):
    """Check that subprocess can access and update shared memory memmap"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_pool_with_memmap', 'test_pool_with_memmap(factory, tmpdir)', {'np': np, 'inplace_double': inplace_double, 'os': os, 'raises': raises, 'check_array': check_array, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'MemmappingPool': MemmappingPool, 'TestExecutor': TestExecutor, 'factory': factory, 'tmpdir': tmpdir}, 0)

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_pool_with_memmap_array_view(factory, tmpdir):
    """Check that subprocess can access and update shared memory array"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_pool_with_memmap_array_view', 'test_pool_with_memmap_array_view(factory, tmpdir)', {'np': np, 'has_shareable_memory': has_shareable_memory, 'inplace_double': inplace_double, 'os': os, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'MemmappingPool': MemmappingPool, 'TestExecutor': TestExecutor, 'factory': factory, 'tmpdir': tmpdir}, 0)

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_permission_error_windows_reference_cycle(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_permission_error_windows_reference_cycle', 'test_permission_error_windows_reference_cycle(backend)', {'subprocess': subprocess, 'sys': sys, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'backend': backend}, 0)

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_permission_error_windows_memmap_sent_to_parent(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_permission_error_windows_memmap_sent_to_parent', 'test_permission_error_windows_memmap_sent_to_parent(backend)', {'os': os, '__file__': __file__, 'subprocess': subprocess, 'sys': sys, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'backend': backend}, 0)

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_parallel_isolated_temp_folders(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_parallel_isolated_temp_folders', 'test_parallel_isolated_temp_folders(backend)', {'np': np, 'Parallel': Parallel, 'delayed': delayed, 'os': os, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'backend': backend}, 0)

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_managed_backend_reuse_temp_folder(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_managed_backend_reuse_temp_folder', 'test_managed_backend_reuse_temp_folder(backend)', {'np': np, 'Parallel': Parallel, 'delayed': delayed, 'os': os, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'backend': backend}, 0)

@with_numpy
@with_multiprocessing
def test_memmapping_temp_folder_thread_safety():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_memmapping_temp_folder_thread_safety', 'test_memmapping_temp_folder_thread_safety()', {'np': np, 'Parallel': Parallel, 'delayed': delayed, 'os': os, 'threading': threading, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing}, 0)

@with_numpy
@with_multiprocessing
def test_multithreaded_parallel_termination_resource_tracker_silent():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_multithreaded_parallel_termination_resource_tracker_silent', 'test_multithreaded_parallel_termination_resource_tracker_silent()', {'subprocess': subprocess, 'sys': sys, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing}, 0)

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_many_parallel_calls_on_same_object(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_many_parallel_calls_on_same_object', 'test_many_parallel_calls_on_same_object(backend)', {'os': os, '__file__': __file__, 'subprocess': subprocess, 'sys': sys, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'backend': backend}, 0)

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_memmap_returned_as_regular_array(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_memmap_returned_as_regular_array', 'test_memmap_returned_as_regular_array(backend)', {'np': np, 'Parallel': Parallel, 'delayed': delayed, 'check_memmap_and_send_back': check_memmap_and_send_back, '_get_backing_memmap': _get_backing_memmap, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'backend': backend}, 0)

@with_numpy
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky'])
def test_resource_tracker_silent_when_reference_cycles(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_resource_tracker_silent_when_reference_cycles', 'test_resource_tracker_silent_when_reference_cycles(backend)', {'sys': sys, 'pytest': pytest, 'subprocess': subprocess, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'backend': backend}, 0)

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_pool_for_large_arrays(factory, tmpdir):
    """Check that large arrays are not copied in memory"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_memmapping_pool_for_large_arrays', 'test_memmapping_pool_for_large_arrays(factory, tmpdir)', {'os': os, 'np': np, 'check_array': check_array, 'has_shareable_memory': has_shareable_memory, 'sleep': sleep, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'MemmappingPool': MemmappingPool, 'TestExecutor': TestExecutor, 'factory': factory, 'tmpdir': tmpdir}, 0)

@with_numpy
@with_multiprocessing
@parametrize('backend', [pytest.param('multiprocessing', marks=pytest.mark.xfail(reason='https://github.com/joblib/joblib/issues/1086')), 'loky'])
def test_child_raises_parent_exits_cleanly(backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_child_raises_parent_exits_cleanly', 'test_child_raises_parent_exits_cleanly(backend)', {'os': os, '__file__': __file__, 'subprocess': subprocess, 'sys': sys, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'pytest': pytest, 'backend': backend}, 0)

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_pool_for_large_arrays_disabled(factory, tmpdir):
    """Check that large arrays memmapping can be disabled"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_memmapping_pool_for_large_arrays_disabled', 'test_memmapping_pool_for_large_arrays_disabled(factory, tmpdir)', {'os': os, 'np': np, 'check_array': check_array, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'MemmappingPool': MemmappingPool, 'TestExecutor': TestExecutor, 'factory': factory, 'tmpdir': tmpdir}, 0)

@with_numpy
@with_multiprocessing
@with_dev_shm
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_on_large_enough_dev_shm(factory):
    """Check that memmapping uses /dev/shm when possible"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_memmapping_on_large_enough_dev_shm', 'test_memmapping_on_large_enough_dev_shm(factory)', {'jmr': jmr, 'os': os, 'np': np, 'sleep': sleep, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'with_dev_shm': with_dev_shm, 'parametrize': parametrize, 'MemmappingPool': MemmappingPool, 'TestExecutor': TestExecutor, 'factory': factory}, 0)

@with_numpy
@with_multiprocessing
@with_dev_shm
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_on_too_small_dev_shm(factory):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_memmapping_on_too_small_dev_shm', 'test_memmapping_on_too_small_dev_shm(factory)', {'jmr': jmr, 'os': os, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'with_dev_shm': with_dev_shm, 'parametrize': parametrize, 'MemmappingPool': MemmappingPool, 'TestExecutor': TestExecutor, 'factory': factory}, 0)

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_memmapping_pool_for_large_arrays_in_return(factory, tmpdir):
    """Check that large arrays are not copied in memory in return"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_memmapping_pool_for_large_arrays_in_return', 'test_memmapping_pool_for_large_arrays_in_return(factory, tmpdir)', {'np': np, 'has_shareable_memory': has_shareable_memory, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'MemmappingPool': MemmappingPool, 'TestExecutor': TestExecutor, 'factory': factory, 'tmpdir': tmpdir}, 0)

def _worker_multiply(a, n_times):
    """Multiplication function to be executed by subprocess"""
    assert has_shareable_memory(a)
    return a * n_times

@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_workaround_against_bad_memmap_with_copied_buffers(factory, tmpdir):
    """Check that memmaps with a bad buffer are returned as regular arrays

    Unary operations and ufuncs on memmap instances return a new memmap
    instance with an in-memory buffer (probably a numpy bug).
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_workaround_against_bad_memmap_with_copied_buffers', 'test_workaround_against_bad_memmap_with_copied_buffers(factory, tmpdir)', {'np': np, '_worker_multiply': _worker_multiply, 'has_shareable_memory': has_shareable_memory, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'MemmappingPool': MemmappingPool, 'TestExecutor': TestExecutor, 'factory': factory, 'tmpdir': tmpdir}, 0)

def identity(arg):
    return arg

@with_numpy
@with_multiprocessing
@parametrize('factory,retry_no', list(itertools.product([MemmappingPool, TestExecutor.get_memmapping_executor], range(3))), ids=['{}, {}'.format(x, y) for (x, y) in itertools.product(['multiprocessing', 'loky'], map(str, range(3)))])
def test_pool_memmap_with_big_offset(factory, retry_no, tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_pool_memmap_with_big_offset', 'test_pool_memmap_with_big_offset(factory, retry_no, tmpdir)', {'mmap': mmap, 'make_memmap': make_memmap, 'identity': identity, 'np': np, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'list': list, 'itertools': itertools, 'MemmappingPool': MemmappingPool, 'TestExecutor': TestExecutor, 'range': range, 'x': x, 'y': y, 'map': map, 'str': str, 'factory': factory, 'retry_no': retry_no, 'tmpdir': tmpdir}, 0)

def test_pool_get_temp_dir(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_pool_get_temp_dir', 'test_pool_get_temp_dir(tmpdir)', {'_get_temp_dir': _get_temp_dir, 'sys': sys, 'tmpdir': tmpdir}, 0)

def test_pool_get_temp_dir_no_statvfs(tmpdir, monkeypatch):
    """Check that _get_temp_dir works when os.statvfs is not defined

    Regression test for #902
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_pool_get_temp_dir_no_statvfs', 'test_pool_get_temp_dir_no_statvfs(tmpdir, monkeypatch)', {'joblib': joblib, '_get_temp_dir': _get_temp_dir, 'sys': sys, 'tmpdir': tmpdir, 'monkeypatch': monkeypatch}, 0)

@with_numpy
@skipif(sys.platform == 'win32', reason='This test fails with a PermissionError on Windows')
@parametrize('mmap_mode', ['r+', 'w+'])
def test_numpy_arrays_use_different_memory(mmap_mode):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_numpy_arrays_use_different_memory', 'test_numpy_arrays_use_different_memory(mmap_mode)', {'np': np, 'Parallel': Parallel, 'delayed': delayed, 'with_numpy': with_numpy, 'skipif': skipif, 'sys': sys, 'parametrize': parametrize, 'mmap_mode': mmap_mode}, 1)

@with_numpy
def test_weak_array_key_map():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_weak_array_key_map', 'test_weak_array_key_map()', {'gc': gc, 'sleep': sleep, 'np': np, '_WeakArrayKeyMap': _WeakArrayKeyMap, 'raises': raises, 'platform': platform, 'sys': sys, 'with_numpy': with_numpy}, 1)

def test_weak_array_key_map_no_pickling():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_weak_array_key_map_no_pickling', 'test_weak_array_key_map_no_pickling()', {'_WeakArrayKeyMap': _WeakArrayKeyMap, 'raises': raises, 'pickle': pickle}, 0)

@with_numpy
@with_multiprocessing
def test_direct_mmap(tmpdir):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memmapping.test_direct_mmap', 'test_direct_mmap(tmpdir)', {'np': np, 'mmap': mmap, 'Parallel': Parallel, 'delayed': delayed, 'with_numpy': with_numpy, 'with_multiprocessing': with_multiprocessing, 'tmpdir': tmpdir}, 1)

