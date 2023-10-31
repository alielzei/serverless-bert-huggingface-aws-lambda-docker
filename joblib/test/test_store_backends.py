try:
    import cPickle as cpickle
except ImportError:
    import pickle as cpickle
import functools
from pickle import PicklingError
import time
import pytest
from joblib.testing import parametrize, timeout
from joblib.test.common import with_multiprocessing
from joblib.backports import concurrency_safe_rename
from joblib import Parallel, delayed
from joblib._store_backends import concurrency_safe_write, FileSystemStoreBackend, CacheWarning

def write_func(output, filename):
    with open(filename, 'wb') as f:
        cpickle.dump(output, f)

def load_func(expected, filename):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_store_backends.load_func', 'load_func(expected, filename)', {'cpickle': cpickle, 'IOError': IOError, 'time': time, 'expected': expected, 'filename': filename}, 0)

def concurrency_safe_write_rename(to_write, filename, write_func):
    temporary_filename = concurrency_safe_write(to_write, filename, write_func)
    concurrency_safe_rename(temporary_filename, filename)

@timeout(0)
@with_multiprocessing
@parametrize('backend', ['multiprocessing', 'loky', 'threading'])
def test_concurrency_safe_write(tmpdir, backend):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_store_backends.test_concurrency_safe_write', 'test_concurrency_safe_write(tmpdir, backend)', {'functools': functools, 'concurrency_safe_write_rename': concurrency_safe_write_rename, 'write_func': write_func, 'load_func': load_func, 'Parallel': Parallel, 'delayed': delayed, 'timeout': timeout, 'with_multiprocessing': with_multiprocessing, 'parametrize': parametrize, 'tmpdir': tmpdir, 'backend': backend}, 0)

def test_warning_on_dump_failure(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_store_backends.test_warning_on_dump_failure', 'test_warning_on_dump_failure(tmpdir)', {'FileSystemStoreBackend': FileSystemStoreBackend, 'pytest': pytest, 'CacheWarning': CacheWarning, 'tmpdir': tmpdir}, 0)

def test_warning_on_pickling_error(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_store_backends.test_warning_on_pickling_error', 'test_warning_on_pickling_error(tmpdir)', {'PicklingError': PicklingError, 'FileSystemStoreBackend': FileSystemStoreBackend, 'pytest': pytest, 'tmpdir': tmpdir}, 0)

