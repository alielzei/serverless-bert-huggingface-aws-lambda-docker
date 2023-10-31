"""
Test the hashing module.
"""

import time
import hashlib
import sys
import gc
import io
import collections
import itertools
import pickle
import random
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
from joblib.hashing import hash
from joblib.func_inspect import filter_args
from joblib.memory import Memory
from joblib.testing import raises, skipif, fixture, parametrize
from joblib.test.common import np, with_numpy

def unicode(s):
    return s

def time_func(func, *args):
    """ Time function func on *args.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_hashing.time_func', 'time_func(func, *args)', {'time': time, 'func': func, 'args': args}, 1)

def relative_time(func1, func2, *args):
    """ Return the relative time between func1 and func2 applied on
        *args.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_hashing.relative_time', 'relative_time(func1, func2, *args)', {'time_func': time_func, 'func1': func1, 'func2': func2, 'args': args}, 1)


class Klass(object):
    
    def f(self, x):
        return x



class KlassWithCachedMethod(object):
    
    def __init__(self, cachedir):
        mem = Memory(location=cachedir)
        self.f = mem.cache(self.f)
    
    def f(self, x):
        return x

input_list = [1, 2, 1.0, 2.0, 1 + 1j, 2.0 + 1j, 'a', 'b', (1, ), (1, 1), [1], [1, 1], {1: 1}, {1: 2}, {2: 1}, None, gc.collect, [1].append, set(('a', 1)), set(('a', 1, ('a', 1))), {'a': 1, 1: 2}, {'a': 1, 1: 2, 'd': {'a': 1}}]

@parametrize('obj1', input_list)
@parametrize('obj2', input_list)
def test_trivial_hash(obj1, obj2):
    """Smoke test hash on various types."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_trivial_hash', 'test_trivial_hash(obj1, obj2)', {'parametrize': parametrize, 'input_list': input_list, 'obj1': obj1, 'obj2': obj2}, 0)

def test_hash_methods():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_hash_methods', 'test_hash_methods()', {'io': io, 'unicode': unicode, 'collections': collections}, 0)

@fixture(scope='function')
@with_numpy
def three_np_arrays():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_hashing.three_np_arrays', 'three_np_arrays()', {'np': np, 'fixture': fixture, 'with_numpy': with_numpy}, 3)

def test_hash_numpy_arrays(three_np_arrays):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_hash_numpy_arrays', 'test_hash_numpy_arrays(three_np_arrays)', {'itertools': itertools, 'np': np, 'three_np_arrays': three_np_arrays}, 0)

def test_hash_numpy_dict_of_arrays(three_np_arrays):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_hash_numpy_dict_of_arrays', 'test_hash_numpy_dict_of_arrays(three_np_arrays)', {'three_np_arrays': three_np_arrays}, 0)

@with_numpy
@parametrize('dtype', ['datetime64[s]', 'timedelta64[D]'])
def test_numpy_datetime_array(dtype):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_numpy_datetime_array', 'test_numpy_datetime_array(dtype)', {'np': np, 'with_numpy': with_numpy, 'parametrize': parametrize, 'dtype': dtype}, 0)

@with_numpy
def test_hash_numpy_noncontiguous():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_hash_numpy_noncontiguous', 'test_hash_numpy_noncontiguous()', {'np': np, 'with_numpy': with_numpy}, 0)

@with_numpy
@parametrize('coerce_mmap', [True, False])
def test_hash_memmap(tmpdir, coerce_mmap):
    """Check that memmap and arrays hash identically if coerce_mmap is True."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_hash_memmap', 'test_hash_memmap(tmpdir, coerce_mmap)', {'np': np, 'gc': gc, 'with_numpy': with_numpy, 'parametrize': parametrize, 'tmpdir': tmpdir, 'coerce_mmap': coerce_mmap}, 0)

@with_numpy
@skipif(sys.platform == 'win32', reason='This test is not stable under windows for some reason')
def test_hash_numpy_performance():
    """ Check the performance of hashing numpy arrays:

        In [22]: a = np.random.random(1000000)

        In [23]: %timeit hashlib.md5(a).hexdigest()
        100 loops, best of 3: 20.7 ms per loop

        In [24]: %timeit hashlib.md5(pickle.dumps(a, protocol=2)).hexdigest()
        1 loops, best of 3: 73.1 ms per loop

        In [25]: %timeit hashlib.md5(cPickle.dumps(a, protocol=2)).hexdigest()
        10 loops, best of 3: 53.9 ms per loop

        In [26]: %timeit hash(a)
        100 loops, best of 3: 20.8 ms per loop
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_hash_numpy_performance', 'test_hash_numpy_performance()', {'np': np, 'hashlib': hashlib, 'relative_time': relative_time, 'time_func': time_func, 'with_numpy': with_numpy, 'skipif': skipif, 'sys': sys}, 1)

def test_bound_methods_hash():
    """ Make sure that calling the same method on two different instances
    of the same class does resolve to the same hashes.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_bound_methods_hash', 'test_bound_methods_hash()', {'Klass': Klass, 'filter_args': filter_args}, 0)

def test_bound_cached_methods_hash(tmpdir):
    """ Make sure that calling the same _cached_ method on two different
    instances of the same class does resolve to the same hashes.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_bound_cached_methods_hash', 'test_bound_cached_methods_hash(tmpdir)', {'KlassWithCachedMethod': KlassWithCachedMethod, 'filter_args': filter_args, 'tmpdir': tmpdir}, 0)

@with_numpy
def test_hash_object_dtype():
    """ Make sure that ndarrays with dtype `object' hash correctly."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_hash_object_dtype', 'test_hash_object_dtype()', {'np': np, 'with_numpy': with_numpy}, 0)

@with_numpy
def test_numpy_scalar():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_numpy_scalar', 'test_numpy_scalar()', {'np': np, 'with_numpy': with_numpy}, 0)

def test_dict_hash(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_dict_hash', 'test_dict_hash(tmpdir)', {'KlassWithCachedMethod': KlassWithCachedMethod, 'tmpdir': tmpdir}, 0)

def test_set_hash(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_set_hash', 'test_set_hash(tmpdir)', {'KlassWithCachedMethod': KlassWithCachedMethod, 'tmpdir': tmpdir}, 0)

def test_set_decimal_hash():
    assert hash(set([Decimal(0), Decimal('NaN')])) == hash(set([Decimal('NaN'), Decimal(0)]))

def test_string():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_string', 'test_string()', {'pickle': pickle}, 0)

@with_numpy
def test_numpy_dtype_pickling():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_numpy_dtype_pickling', 'test_numpy_dtype_pickling()', {'np': np, 'pickle': pickle, 'with_numpy': with_numpy}, 0)

@parametrize('to_hash,expected', [('This is a string to hash', '71b3f47df22cb19431d85d92d0b230b2'), ("C'est lété", '2d8d189e9b2b0b2e384d93c868c0e576'), ((123456, 54321, -98765), 'e205227dd82250871fa25aa0ec690aa3'), ([random.Random(42).random() for _ in range(5)], 'a11ffad81f9682a7d901e6edc3d16c84'), ({'abcde': 123, 'sadfas': [-9999, 2, 3]}, 'aeda150553d4bb5c69f0e69d51b0e2ef')])
def test_hashes_stay_the_same(to_hash, expected):
    assert hash(to_hash) == expected

@with_numpy
def test_hashes_are_different_between_c_and_fortran_contiguous_arrays():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_hashes_are_different_between_c_and_fortran_contiguous_arrays', 'test_hashes_are_different_between_c_and_fortran_contiguous_arrays()', {'np': np, 'with_numpy': with_numpy}, 0)

@with_numpy
def test_0d_array():
    hash(np.array(0))

@with_numpy
def test_0d_and_1d_array_hashing_is_different():
    assert hash(np.array(0)) != hash(np.array([0]))

@with_numpy
def test_hashes_stay_the_same_with_numpy_objects():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_hashes_stay_the_same_with_numpy_objects', 'test_hashes_stay_the_same_with_numpy_objects()', {'np': np, 'ProcessPoolExecutor': ProcessPoolExecutor, 'with_numpy': with_numpy}, 1)

def test_hashing_pickling_error():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_hashing_pickling_error', 'test_hashing_pickling_error()', {'raises': raises, 'pickle': pickle}, 1)

def test_wrong_hash_name():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_hashing.test_wrong_hash_name', 'test_wrong_hash_name()', {'raises': raises}, 0)

