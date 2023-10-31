"""
Test the memory module.
"""

import functools
import gc
import logging
import shutil
import os
import os.path
import pathlib
import pickle
import sys
import time
import datetime
import textwrap
import pytest
from joblib.memory import Memory
from joblib.memory import expires_after
from joblib.memory import MemorizedFunc, NotMemorizedFunc
from joblib.memory import MemorizedResult, NotMemorizedResult
from joblib.memory import _FUNCTION_HASHES
from joblib.memory import register_store_backend, _STORE_BACKENDS
from joblib.memory import _build_func_identifier, _store_backend_factory
from joblib.memory import JobLibCollisionWarning
from joblib.parallel import Parallel, delayed
from joblib._store_backends import StoreBackendBase, FileSystemStoreBackend
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.testing import parametrize, raises, warns
from joblib.hashing import hash

def f(x, y=1):
    """ A module-level function for testing purposes.
    """
    return x**2 + y

def check_identity_lazy(func, accumulator, location):
    """ Given a function and an accumulator (a list that grows every
        time the function is called), check that the function can be
        decorated by memory to be a lazy identity.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.check_identity_lazy', 'check_identity_lazy(func, accumulator, location)', {'Memory': Memory, 'func': func, 'accumulator': accumulator, 'location': location}, 0)

def corrupt_single_cache_item(memory):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.corrupt_single_cache_item', 'corrupt_single_cache_item(memory)', {'os': os, 'memory': memory}, 0)

def monkeypatch_cached_func_warn(func, monkeypatch_fixture):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.monkeypatch_cached_func_warn', 'monkeypatch_cached_func_warn(func, monkeypatch_fixture)', {'func': func, 'monkeypatch_fixture': monkeypatch_fixture}, 1)

def test_memory_integration(tmpdir):
    """ Simple test of memory lazy evaluation.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_integration', 'test_memory_integration(tmpdir)', {'check_identity_lazy': check_identity_lazy, 'Memory': Memory, 'shutil': shutil, 'tmpdir': tmpdir}, 1)

@parametrize('call_before_reducing', [True, False])
def test_parallel_call_cached_function_defined_in_jupyter(tmpdir, call_before_reducing):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_parallel_call_cached_function_defined_in_jupyter', 'test_parallel_call_cached_function_defined_in_jupyter(tmpdir, call_before_reducing)', {'textwrap': textwrap, 'Memory': Memory, 'os': os, 'Parallel': Parallel, 'delayed': delayed, 'parametrize': parametrize, 'tmpdir': tmpdir, 'call_before_reducing': call_before_reducing}, 0)

def test_no_memory():
    """ Test memory with location=None: no memoize """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_no_memory', 'test_no_memory()', {'Memory': Memory}, 1)

def test_memory_kwarg(tmpdir):
    """ Test memory with a function with keyword arguments."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_kwarg', 'test_memory_kwarg(tmpdir)', {'check_identity_lazy': check_identity_lazy, 'Memory': Memory, 'tmpdir': tmpdir}, 1)

def test_memory_lambda(tmpdir):
    """ Test memory with a function with a lambda."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_lambda', 'test_memory_lambda(tmpdir)', {'check_identity_lazy': check_identity_lazy, 'tmpdir': tmpdir}, 1)

def test_memory_name_collision(tmpdir):
    """ Check that name collisions with functions will raise warnings"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_name_collision', 'test_memory_name_collision(tmpdir)', {'Memory': Memory, 'warns': warns, 'JobLibCollisionWarning': JobLibCollisionWarning, 'tmpdir': tmpdir}, 1)

def test_memory_warning_lambda_collisions(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_warning_lambda_collisions', 'test_memory_warning_lambda_collisions(tmpdir)', {'Memory': Memory, 'warns': warns, 'JobLibCollisionWarning': JobLibCollisionWarning, 'tmpdir': tmpdir}, 0)

def test_memory_warning_collision_detection(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_warning_collision_detection', 'test_memory_warning_collision_detection(tmpdir)', {'Memory': Memory, 'warns': warns, 'JobLibCollisionWarning': JobLibCollisionWarning, 'tmpdir': tmpdir}, 0)

def test_memory_partial(tmpdir):
    """ Test memory with functools.partial."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_partial', 'test_memory_partial(tmpdir)', {'check_identity_lazy': check_identity_lazy, 'tmpdir': tmpdir}, 1)

def test_memory_eval(tmpdir):
    """ Smoke test memory with a function with a function defined in an eval."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_eval', 'test_memory_eval(tmpdir)', {'Memory': Memory, 'tmpdir': tmpdir}, 0)

def count_and_append(x=[]):
    """ A function with a side effect in its arguments.

        Return the length of its argument and append one element.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.count_and_append', 'count_and_append(x=[])', {'x': x}, 1)

def test_argument_change(tmpdir):
    """ Check that if a function has a side effect in its arguments, it
        should use the hash of changing arguments.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_argument_change', 'test_argument_change(tmpdir)', {'Memory': Memory, 'count_and_append': count_and_append, 'tmpdir': tmpdir}, 0)

@with_numpy
@parametrize('mmap_mode', [None, 'r'])
def test_memory_numpy(tmpdir, mmap_mode):
    """ Test memory with a function with numpy arrays."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_numpy', 'test_memory_numpy(tmpdir, mmap_mode)', {'Memory': Memory, 'np': np, 'with_numpy': with_numpy, 'parametrize': parametrize, 'tmpdir': tmpdir, 'mmap_mode': mmap_mode}, 1)

@with_numpy
def test_memory_numpy_check_mmap_mode(tmpdir, monkeypatch):
    """Check that mmap_mode is respected even at the first call"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_numpy_check_mmap_mode', 'test_memory_numpy_check_mmap_mode(tmpdir, monkeypatch)', {'Memory': Memory, 'np': np, 'gc': gc, 'corrupt_single_cache_item': corrupt_single_cache_item, 'monkeypatch_cached_func_warn': monkeypatch_cached_func_warn, 'with_numpy': with_numpy, 'tmpdir': tmpdir, 'monkeypatch': monkeypatch}, 1)

def test_memory_exception(tmpdir):
    """ Smoketest the exception handling of Memory.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_exception', 'test_memory_exception(tmpdir)', {'Memory': Memory, 'raises': raises, 'tmpdir': tmpdir}, 0)

def test_memory_ignore(tmpdir):
    """ Test the ignore feature of memory """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_ignore', 'test_memory_ignore(tmpdir)', {'Memory': Memory, 'tmpdir': tmpdir}, 0)

def test_memory_ignore_decorated(tmpdir):
    """ Test the ignore feature of memory on a decorated function """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_ignore_decorated', 'test_memory_ignore_decorated(tmpdir)', {'Memory': Memory, 'functools': functools, 'tmpdir': tmpdir}, 1)

def test_memory_args_as_kwargs(tmpdir):
    """Non-regression test against 0.12.0 changes.

    https://github.com/joblib/joblib/pull/751
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_args_as_kwargs', 'test_memory_args_as_kwargs(tmpdir)', {'Memory': Memory, 'tmpdir': tmpdir}, 1)

@parametrize('ignore, verbose, mmap_mode', [(['x'], 100, 'r'), ([], 10, None)])
def test_partial_decoration(tmpdir, ignore, verbose, mmap_mode):
    """Check cache may be called with kwargs before decorating"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_partial_decoration', 'test_partial_decoration(tmpdir, ignore, verbose, mmap_mode)', {'Memory': Memory, 'parametrize': parametrize, 'tmpdir': tmpdir, 'ignore': ignore, 'verbose': verbose, 'mmap_mode': mmap_mode}, 0)

def test_func_dir(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_func_dir', 'test_func_dir(tmpdir)', {'Memory': Memory, '__name__': __name__, 'f': f, '_build_func_identifier': _build_func_identifier, 'os': os, '_FUNCTION_HASHES': _FUNCTION_HASHES, 'tmpdir': tmpdir}, 0)

def test_persistence(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_persistence', 'test_persistence(tmpdir)', {'Memory': Memory, 'f': f, 'pickle': pickle, 'os': os, 'tmpdir': tmpdir}, 0)

def test_check_call_in_cache(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_check_call_in_cache', 'test_check_call_in_cache(tmpdir)', {'MemorizedFunc': MemorizedFunc, 'f': f, 'Memory': Memory, 'tmpdir': tmpdir}, 0)

def test_call_and_shelve(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_call_and_shelve', 'test_call_and_shelve(tmpdir)', {'MemorizedFunc': MemorizedFunc, 'f': f, 'NotMemorizedFunc': NotMemorizedFunc, 'Memory': Memory, 'MemorizedResult': MemorizedResult, 'NotMemorizedResult': NotMemorizedResult, 'raises': raises, 'tmpdir': tmpdir}, 0)

def test_call_and_shelve_argument_hash(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_call_and_shelve_argument_hash', 'test_call_and_shelve_argument_hash(tmpdir)', {'Memory': Memory, 'f': f, 'MemorizedResult': MemorizedResult, 'warns': warns, 'tmpdir': tmpdir}, 0)

def test_call_and_shelve_lazily_load_stored_result(tmpdir):
    """Check call_and_shelve only load stored data if needed."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_call_and_shelve_lazily_load_stored_result', 'test_call_and_shelve_lazily_load_stored_result(tmpdir)', {'os': os, 'time': time, 'pytest': pytest, 'Memory': Memory, 'f': f, 'MemorizedResult': MemorizedResult, 'tmpdir': tmpdir}, 0)

def test_memorized_pickling(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memorized_pickling', 'test_memorized_pickling(tmpdir)', {'MemorizedFunc': MemorizedFunc, 'f': f, 'NotMemorizedFunc': NotMemorizedFunc, 'pickle': pickle, 'os': os, 'tmpdir': tmpdir}, 0)

def test_memorized_repr(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memorized_repr', 'test_memorized_repr(tmpdir)', {'MemorizedFunc': MemorizedFunc, 'f': f, 'NotMemorizedFunc': NotMemorizedFunc, 'time': time, 'tmpdir': tmpdir}, 0)

def test_memory_file_modification(capsys, tmpdir, monkeypatch):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_file_modification', 'test_memory_file_modification(capsys, tmpdir, monkeypatch)', {'os': os, 'Memory': Memory, 'shutil': shutil, 'sys': sys, 'capsys': capsys, 'tmpdir': tmpdir, 'monkeypatch': monkeypatch}, 0)

def _function_to_cache(a, b):
    pass

def _sum(a, b):
    return a + b

def _product(a, b):
    return a * b

def test_memory_in_memory_function_code_change(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_in_memory_function_code_change', 'test_memory_in_memory_function_code_change(tmpdir)', {'_function_to_cache': _function_to_cache, '_sum': _sum, 'Memory': Memory, 'warns': warns, 'JobLibCollisionWarning': JobLibCollisionWarning, '_product': _product, 'tmpdir': tmpdir}, 0)

def test_clear_memory_with_none_location():
    memory = Memory(location=None)
    memory.clear()

def func_with_kwonly_args(a, b, *, kw1='kw1', kw2='kw2'):
    return (a, b, kw1, kw2)

def func_with_signature(a: int, b: float) -> float:
    return a + b

def test_memory_func_with_kwonly_args(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_func_with_kwonly_args', 'test_memory_func_with_kwonly_args(tmpdir)', {'Memory': Memory, 'func_with_kwonly_args': func_with_kwonly_args, 'raises': raises, 'tmpdir': tmpdir}, 0)

def test_memory_func_with_signature(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_func_with_signature', 'test_memory_func_with_signature(tmpdir)', {'Memory': Memory, 'func_with_signature': func_with_signature, 'tmpdir': tmpdir}, 0)

def _setup_toy_cache(tmpdir, num_inputs=10):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory._setup_toy_cache', '_setup_toy_cache(tmpdir, num_inputs=10)', {'Memory': Memory, '_build_func_identifier': _build_func_identifier, 'os': os, 'tmpdir': tmpdir, 'num_inputs': num_inputs}, 1)

def test__get_items(tmpdir):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test__get_items', 'test__get_items(tmpdir)', {'_setup_toy_cache': _setup_toy_cache, 'os': os, 'datetime': datetime, 'tmpdir': tmpdir}, 1)

def test__get_items_to_delete(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test__get_items_to_delete', 'test__get_items_to_delete(tmpdir)', {'_setup_toy_cache': _setup_toy_cache, 'tmpdir': tmpdir}, 0)

def test_memory_reduce_size_bytes_limit(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_reduce_size_bytes_limit', 'test_memory_reduce_size_bytes_limit(tmpdir)', {'_setup_toy_cache': _setup_toy_cache, 'tmpdir': tmpdir}, 0)

def test_memory_reduce_size_items_limit(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_reduce_size_items_limit', 'test_memory_reduce_size_items_limit(tmpdir)', {'_setup_toy_cache': _setup_toy_cache, 'tmpdir': tmpdir}, 0)

def test_memory_reduce_size_age_limit(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_reduce_size_age_limit', 'test_memory_reduce_size_age_limit(tmpdir)', {'_setup_toy_cache': _setup_toy_cache, 'tmpdir': tmpdir}, 0)

def test_memory_clear(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_clear', 'test_memory_clear(tmpdir)', {'_setup_toy_cache': _setup_toy_cache, 'os': os, 'tmpdir': tmpdir}, 0)

def fast_func_with_complex_output():
    complex_obj = ['a' * 1000] * 1000
    return complex_obj

def fast_func_with_conditional_complex_output(complex_output=True):
    complex_obj = {str(i): i for i in range(int(100000.0))}
    return (complex_obj if complex_output else 'simple output')

@with_multiprocessing
def test_cached_function_race_condition_when_persisting_output(tmpdir, capfd):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_cached_function_race_condition_when_persisting_output', 'test_cached_function_race_condition_when_persisting_output(tmpdir, capfd)', {'Memory': Memory, 'fast_func_with_complex_output': fast_func_with_complex_output, 'Parallel': Parallel, 'delayed': delayed, 'with_multiprocessing': with_multiprocessing, 'tmpdir': tmpdir, 'capfd': capfd}, 0)

@with_multiprocessing
def test_cached_function_race_condition_when_persisting_output_2(tmpdir, capfd):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_cached_function_race_condition_when_persisting_output_2', 'test_cached_function_race_condition_when_persisting_output_2(tmpdir, capfd)', {'Memory': Memory, 'fast_func_with_conditional_complex_output': fast_func_with_conditional_complex_output, 'Parallel': Parallel, 'delayed': delayed, 'with_multiprocessing': with_multiprocessing, 'tmpdir': tmpdir, 'capfd': capfd}, 0)

def test_memory_recomputes_after_an_error_while_loading_results(tmpdir, monkeypatch):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_recomputes_after_an_error_while_loading_results', 'test_memory_recomputes_after_an_error_while_loading_results(tmpdir, monkeypatch)', {'Memory': Memory, 'time': time, 'corrupt_single_cache_item': corrupt_single_cache_item, 'monkeypatch_cached_func_warn': monkeypatch_cached_func_warn, 'tmpdir': tmpdir, 'monkeypatch': monkeypatch}, 2)


class IncompleteStoreBackend(StoreBackendBase):
    """This backend cannot be instantiated and should raise a TypeError."""
    pass



class DummyStoreBackend(StoreBackendBase):
    """A dummy store backend that does nothing."""
    
    def _open_item(self, *args, **kwargs):
        """Open an item on store."""
        'Does nothing'
    
    def _item_exists(self, location):
        """Check if an item location exists."""
        'Does nothing'
    
    def _move_item(self, src, dst):
        """Move an item from src to dst in store."""
        'Does nothing'
    
    def create_location(self, location):
        """Create location on store."""
        'Does nothing'
    
    def exists(self, obj):
        """Check if an object exists in the store"""
        return False
    
    def clear_location(self, obj):
        """Clear object on store"""
        'Does nothing'
    
    def get_items(self):
        """Returns the whole list of items available in cache."""
        return []
    
    def configure(self, location, *args, **kwargs):
        """Configure the store"""
        'Does nothing'


@parametrize('invalid_prefix', [None, dict(), list()])
def test_register_invalid_store_backends_key(invalid_prefix):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_register_invalid_store_backends_key', 'test_register_invalid_store_backends_key(invalid_prefix)', {'raises': raises, 'register_store_backend': register_store_backend, 'parametrize': parametrize, 'dict': dict, 'list': list, 'invalid_prefix': invalid_prefix}, 0)

def test_register_invalid_store_backends_object():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_register_invalid_store_backends_object', 'test_register_invalid_store_backends_object()', {'raises': raises, 'register_store_backend': register_store_backend}, 0)

def test_memory_default_store_backend():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_default_store_backend', 'test_memory_default_store_backend()', {'raises': raises, 'Memory': Memory}, 0)

def test_warning_on_unknown_location_type():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_warning_on_unknown_location_type', 'test_warning_on_unknown_location_type()', {'warns': warns, '_store_backend_factory': _store_backend_factory}, 0)

def test_instanciate_incomplete_store_backend():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_instanciate_incomplete_store_backend', 'test_instanciate_incomplete_store_backend()', {'register_store_backend': register_store_backend, 'IncompleteStoreBackend': IncompleteStoreBackend, '_STORE_BACKENDS': _STORE_BACKENDS, 'raises': raises, '_store_backend_factory': _store_backend_factory}, 0)

def test_dummy_store_backend():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_dummy_store_backend', 'test_dummy_store_backend()', {'register_store_backend': register_store_backend, 'DummyStoreBackend': DummyStoreBackend, '_STORE_BACKENDS': _STORE_BACKENDS, '_store_backend_factory': _store_backend_factory}, 0)

def test_instanciate_store_backend_with_pathlib_path():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_instanciate_store_backend_with_pathlib_path', 'test_instanciate_store_backend_with_pathlib_path()', {'pathlib': pathlib, '_store_backend_factory': _store_backend_factory}, 0)

def test_filesystem_store_backend_repr(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_filesystem_store_backend_repr', 'test_filesystem_store_backend_repr(tmpdir)', {'FileSystemStoreBackend': FileSystemStoreBackend, 'tmpdir': tmpdir}, 0)

def test_memory_objects_repr(tmpdir):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_objects_repr', 'test_memory_objects_repr(tmpdir)', {'Memory': Memory, 'tmpdir': tmpdir}, 1)

def test_memorized_result_pickle(tmpdir):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memorized_result_pickle', 'test_memorized_result_pickle(tmpdir)', {'Memory': Memory, 'pickle': pickle, 'tmpdir': tmpdir}, 1)

def compare(left, right, ignored_attrs=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.compare', 'compare(left, right, ignored_attrs=None)', {'left': left, 'right': right, 'ignored_attrs': ignored_attrs}, 0)

@pytest.mark.parametrize('memory_kwargs', [{'compress': 3, 'verbose': 2}, {'mmap_mode': 'r', 'verbose': 5, 'backend_options': {'parameter': 'unused'}}])
def test_memory_pickle_dump_load(tmpdir, memory_kwargs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_memory_pickle_dump_load', 'test_memory_pickle_dump_load(tmpdir, memory_kwargs)', {'Memory': Memory, 'pickle': pickle, 'compare': compare, 'f': f, 'pytest': pytest, 'tmpdir': tmpdir, 'memory_kwargs': memory_kwargs}, 0)

def test_info_log(tmpdir, caplog):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_memory.test_info_log', 'test_info_log(tmpdir, caplog)', {'logging': logging, 'Memory': Memory, 'tmpdir': tmpdir, 'caplog': caplog}, 1)

def test_deprecated_bytes_limit(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_memory.test_deprecated_bytes_limit', 'test_deprecated_bytes_limit(tmpdir)', {'pytest': pytest, 'Memory': Memory, 'tmpdir': tmpdir}, 0)


class TestCacheValidationCallback:
    """Tests on parameter `cache_validation_callback`"""
    
    @pytest.fixture()
    def memory(self, tmp_path):
        mem = Memory(location=tmp_path)
        yield mem
        mem.clear()
    
    def foo(self, x, d, delay=None):
        d['run'] = True
        if delay is not None:
            time.sleep(delay)
        return x * 2
    
    def test_invalid_cache_validation_callback(self, memory):
        """Test invalid values for `cache_validation_callback"""
        match = 'cache_validation_callback needs to be callable. Got True.'
        with pytest.raises(ValueError, match=match):
            memory.cache(cache_validation_callback=True)
    
    @pytest.mark.parametrize('consider_cache_valid', [True, False])
    def test_constant_cache_validation_callback(self, memory, consider_cache_valid):
        """Test expiry of old results"""
        f = memory.cache(self.foo, cache_validation_callback=lambda _: consider_cache_valid, ignore=['d'])
        (d1, d2) = ({'run': False}, {'run': False})
        assert f(2, d1) == 4
        assert f(2, d2) == 4
        assert d1['run']
        assert d2['run'] != consider_cache_valid
    
    def test_memory_only_cache_long_run(self, memory):
        """Test cache validity based on run duration."""
        
        def cache_validation_callback(metadata):
            duration = metadata['duration']
            if duration > 0.1:
                return True
        f = memory.cache(self.foo, cache_validation_callback=cache_validation_callback, ignore=['d'])
        (d1, d2) = ({'run': False}, {'run': False})
        assert f(2, d1, delay=0) == 4
        assert f(2, d2, delay=0) == 4
        assert d1['run']
        assert d2['run']
        (d1, d2) = ({'run': False}, {'run': False})
        assert f(2, d1, delay=0.2) == 4
        assert f(2, d2, delay=0.2) == 4
        assert d1['run']
        assert not d2['run']
    
    def test_memory_expires_after(self, memory):
        """Test expiry of old cached results"""
        f = memory.cache(self.foo, cache_validation_callback=expires_after(seconds=0.3), ignore=['d'])
        (d1, d2, d3) = ({'run': False}, {'run': False}, {'run': False})
        assert f(2, d1) == 4
        assert f(2, d2) == 4
        time.sleep(0.5)
        assert f(2, d3) == 4
        assert d1['run']
        assert not d2['run']
        assert d3['run']


