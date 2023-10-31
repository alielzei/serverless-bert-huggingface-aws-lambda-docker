"""
Small utilities for testing.
"""

import os
import gc
import sys
from joblib._multiprocessing_helpers import mp
from joblib.testing import SkipTest, skipif
try:
    import lz4
except ImportError:
    lz4 = None
IS_PYPY = hasattr(sys, 'pypy_version_info')
try:
    import numpy as np
    
    def with_numpy(func):
        """A decorator to skip tests requiring numpy."""
        return func
except ImportError:
    
    def with_numpy(func):
        """A decorator to skip tests requiring numpy."""
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('joblib.test.common.with_numpy', 'with_numpy(func)', {'SkipTest': SkipTest, 'func': func}, 1)
    np = None
try:
    from memory_profiler import memory_usage
    
    def with_memory_profiler(func):
        """A decorator to skip tests requiring memory_profiler."""
        return func
    
    def memory_used(func, *args, **kwargs):
        """Compute memory usage when executing func."""
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('joblib.test.common.memory_used', 'memory_used(func, *args, **kwargs)', {'gc': gc, 'memory_usage': memory_usage, 'func': func, 'args': args, 'kwargs': kwargs}, 1)
except ImportError:
    
    def with_memory_profiler(func):
        """A decorator to skip tests requiring memory_profiler."""
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('joblib.test.common.with_memory_profiler', 'with_memory_profiler(func)', {'SkipTest': SkipTest, 'func': func}, 1)
    memory_usage = memory_used = None

def force_gc_pypy():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.common.force_gc_pypy', 'force_gc_pypy()', {'IS_PYPY': IS_PYPY}, 0)
with_multiprocessing = skipif(mp is None, reason='Needs multiprocessing to run.')
with_dev_shm = skipif(not os.path.exists('/dev/shm'), reason='This test requires a large /dev/shm shared memory fs.')
with_lz4 = skipif(lz4 is None, reason='Needs lz4 compression to run')
without_lz4 = skipif(lz4 is not None, reason='Needs lz4 not being installed to run')

