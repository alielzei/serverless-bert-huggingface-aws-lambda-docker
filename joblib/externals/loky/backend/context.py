import os
import sys
import math
import subprocess
import traceback
import warnings
import multiprocessing as mp
from multiprocessing import get_context as mp_get_context
from multiprocessing.context import BaseContext
from .process import LokyProcess, LokyInitMainProcess
if sys.version_info >= (3, 8):
    from concurrent.futures.process import _MAX_WINDOWS_WORKERS
    if sys.version_info < (3, 10):
        _MAX_WINDOWS_WORKERS = _MAX_WINDOWS_WORKERS - 1
else:
    _MAX_WINDOWS_WORKERS = 60
START_METHODS = ['loky', 'loky_init_main', 'spawn']
if sys.platform != 'win32':
    START_METHODS += ['fork', 'forkserver']
_DEFAULT_START_METHOD = None
physical_cores_cache = None

def get_context(method=None):
    method = (method or _DEFAULT_START_METHOD or 'loky')
    if method == 'fork':
        warnings.warn('`fork` start method should not be used with `loky` as it does not respect POSIX. Try using `spawn` or `loky` instead.', UserWarning)
    try:
        return mp_get_context(method)
    except ValueError:
        raise ValueError(f"Unknown context '{method}'. Value should be in {START_METHODS}.")

def set_start_method(method, force=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.backend.context.set_start_method', 'set_start_method(method, force=False)', {'START_METHODS': START_METHODS, 'method': method, 'force': force}, 0)

def get_start_method():
    return _DEFAULT_START_METHOD

def cpu_count(only_physical_cores=False):
    """Return the number of CPUs the current process can use.

    The returned number of CPUs accounts for:
     * the number of CPUs in the system, as given by
       ``multiprocessing.cpu_count``;
     * the CPU affinity settings of the current process
       (available on some Unix systems);
     * Cgroup CPU bandwidth limit (available on Linux only, typically
       set by docker and similar container orchestration systems);
     * the value of the LOKY_MAX_CPU_COUNT environment variable if defined.
    and is given as the minimum of these constraints.

    If ``only_physical_cores`` is True, return the number of physical cores
    instead of the number of logical cores (hyperthreading / SMT). Note that
    this option is not enforced if the number of usable cores is controlled in
    any other way such as: process affinity, Cgroup restricted CPU bandwidth
    or the LOKY_MAX_CPU_COUNT environment variable. If the number of physical
    cores is not found, return the number of logical cores.

    Note that on Windows, the returned number of CPUs cannot exceed 61 (or 60 for
    Python < 3.10), see:
    https://bugs.python.org/issue26903.

    It is also always larger or equal to 1.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.context.cpu_count', 'cpu_count(only_physical_cores=False)', {'os': os, 'sys': sys, '_MAX_WINDOWS_WORKERS': _MAX_WINDOWS_WORKERS, '_cpu_count_user': _cpu_count_user, '_count_physical_cores': _count_physical_cores, 'warnings': warnings, 'traceback': traceback, 'only_physical_cores': only_physical_cores}, 1)

def _cpu_count_cgroup(os_cpu_count):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.context._cpu_count_cgroup', '_cpu_count_cgroup(os_cpu_count)', {'os': os, 'math': math, 'os_cpu_count': os_cpu_count}, 1)

def _cpu_count_affinity(os_cpu_count):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.context._cpu_count_affinity', '_cpu_count_affinity(os_cpu_count)', {'os': os, 'sys': sys, 'warnings': warnings, 'os_cpu_count': os_cpu_count}, 1)

def _cpu_count_user(os_cpu_count):
    """Number of user defined available CPUs"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.context._cpu_count_user', '_cpu_count_user(os_cpu_count)', {'_cpu_count_affinity': _cpu_count_affinity, '_cpu_count_cgroup': _cpu_count_cgroup, 'os': os, 'os_cpu_count': os_cpu_count}, 1)

def _count_physical_cores():
    """Return a tuple (number of physical cores, exception)

    If the number of physical cores is found, exception is set to None.
    If it has not been found, return ("not found", exception).

    The number of physical cores is cached to avoid repeating subprocess calls.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.context._count_physical_cores', '_count_physical_cores()', {'sys': sys, 'subprocess': subprocess}, 2)


class LokyContext(BaseContext):
    """Context relying on the LokyProcess."""
    _name = 'loky'
    Process = LokyProcess
    cpu_count = staticmethod(cpu_count)
    
    def Queue(self, maxsize=0, reducers=None):
        """Returns a queue object"""
        from .queues import Queue
        return Queue(maxsize, reducers=reducers, ctx=self.get_context())
    
    def SimpleQueue(self, reducers=None):
        """Returns a queue object"""
        from .queues import SimpleQueue
        return SimpleQueue(reducers=reducers, ctx=self.get_context())
    if sys.platform != 'win32':
        'For Unix platform, use our custom implementation of synchronize\n        ensuring that we use the loky.backend.resource_tracker to clean-up\n        the semaphores in case of a worker crash.\n        '
        
        def Semaphore(self, value=1):
            """Returns a semaphore object"""
            from .synchronize import Semaphore
            return Semaphore(value=value)
        
        def BoundedSemaphore(self, value):
            """Returns a bounded semaphore object"""
            from .synchronize import BoundedSemaphore
            return BoundedSemaphore(value)
        
        def Lock(self):
            """Returns a lock object"""
            from .synchronize import Lock
            return Lock()
        
        def RLock(self):
            """Returns a recurrent lock object"""
            from .synchronize import RLock
            return RLock()
        
        def Condition(self, lock=None):
            """Returns a condition object"""
            from .synchronize import Condition
            return Condition(lock)
        
        def Event(self):
            """Returns an event object"""
            from .synchronize import Event
            return Event()



class LokyInitMainContext(LokyContext):
    """Extra context with LokyProcess, which does load the main module

    This context is used for compatibility in the case ``cloudpickle`` is not
    present on the running system. This permits to load functions defined in
    the ``main`` module, using proper safeguards. The declaration of the
    ``executor`` should be protected by ``if __name__ == "__main__":`` and the
    functions and variable used from main should be out of this block.

    This mimics the default behavior of multiprocessing under Windows and the
    behavior of the ``spawn`` start method on a posix system.
    For more details, see the end of the following section of python doc
    https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
    """
    _name = 'loky_init_main'
    Process = LokyInitMainProcess

ctx_loky = LokyContext()
mp.context._concrete_contexts['loky'] = ctx_loky
mp.context._concrete_contexts['loky_init_main'] = LokyInitMainContext()

