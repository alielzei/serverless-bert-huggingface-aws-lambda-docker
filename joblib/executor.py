"""Utility function to construct a loky.ReusableExecutor with custom pickler.

This module provides efficient ways of working with data stored in
shared memory with numpy.memmap arrays without inducing any memory
copy between the parent and child processes.
"""

from ._memmapping_reducer import get_memmapping_reducers
from ._memmapping_reducer import TemporaryResourcesManager
from .externals.loky.reusable_executor import _ReusablePoolExecutor
_executor_args = None

def get_memmapping_executor(n_jobs, **kwargs):
    return MemmappingExecutor.get_memmapping_executor(n_jobs, **kwargs)


class MemmappingExecutor(_ReusablePoolExecutor):
    
    @classmethod
    def get_memmapping_executor(cls, n_jobs, timeout=300, initializer=None, initargs=(), env=None, temp_folder=None, context_id=None, **backend_args):
        """Factory for ReusableExecutor with automatic memmapping for large
        numpy arrays.
        """
        global _executor_args
        executor_args = backend_args.copy()
        executor_args.update((env if env else {}))
        executor_args.update(dict(timeout=timeout, initializer=initializer, initargs=initargs))
        reuse = (_executor_args is None or _executor_args == executor_args)
        _executor_args = executor_args
        manager = TemporaryResourcesManager(temp_folder)
        (job_reducers, result_reducers) = get_memmapping_reducers(unlink_on_gc_collect=True, temp_folder_resolver=manager.resolve_temp_folder_name, **backend_args)
        (_executor, executor_is_reused) = super().get_reusable_executor(n_jobs, job_reducers=job_reducers, result_reducers=result_reducers, reuse=reuse, timeout=timeout, initializer=initializer, initargs=initargs, env=env)
        if not executor_is_reused:
            _executor._temp_folder_manager = manager
        if context_id is not None:
            _executor._temp_folder_manager.register_new_context(context_id)
        return _executor
    
    def terminate(self, kill_workers=False):
        self.shutdown(kill_workers=kill_workers)
        with self._submit_resize_lock:
            self._temp_folder_manager._clean_temporary_resources(force=kill_workers, allow_non_empty=True)
    
    @property
    def _temp_folder(self):
        if getattr(self, '_cached_temp_folder', None) is not None:
            return self._cached_temp_folder
        else:
            self._cached_temp_folder = self._temp_folder_manager.resolve_temp_folder_name()
            return self._cached_temp_folder



class _TestingMemmappingExecutor(MemmappingExecutor):
    """Wrapper around ReusableExecutor to ease memmapping testing with Pool
    and Executor. This is only for testing purposes.

    """
    
    def apply_async(self, func, args):
        """Schedule a func to be run"""
        future = self.submit(func, *args)
        future.get = future.result
        return future
    
    def map(self, f, *args):
        return list(super().map(f, *args))


