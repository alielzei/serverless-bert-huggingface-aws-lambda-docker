"""
Thin wrappers around `concurrent.futures`.
"""

from contextlib import contextmanager
from operator import length_hint
from os import cpu_count
from ..auto import tqdm as tqdm_auto
from ..std import TqdmWarning
__author__ = {'github.com/': ['casperdcl']}
__all__ = ['thread_map', 'process_map']

@contextmanager
def ensure_lock(tqdm_class, lock_name=''):
    """get (create if necessary) and then restore `tqdm_class`'s lock"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('tqdm.contrib.concurrent.ensure_lock', "ensure_lock(tqdm_class, lock_name='')", {'contextmanager': contextmanager, 'tqdm_class': tqdm_class, 'lock_name': lock_name}, 0)

def _executor_map(PoolExecutor, fn, *iterables, **tqdm_kwargs):
    """
    Implementation of `thread_map` and `process_map`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    max_workers  : [default: min(32, cpu_count() + 4)].
    chunksize  : [default: 1].
    lock_name  : [default: "":str].
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('tqdm.contrib.concurrent._executor_map', '_executor_map(PoolExecutor, fn, *iterables, **tqdm_kwargs)', {'length_hint': length_hint, 'tqdm_auto': tqdm_auto, 'cpu_count': cpu_count, 'ensure_lock': ensure_lock, 'PoolExecutor': PoolExecutor, 'fn': fn, 'iterables': iterables, 'tqdm_kwargs': tqdm_kwargs}, 1)

def thread_map(fn, *iterables, **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ThreadPoolExecutor`.

    Parameters
    ----------
    tqdm_class  : optional
        `tqdm` class to use for bars [default: tqdm.auto.tqdm].
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to
        `concurrent.futures.ThreadPoolExecutor.__init__`.
        [default: max(32, cpu_count() + 4)].
    """
    from concurrent.futures import ThreadPoolExecutor
    return _executor_map(ThreadPoolExecutor, fn, *iterables, **tqdm_kwargs)

def process_map(fn, *iterables, **tqdm_kwargs):
    """
    Equivalent of `list(map(fn, *iterables))`
    driven by `concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    tqdm_class  : optional
        `tqdm` class to use for bars [default: tqdm.auto.tqdm].
    max_workers  : int, optional
        Maximum number of workers to spawn; passed to
        `concurrent.futures.ProcessPoolExecutor.__init__`.
        [default: min(32, cpu_count() + 4)].
    chunksize  : int, optional
        Size of chunks sent to worker processes; passed to
        `concurrent.futures.ProcessPoolExecutor.map`. [default: 1].
    lock_name  : str, optional
        Member of `tqdm_class.get_lock()` to use [default: mp_lock].
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('tqdm.contrib.concurrent.process_map', 'process_map(fn, *iterables, **tqdm_kwargs)', {'length_hint': length_hint, 'TqdmWarning': TqdmWarning, '_executor_map': _executor_map, 'fn': fn, 'iterables': iterables, 'tqdm_kwargs': tqdm_kwargs}, 1)

