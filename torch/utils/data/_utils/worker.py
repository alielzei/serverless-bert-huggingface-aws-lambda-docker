""""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import random
import os
from collections import namedtuple
from torch._six import queue
from torch._utils import ExceptionWrapper
from . import signal_handling, MP_STATUS_CHECK_INTERVAL, IS_WINDOWS
if IS_WINDOWS:
    import ctypes
    from ctypes.wintypes import DWORD, BOOL, HANDLE
    
    
    class ManagerWatchdog(object):
        
        def __init__(self):
            self.manager_pid = os.getppid()
            self.kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            self.kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
            self.kernel32.OpenProcess.restype = HANDLE
            self.kernel32.WaitForSingleObject.argtypes = (HANDLE, DWORD)
            self.kernel32.WaitForSingleObject.restype = DWORD
            SYNCHRONIZE = 1048576
            self.manager_handle = self.kernel32.OpenProcess(SYNCHRONIZE, 0, self.manager_pid)
            if not self.manager_handle:
                raise ctypes.WinError(ctypes.get_last_error())
            self.manager_dead = False
        
        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = self.kernel32.WaitForSingleObject(self.manager_handle, 0) == 0
            return not self.manager_dead
    
else:
    
    
    class ManagerWatchdog(object):
        
        def __init__(self):
            self.manager_pid = os.getppid()
            self.manager_dead = False
        
        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead
    
_worker_info = None


class WorkerInfo(object):
    __initialized = False
    
    def __init__(self, **kwargs):
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        self.__initialized = True
    
    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError('Cannot assign attributes to {} objects'.format(self.__class__.__name__))
        return super(WorkerInfo, self).__setattr__(key, val)


def get_worker_info():
    """Returns the information about the current
    :class:`~torch.utils.data.DataLoader` iterator worker process.

    When called in a worker, this returns an object guaranteed to have the
    following attributes:

    * :attr:`id`: the current worker id.
    * :attr:`num_workers`: the total number of workers.
    * :attr:`seed`: the random seed set for the current worker. This value is
      determined by main process RNG and the worker id. See
      :class:`~torch.utils.data.DataLoader`'s documentation for more details.
    * :attr:`dataset`: the copy of the dataset object in **this** process. Note
      that this will be a different object in a different process than the one
      in the main process.

    When called in the main process, this returns ``None``.

    .. note::
       When used in a :attr:`worker_init_fn` passed over to
       :class:`~torch.utils.data.DataLoader`, this method can be useful to
       set up each worker process differently, for instance, using ``worker_id``
       to configure the ``dataset`` object to only read a specific fraction of a
       sharded dataset, or use ``seed`` to seed other libraries used in dataset
       code (e.g., NumPy).
    """
    return _worker_info
'Dummy class used to signal the end of an IterableDataset'
_IterableDatasetStopIteration = namedtuple('_IterableDatasetStopIteration', ['worker_id'])

def _worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event, auto_collation, collate_fn, drop_last, seed, init_fn, worker_id, num_workers):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.data._utils.worker._worker_loop', '_worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event, auto_collation, collate_fn, drop_last, seed, init_fn, worker_id, num_workers)', {'signal_handling': signal_handling, 'torch': torch, 'random': random, 'WorkerInfo': WorkerInfo, 'ExceptionWrapper': ExceptionWrapper, 'ManagerWatchdog': ManagerWatchdog, 'MP_STATUS_CHECK_INTERVAL': MP_STATUS_CHECK_INTERVAL, 'queue': queue, '_IterableDatasetStopIteration': _IterableDatasetStopIteration, 'dataset_kind': dataset_kind, 'dataset': dataset, 'index_queue': index_queue, 'data_queue': data_queue, 'done_event': done_event, 'auto_collation': auto_collation, 'collate_fn': collate_fn, 'drop_last': drop_last, 'seed': seed, 'init_fn': init_fn, 'worker_id': worker_id, 'num_workers': num_workers}, 0)

