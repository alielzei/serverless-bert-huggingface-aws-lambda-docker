import torch
import torch.utils.hooks
from torch._namedtensor_internals import check_serializing_named_tensor
import os
import threading
import errno
import multiprocessing
from multiprocessing.util import register_after_fork
from multiprocessing.reduction import ForkingPickler
import sys
try:
    import multiprocessing.resource_sharer
except ImportError:
    pass


class StorageWeakRef(object):
    """A weak reference to a Storage.

    The cdata member is a Python number containing the integer representation of
    the Storage pointer."""
    
    def __init__(self, storage):
        self.cdata = storage._weak_ref()
        self._free_weak_ref = torch.Storage._free_weak_ref
    
    def expired(self):
        return torch.Storage._expired(self.cdata)
    
    def __del__(self):
        self._free_weak_ref(self.cdata)



class SharedCache(dict):
    """dictionary from multiprocessing handles to StorageWeakRef"""
    
    def __init__(self):
        self.limit = 128
        self._after_fork()
        register_after_fork(self, SharedCache._after_fork)
    
    def _after_fork(self):
        self.lock = threading.Lock()
    
    def __setitem__(self, key, storage_ref):
        dict.__setitem__(self, key, storage_ref)
        if len(self) > self.limit:
            self.free_dead_references()
    
    def free_dead_references(self):
        with self.lock:
            live = 0
            for (key, storage_ref) in list(self.items()):
                if storage_ref.expired():
                    del self[key]
                else:
                    live += 1
            self.limit = max(128, live * 2)

shared_cache = SharedCache()

def rebuild_event(device, handle):
    return torch.cuda.Event.from_ipc_handle(device, handle)

def reduce_event(event):
    handle = event.ipc_handle()
    return (rebuild_event, (event.device, handle))

def rebuild_tensor(cls, storage, metadata):
    (storage_offset, size, stride, requires_grad) = metadata
    t = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    if cls == torch.nn.parameter.Parameter:
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        t.requires_grad = requires_grad
    return t

def rebuild_cuda_tensor(tensor_cls, tensor_size, tensor_stride, tensor_offset, storage_cls, storage_device, storage_handle, storage_size_bytes, storage_offset_bytes, requires_grad, ref_counter_handle, ref_counter_offset, event_handle, event_sync_required):
    if (storage_handle is None or storage_size_bytes == 0):
        storage = storage_cls(0)
    else:
        storage = storage_from_cache(storage_cls, (storage_handle, storage_offset_bytes))
        if storage is None:
            torch.cuda._lazy_init()
            storage = storage_cls._new_shared_cuda(storage_device, storage_handle, storage_size_bytes, storage_offset_bytes, ref_counter_handle, ref_counter_offset, event_handle, event_sync_required)
            shared_cache[(storage_handle, storage_offset_bytes)] = StorageWeakRef(storage)
        else:
            storage_cls._release_ipc_counter(ref_counter_handle, ref_counter_offset)
    t = torch._utils._rebuild_tensor(storage, tensor_offset, tensor_size, tensor_stride)
    if tensor_cls == torch.nn.parameter.Parameter:
        t = torch.nn.parameter.Parameter(t)
    t.requires_grad = requires_grad
    return t

def reduce_tensor(tensor):
    storage = tensor.storage()
    if (tensor.requires_grad and not tensor.is_leaf):
        raise RuntimeError('Cowardly refusing to serialize non-leaf tensor which requires_grad, since autograd does not support crossing process boundaries.  If you just want to transfer the data, call detach() on the tensor before serializing (e.g., putting it on the queue).')
    check_serializing_named_tensor(tensor)
    torch.utils.hooks.warn_if_has_hooks(tensor)
    if storage.is_cuda:
        (device, handle, storage_size_bytes, storage_offset_bytes, ref_counter_handle, ref_counter_offset, event_handle, event_sync_required) = storage._share_cuda_()
        tensor_offset = tensor.storage_offset()
        shared_cache[handle] = StorageWeakRef(storage)
        return (rebuild_cuda_tensor, (type(tensor), tensor.size(), tensor.stride(), tensor_offset, type(storage), device, handle, storage_size_bytes, storage_offset_bytes, tensor.requires_grad, ref_counter_handle, ref_counter_offset, event_handle, event_sync_required))
    metadata = (tensor.storage_offset(), tensor.size(), tensor.stride(), tensor.requires_grad)
    return (rebuild_tensor, (type(tensor), storage, metadata))

def fd_id(fd):
    stat = os.fstat(fd)
    return (stat.st_ino, stat.st_dev)

def storage_from_cache(cls, key):
    storage_ref = shared_cache.get(key)
    if storage_ref is None:
        return None
    return cls._new_with_weak_ptr(storage_ref.cdata)

def rebuild_storage_fd(cls, df, size):
    if sys.version_info[0] == 2:
        while True:
            try:
                fd = multiprocessing.reduction.rebuild_handle(df)
                break
            except OSError as e:
                if e.errno != getattr(errno, 'EINTR', None):
                    raise
    else:
        fd = df.detach()
    try:
        storage = storage_from_cache(cls, fd_id(fd))
        if storage is not None:
            return storage
        storage = cls._new_shared_fd(fd, size)
        shared_cache[fd_id(fd)] = StorageWeakRef(storage)
        return storage
    finally:
        os.close(fd)

def rebuild_storage_filename(cls, manager, handle, size):
    storage = storage_from_cache(cls, handle)
    if storage is not None:
        return storage._shared_decref()
    storage = cls._new_shared_filename(manager, handle, size)
    shared_cache[handle] = StorageWeakRef(storage)
    return storage._shared_decref()

def rebuild_storage_empty(cls):
    return cls()

def reduce_storage(storage):
    from . import get_sharing_strategy
    if storage.is_cuda:
        raise RuntimeError('Cannot pickle CUDA storage; try pickling a CUDA tensor instead')
    elif get_sharing_strategy() == 'file_system':
        metadata = storage._share_filename_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_filename
        storage._shared_incref()
    elif storage.size() == 0:
        return (rebuild_storage_empty, (type(storage), ))
    else:
        (fd, size) = storage._share_fd_()
        if sys.version_info[0] == 2:
            df = multiprocessing.reduction.reduce_handle(fd)
        else:
            df = multiprocessing.reduction.DupFd(fd)
        cache_key = fd_id(fd)
        metadata = (df, size)
        rebuild = rebuild_storage_fd
    shared_cache[cache_key] = StorageWeakRef(storage)
    return (rebuild, (type(storage), ) + metadata)

def init_reductions():
    ForkingPickler.register(torch.cuda.Event, reduce_event)
    for t in torch._storage_classes:
        ForkingPickler.register(t, reduce_storage)
    for t in torch._tensor_classes:
        ForkingPickler.register(t, reduce_tensor)
    ForkingPickler.register(torch.Tensor, reduce_tensor)
    ForkingPickler.register(torch.nn.parameter.Parameter, reduce_tensor)

