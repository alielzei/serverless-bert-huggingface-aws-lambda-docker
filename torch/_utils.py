import torch
import warnings
from collections import defaultdict
import sys
import traceback

def _type(self, dtype=None, non_blocking=False, **kwargs):
    """Returns the type if `dtype` is not provided, else casts this object to
    the specified type.

    If this is already of the correct type, no copy is performed and the
    original object is returned.

    Args:
        dtype (type or string): The desired type
        non_blocking (bool): If ``True``, and the source is in pinned memory
            and destination is on the GPU or vice versa, the copy is performed
            asynchronously with respect to the host. Otherwise, the argument
            has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument. The ``async`` arg is deprecated.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._type', '_type(self, dtype=None, non_blocking=False, **kwargs)', {'_get_async_or_non_blocking': _get_async_or_non_blocking, '_import_dotted_name': _import_dotted_name, 'torch': torch, 'self': self, 'dtype': dtype, 'non_blocking': non_blocking, 'kwargs': kwargs}, 1)

def _cuda(self, device=None, non_blocking=False, **kwargs):
    """Returns a copy of this object in CUDA memory.

    If this object is already in CUDA memory and on the correct device, then
    no copy is performed and the original object is returned.

    Args:
        device (int): The destination GPU id. Defaults to the current device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._cuda', '_cuda(self, device=None, non_blocking=False, **kwargs)', {'_get_async_or_non_blocking': _get_async_or_non_blocking, 'torch': torch, 'self': self, 'device': device, 'non_blocking': non_blocking, 'kwargs': kwargs}, 1)

def _get_async_or_non_blocking(function_name, non_blocking, kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._get_async_or_non_blocking', '_get_async_or_non_blocking(function_name, non_blocking, kwargs)', {'warnings': warnings, 'function_name': function_name, 'non_blocking': non_blocking, 'kwargs': kwargs}, 1)

def _rebuild_tensor(storage, storage_offset, size, stride):
    t = torch.tensor([], dtype=storage.dtype, device=storage.device)
    return t.set_(storage, storage_offset, size, stride)

def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._rebuild_tensor_v2', '_rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks)', {'_rebuild_tensor': _rebuild_tensor, 'storage': storage, 'storage_offset': storage_offset, 'size': size, 'stride': stride, 'requires_grad': requires_grad, 'backward_hooks': backward_hooks}, 1)

def _rebuild_sparse_tensor(layout, data):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._rebuild_sparse_tensor', '_rebuild_sparse_tensor(layout, data)', {'torch': torch, 'layout': layout, 'data': data}, 1)

def _rebuild_xla_tensor(data, dtype, device, requires_grad):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._rebuild_xla_tensor', '_rebuild_xla_tensor(data, dtype, device, requires_grad)', {'torch': torch, 'data': data, 'dtype': dtype, 'device': device, 'requires_grad': requires_grad}, 1)

def _rebuild_qtensor(storage, storage_offset, size, stride, quantizer_params, requires_grad, backward_hooks):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._rebuild_qtensor', '_rebuild_qtensor(storage, storage_offset, size, stride, quantizer_params, requires_grad, backward_hooks)', {'torch': torch, 'storage': storage, 'storage_offset': storage_offset, 'size': size, 'stride': stride, 'quantizer_params': quantizer_params, 'requires_grad': requires_grad, 'backward_hooks': backward_hooks}, 1)

def _rebuild_parameter(data, requires_grad, backward_hooks):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._rebuild_parameter', '_rebuild_parameter(data, requires_grad, backward_hooks)', {'torch': torch, 'data': data, 'requires_grad': requires_grad, 'backward_hooks': backward_hooks}, 1)

def _import_dotted_name(name):
    components = name.split('.')
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj

def _accumulate(iterable, fn=lambda x, y: x + y):
    """Return running totals"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._accumulate', '_accumulate(iterable, fn=lambda x, y: x + y)', {'iterable': iterable, 'fn': fn}, 1)

def _flatten_dense_tensors(tensors):
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._flatten_dense_tensors', '_flatten_dense_tensors(tensors)', {'torch': torch, 'tensors': tensors}, 1)

def _flatten_sparse_tensors(tensors):
    """Flatten sparse tensors into two contiguous 1D buffers, one of indices and
    one of values. Assume tensors are of same sparse type.

    Arguments:
        tensors (Iterable[Tensor]): sparse tensors to flatten.

    Returns:
        A tuple of two contiguous 1D buffers, one containing input tensors'
        indices and the other containing the values.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._flatten_sparse_tensors', '_flatten_sparse_tensors(tensors)', {'_flatten_dense_tensors': _flatten_dense_tensors, 'torch': torch, 'tensors': tensors}, 2)

def _unflatten_dense_tensors(flat, tensors):
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._unflatten_dense_tensors', '_unflatten_dense_tensors(flat, tensors)', {'flat': flat, 'tensors': tensors}, 1)

def _unflatten_sparse_tensors(flat, tensors):
    """View flat buffer (containing indices and values) using the sizes of
    tensors. Assume that tensors are of same sparse type, and that flat is given
    by _flatten_sparse_tensors.

    Arguments:
        flat (tuple(Tensor, Tensor)): flattened indices and values of sparse
          tensors to unflatten.
        tensors (Iterable[Tensor]): sparse tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened sparse tensors with sizes same as tensors and values from
        flat.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._unflatten_sparse_tensors', '_unflatten_sparse_tensors(flat, tensors)', {'_unflatten_dense_tensors': _unflatten_dense_tensors, 'torch': torch, 'flat': flat, 'tensors': tensors}, 1)

def _reorder_tensors_as(tensors, ordered_tensors):
    """Assume that tensors are of same order as ordered_tensors within their
    types, e.g., from _take_tensors. Reorder them to be of same order as
    ordered_tensors.

    Arguments:
        tensors (Iterable[Tensor]): tensors to be reordered. They should be of
          the same order as ordered_tensors within their own types.
        ordered_tensors (Iterable[Tensor]): tensors whose order will be the
          reference.

    Returns:
        Ordered tuple of tensors with contents from tensors and order of
        ordered_tensors.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils._reorder_tensors_as', '_reorder_tensors_as(tensors, ordered_tensors)', {'defaultdict': defaultdict, 'tensors': tensors, 'ordered_tensors': ordered_tensors}, 1)

def _take_tensors(tensors, size_limit):
    """Group tensors into chunks. This generator yields a chunk at each time,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Yields:
        Blocks of tensors of same type and within size_limit. The yielded
        tensors are only ordered as the original sequence within its types.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch._utils._take_tensors', '_take_tensors(tensors, size_limit)', {'defaultdict': defaultdict, 'torch': torch, 'tensors': tensors, 'size_limit': size_limit}, 0)

def annotate(ret, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._utils.annotate', 'annotate(ret, **kwargs)', {'ret': ret, 'kwargs': kwargs}, 1)


class KeyErrorMessage(str):
    """str subclass that returns itself in repr"""
    
    def __repr__(self):
        return self



class ExceptionWrapper(object):
    """Wraps an exception plus traceback to communicate across threads"""
    
    def __init__(self, exc_info=None, where='in background'):
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = ''.join(traceback.format_exception(*exc_info))
        self.where = where
    
    def reraise(self):
        """Reraises the wrapped exception in the current thread"""
        msg = 'Caught {} {}.\nOriginal {}'.format(self.exc_type.__name__, self.where, self.exc_msg)
        if self.exc_type == KeyError:
            msg = KeyErrorMessage(msg)
        raise self.exc_type(msg)


