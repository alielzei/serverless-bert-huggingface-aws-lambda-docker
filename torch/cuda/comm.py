import torch
from . import nccl
from torch._utils import _take_tensors, _flatten_dense_tensors, _unflatten_dense_tensors, _reorder_tensors_as

def broadcast(tensor, devices):
    """Broadcasts a tensor to a number of GPUs.

    Arguments:
        tensor (Tensor): tensor to broadcast.
        devices (Iterable): an iterable of devices among which to broadcast.
          Note that it should be like (src, dst1, dst2, ...), the first element
          of which is the source device to broadcast from.

    Returns:
        A tuple containing copies of the ``tensor``, placed on devices
        corresponding to indices from ``devices``.
    """
    return torch._C._broadcast(tensor, devices)

def broadcast_coalesced(tensors, devices, buffer_size=10485760):
    """Broadcasts a sequence tensors to the specified GPUs.
    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Arguments:
        tensors (sequence): tensors to broadcast.
        devices (Iterable): an iterable of devices among which to broadcast.
          Note that it should be like (src, dst1, dst2, ...), the first element
          of which is the source device to broadcast from.
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple containing copies of the ``tensor``, placed on devices
        corresponding to indices from ``devices``.
    """
    return torch._C._broadcast_coalesced(tensors, devices, buffer_size)

def reduce_add(inputs, destination=None):
    """Sums tensors from multiple GPUs.

    All inputs should have matching shapes.

    Arguments:
        inputs (Iterable[Tensor]): an iterable of tensors to add.
        destination (int, optional): a device on which the output will be
            placed (default: current device).

    Returns:
        A tensor containing an elementwise sum of all inputs, placed on the
        ``destination`` device.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.cuda.comm.reduce_add', 'reduce_add(inputs, destination=None)', {'torch': torch, 'nccl': nccl, 'inputs': inputs, 'destination': destination}, 1)

def reduce_add_coalesced(inputs, destination=None, buffer_size=10485760):
    """Sums tensors from multiple GPUs.

    Small tensors are first coalesced into a buffer to reduce the number
    of synchronizations.

    Arguments:
        inputs (Iterable[Iterable[Tensor]]): iterable of iterables that
            contain tensors from a single device.
        destination (int, optional): a device on which the output will be
            placed (default: current device).
        buffer_size (int): maximum size of the buffer used for coalescing

    Returns:
        A tuple of tensors containing an elementwise sum of each group of
        inputs, placed on the ``destination`` device.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.cuda.comm.reduce_add_coalesced', 'reduce_add_coalesced(inputs, destination=None, buffer_size=10485760)', {'reduce_add': reduce_add, '_take_tensors': _take_tensors, '_flatten_dense_tensors': _flatten_dense_tensors, '_unflatten_dense_tensors': _unflatten_dense_tensors, '_reorder_tensors_as': _reorder_tensors_as, 'inputs': inputs, 'destination': destination, 'buffer_size': buffer_size}, 1)

def scatter(tensor, devices, chunk_sizes=None, dim=0, streams=None):
    """Scatters tensor across multiple GPUs.

    Arguments:
        tensor (Tensor): tensor to scatter.
        devices (Iterable[int]): iterable of ints, specifying among which
            devices the tensor should be scattered.
        chunk_sizes (Iterable[int], optional): sizes of chunks to be placed on
            each device. It should match ``devices`` in length and sum to
            ``tensor.size(dim)``. If not specified, the tensor will be divided
            into equal chunks.
        dim (int, optional): A dimension along which to chunk the tensor.

    Returns:
        A tuple containing chunks of the ``tensor``, spread across given
        ``devices``.
    """
    return tuple(torch._C._scatter(tensor, devices, chunk_sizes, dim, streams))

def gather(tensors, dim=0, destination=None):
    """Gathers tensors from multiple GPUs.

    Tensor sizes in all dimension different than ``dim`` have to match.

    Arguments:
        tensors (Iterable[Tensor]): iterable of tensors to gather.
        dim (int): a dimension along which the tensors will be concatenated.
        destination (int, optional): output device (-1 means CPU, default:
            current device)

    Returns:
        A tensor located on ``destination`` device, that is a result of
        concatenating ``tensors`` along ``dim``.
    """
    return torch._C._gather(tensors, dim, destination)

