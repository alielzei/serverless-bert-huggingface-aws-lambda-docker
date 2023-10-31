import torch
import warnings
from torch._six import string_classes
from datetime import timedelta
from .constants import default_pg_timeout
from .rendezvous import rendezvous, register_rendezvous_handler
from . import AllreduceOptions, AllreduceCoalescedOptions, BroadcastOptions, GatherOptions, ReduceOptions, ReduceScatterOptions, ScatterOptions
from . import ReduceOp
from . import PrefixStore
_MPI_AVAILABLE = True
_NCCL_AVAILABLE = True
_GLOO_AVAILABLE = True
try:
    from . import ProcessGroupMPI
except ImportError:
    _MPI_AVAILABLE = False
try:
    from . import ProcessGroupNCCL
except ImportError:
    _NCCL_AVAILABLE = False
try:
    from . import ProcessGroupGloo
except ImportError:
    _GLOO_AVAILABLE = False


class Backend(object):
    """
    An enum-like class of available backends: GLOO, NCCL, and MPI.

    The values of this class are lowercase strings, e.g., ``"gloo"``. They can
    be accessed as attributes, e.g., ``Backend.NCCL``.

    This class can be directly called to parse the string, e.g.,
    ``Backend(backend_str)`` will check if ``backend_str`` is valid, and
    return the parsed lowercase string if so. It also accepts uppercase strings,
    e.g., ``Backend("GLOO")`` returns ``"gloo"``.

    .. note:: The entry ``Backend.UNDEFINED`` is present but only used as
              initial value of some fields. Users should neither use it directly
              nor assume its existence.
    """
    UNDEFINED = 'undefined'
    GLOO = 'gloo'
    NCCL = 'nccl'
    MPI = 'mpi'
    TCP = 'tcp'
    
    def __new__(cls, name):
        if not isinstance(name, string_classes):
            raise ValueError('Backend name must be a string, but got: {}'.format(name))
        value = getattr(Backend, name.upper(), Backend.UNDEFINED)
        if value == Backend.TCP:
            raise ValueError('TCP backend has been deprecated. Please use Gloo or MPI backend for collective operations on CPU tensors.')
        elif value == Backend.UNDEFINED:
            raise ValueError("Invalid backend: '{}'".format(name))
        return value

_backend = Backend.UNDEFINED
dist_backend = Backend


class reduce_op(object):
    """
    Deprecated enum-like class for reduction operations: ``SUM``, ``PRODUCT``,
    ``MIN``, and ``MAX``.

    :class:`~torch.distributed.ReduceOp` is recommended to use instead.
    """
    
    def __init__(self):
        for (k, v) in ReduceOp.__members__.items():
            setattr(self, k, v)
        self.__members__ = ReduceOp.__members__
    
    def __getattribute__(self, key):
        warnings.warn('torch.distributed.reduce_op is deprecated, please use torch.distributed.ReduceOp instead')
        return object.__getattribute__(self, key)

reduce_op = reduce_op()


class group(object):
    WORLD = object()



class GroupMember(object):
    WORLD = group.WORLD
    NON_GROUP_MEMBER = object()

_pg_map = {}
_pg_names = {}
_pg_group_ranks = {}
_default_pg = None
_default_pg_init_method = None
_group_count = 0

def _rank_not_in_group(group):
    """
    Helper that checks if the current process's rank is not in a given group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d._rank_not_in_group', '_rank_not_in_group(group)', {'GroupMember': GroupMember, 'group': group}, 1)

def _get_group_rank(group, rank):
    """
    Helper that gets a given group's local rank in the group from a given global
    rank

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d._get_group_rank', '_get_group_rank(group, rank)', {'GroupMember': GroupMember, '_pg_group_ranks': _pg_group_ranks, 'group': group, 'rank': rank}, 1)

def _get_global_rank(group, group_rank):
    """
    Helper that gets a given group's global rank from a given local rank in the
    group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d._get_global_rank', '_get_global_rank(group, group_rank)', {'GroupMember': GroupMember, '_pg_group_ranks': _pg_group_ranks, 'group': group, 'group_rank': group_rank}, 1)

def _check_default_pg():
    """
    Helper that checks if the default ProcessGroup has been initialized, with
    assertion

    """
    assert _default_pg is not None, 'Default process group is not initialized'

def _get_group_size(group):
    """
    Helper that gets a given group's world size

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d._get_group_size', '_get_group_size(group)', {'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, '_pg_group_ranks': _pg_group_ranks, 'group': group}, 1)

def _check_single_tensor(param, param_name):
    """
    Helper to check that the parameter ``param_name`` is a single tensor.

    """
    if not isinstance(param, torch.Tensor):
        raise RuntimeError('Invalid function argument. Expected parameter `{}` to be of type torch.Tensor.'.format(param_name))

def _check_tensor_list(param, param_name):
    """
    Helper to check that the parameter ``param_name`` is a list of tensors.

    """
    if (not isinstance(param, list) or not all((isinstance(p, torch.Tensor) for p in param))):
        raise RuntimeError('Invalid function argument. Expected parameter `{}` to be of type List[torch.Tensor].'.format(param_name))

def is_mpi_available():
    """
    Checks if the MPI backend is available.

    """
    return _MPI_AVAILABLE

def is_nccl_available():
    """
    Checks if the NCCL backend is available.

    """
    return _NCCL_AVAILABLE

def is_gloo_available():
    """
    Checks if the Gloo backend is available.

    """
    return _GLOO_AVAILABLE

def is_initialized():
    """
    Checking if the default process group has been initialized

    """
    return _default_pg is not None

def _get_default_group():
    """
    Getting the default process group created by init_process_group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d._get_default_group', '_get_default_group()', {'is_initialized': is_initialized, '_default_pg': _default_pg}, 1)

def _get_default_store():
    """
    Getting the default store created by init_process_group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d._get_default_store', '_get_default_store()', {'is_initialized': is_initialized, '_pg_map': _pg_map, '_default_pg': _default_pg}, 1)

def get_backend(group=group.WORLD):
    """
    Returns the backend of the given process group.

    Arguments:
        group (ProcessGroup, optional): The process group to work on. The
            default is the general main process group. If another specific group
            is specified, the calling process must be part of :attr:`group`.

    Returns:
        The backend of the given process group as a lower case string.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.get_backend', 'get_backend(group=group.WORLD)', {'_check_default_pg': _check_default_pg, 'GroupMember': GroupMember, '_default_pg': _default_pg, '_rank_not_in_group': _rank_not_in_group, '_pg_map': _pg_map, 'group': group}, 1)

def init_process_group(backend, init_method=None, timeout=default_pg_timeout, world_size=-1, rank=-1, store=None, group_name=''):
    """
    Initializes the default distributed process group, and this will also
    initialize the distributed package.

    There are 2 main ways to initialize a process group:
        1. Specify ``store``, ``rank``, and ``world_size`` explicitly.
        2. Specify ``init_method`` (a URL string) which indicates where/how
           to discover peers. Optionally specify ``rank`` and ``world_size``,
           or encode all required parameters in the URL and omit them.

    If neither is specified, ``init_method`` is assumed to be "env://".


    Arguments:
        backend (str or Backend): The backend to use. Depending on
            build-time configurations, valid values include ``mpi``, ``gloo``,
            and ``nccl``. This field should be given as a lowercase string
            (e.g., ``"gloo"``), which can also be accessed via
            :class:`Backend` attributes (e.g., ``Backend.GLOO``). If using
            multiple processes per machine with ``nccl`` backend, each process
            must have exclusive access to every GPU it uses, as sharing GPUs
            between processes can result in deadlocks.
        init_method (str, optional): URL specifying how to initialize the
                                     process group. Default is "env://" if no
                                     ``init_method`` or ``store`` is specified.
                                     Mutually exclusive with ``store``.
        world_size (int, optional): Number of processes participating in
                                    the job. Required if ``store`` is specified.
        rank (int, optional): Rank of the current process.
                              Required if ``store`` is specified.
        store(Store, optional): Key/value store accessible to all workers, used
                                to exchange connection/address information.
                                Mutually exclusive with ``init_method``.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group. Default value equals 30 minutes.
            This is applicable for the ``gloo`` backend. For ``nccl``, this is
            applicable only if the environment variable ``NCCL_BLOCKING_WAIT``
            is set to 1.
        group_name (str, optional, deprecated): Group name.

    To enable ``backend == Backend.MPI``, PyTorch needs to built from source
    on a system that supports MPI.

    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.init_process_group', "init_process_group(backend, init_method=None, timeout=default_pg_timeout, world_size=-1, rank=-1, store=None, group_name='')", {'timedelta': timedelta, 'Backend': Backend, '_new_process_group_helper': _new_process_group_helper, 'rendezvous': rendezvous, '_pg_group_ranks': _pg_group_ranks, '_pg_map': _pg_map, 'backend': backend, 'init_method': init_method, 'timeout': timeout, 'world_size': world_size, 'rank': rank, 'store': store, 'group_name': group_name, 'default_pg_timeout': default_pg_timeout}, 0)

def _new_process_group_helper(world_size, rank, group_ranks, backend, store, group_name=None, timeout=default_pg_timeout):
    """
    Create a new distributed process group.

    This function must be called by ALL processes in the global group, even if
    the calling process is not part of the newly created group. In that case,
    this function returns GroupMember.NON_GROUP_MEMBER.

    This function is called with ``group_ranks == []`` for the default group.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d._new_process_group_helper', '_new_process_group_helper(world_size, rank, group_ranks, backend, store, group_name=None, timeout=default_pg_timeout)', {'_pg_names': _pg_names, 'timedelta': timedelta, 'Backend': Backend, 'is_mpi_available': is_mpi_available, 'ProcessGroupMPI': ProcessGroupMPI, 'GroupMember': GroupMember, '_pg_map': _pg_map, '_default_pg': _default_pg, 'PrefixStore': PrefixStore, 'ProcessGroupGloo': ProcessGroupGloo, 'is_nccl_available': is_nccl_available, 'ProcessGroupNCCL': ProcessGroupNCCL, 'world_size': world_size, 'rank': rank, 'group_ranks': group_ranks, 'backend': backend, 'store': store, 'group_name': group_name, 'timeout': timeout, 'default_pg_timeout': default_pg_timeout}, 1)

def destroy_process_group(group=group.WORLD):
    """
    Destroy a given process group, and deinitialize the distributed package

    Arguments:
        group (ProcessGroup, optional): The process group to be destroyed, if
                                        group.WORLD is given, all process
                                        groups including the default one will
                                        be destroyed.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.destroy_process_group', 'destroy_process_group(group=group.WORLD)', {'GroupMember': GroupMember, '_pg_map': _pg_map, '_pg_names': _pg_names, '_pg_group_ranks': _pg_group_ranks, 'group': group}, 1)

def get_rank(group=group.WORLD):
    """
    Returns the rank of current process group

    Rank is a unique identifier assigned to each process within a distributed
    process group. They are always consecutive integers ranging from 0 to
    ``world_size``.

    Arguments:
        group (ProcessGroup, optional): The process group to work on

    Returns:
        The rank of the process group
        -1, if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.get_rank', 'get_rank(group=group.WORLD)', {'_rank_not_in_group': _rank_not_in_group, '_check_default_pg': _check_default_pg, 'GroupMember': GroupMember, '_default_pg': _default_pg, '_get_group_rank': _get_group_rank, 'group': group}, 1)

def get_world_size(group=group.WORLD):
    """
    Returns the number of processes in the current process group

    Arguments:
        group (ProcessGroup, optional): The process group to work on

    Returns:
        The world size of the process group
        -1, if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.get_world_size', 'get_world_size(group=group.WORLD)', {'_rank_not_in_group': _rank_not_in_group, '_get_group_size': _get_group_size, 'group': group}, 1)

def isend(tensor, dst, group=group.WORLD, tag=0):
    """
    Sends a tensor asynchronously.

    Arguments:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match send with remote recv

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.isend', 'isend(tensor, dst, group=group.WORLD, tag=0)', {'_check_single_tensor': _check_single_tensor, '_rank_not_in_group': _rank_not_in_group, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, '_get_group_rank': _get_group_rank, 'tensor': tensor, 'dst': dst, 'group': group, 'tag': tag}, 1)

def irecv(tensor, src, group=group.WORLD, tag=0):
    """
    Receives a tensor asynchronously.

    Arguments:
        tensor (Tensor): Tensor to fill with received data.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match recv with remote send

    Returns:
        A distributed request object.
        None, if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.irecv', 'irecv(tensor, src, group=group.WORLD, tag=0)', {'_check_single_tensor': _check_single_tensor, '_rank_not_in_group': _rank_not_in_group, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, '_get_group_rank': _get_group_rank, 'tensor': tensor, 'src': src, 'group': group, 'tag': tag}, 1)

def send(tensor, dst, group=group.WORLD, tag=0):
    """
    Sends a tensor synchronously.

    Arguments:
        tensor (Tensor): Tensor to send.
        dst (int): Destination rank.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match send with remote recv

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.send', 'send(tensor, dst, group=group.WORLD, tag=0)', {'_check_single_tensor': _check_single_tensor, '_rank_not_in_group': _rank_not_in_group, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, '_get_group_rank': _get_group_rank, 'tensor': tensor, 'dst': dst, 'group': group, 'tag': tag}, 1)

def recv(tensor, src=None, group=group.WORLD, tag=0):
    """
    Receives a tensor synchronously.

    Arguments:
        tensor (Tensor): Tensor to fill with received data.
        src (int, optional): Source rank. Will receive from any
            process if unspecified.
        group (ProcessGroup, optional): The process group to work on
        tag (int, optional): Tag to match recv with remote send

    Returns:
        Sender rank
        -1, if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.recv', 'recv(tensor, src=None, group=group.WORLD, tag=0)', {'_check_single_tensor': _check_single_tensor, '_rank_not_in_group': _rank_not_in_group, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, '_get_global_rank': _get_global_rank, '_get_group_rank': _get_group_rank, 'tensor': tensor, 'src': src, 'group': group, 'tag': tag}, 1)

def broadcast_multigpu(tensor_list, src, group=group.WORLD, async_op=False, src_tensor=0):
    """
    Broadcasts the tensor to the whole group with multiple GPU tensors
    per node.

    ``tensor`` must have the same number of elements in all the GPUs from
    all processes participating in the collective. each tensor in the list must
    be on a different GPU

    Only nccl and gloo backend are currently supported
    tensors should only be GPU tensors

    Arguments:
        tensor_list (List[Tensor]): Tensors that participate in the collective
            operation. If ``src`` is the rank, then the specified ``src_tensor``
            element of ``tensor_list`` (``tensor_list[src_tensor]``) will be
            broadcast to all other tensors (on different GPUs) in the src process
            and all tensors in ``tensor_list`` of other non-src processes.
            You also need to make sure that ``len(tensor_list)`` is the same
            for all the distributed processes calling this function.

        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op
        src_tensor (int, optional): Source tensor rank within ``tensor_list``

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.broadcast_multigpu', 'broadcast_multigpu(tensor_list, src, group=group.WORLD, async_op=False, src_tensor=0)', {'_rank_not_in_group': _rank_not_in_group, 'BroadcastOptions': BroadcastOptions, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, '_get_group_rank': _get_group_rank, 'tensor_list': tensor_list, 'src': src, 'group': group, 'async_op': async_op, 'src_tensor': src_tensor}, 1)

def broadcast(tensor, src, group=group.WORLD, async_op=False):
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process, and tensor to be used to save received data otherwise.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.broadcast', 'broadcast(tensor, src, group=group.WORLD, async_op=False)', {'_check_single_tensor': _check_single_tensor, '_rank_not_in_group': _rank_not_in_group, 'BroadcastOptions': BroadcastOptions, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, '_get_group_rank': _get_group_rank, 'tensor': tensor, 'src': src, 'group': group, 'async_op': async_op}, 1)

def all_reduce_multigpu(tensor_list, op=ReduceOp.SUM, group=group.WORLD, async_op=False):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result. This function reduces a number of tensors on every node,
    while each tensor resides on different GPUs.
    Therefore, the input tensor in the tensor list needs to be GPU tensors.
    Also, each tensor in the tensor list needs to reside on a different GPU.

    After the call, all ``tensor`` in ``tensor_list`` is going to be bitwise
    identical in all processes.

    Only nccl and gloo backend is currently supported
    tensors should only be GPU tensors

    Arguments:
        tensor list (List[Tensor]): List of input and output tensors of
            the collective. The function operates in-place and requires that
            each tensor to be a GPU tensor on different GPUs.
            You also need to make sure that ``len(tensor_list)`` is the same for
            all the distributed processes calling this function.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.all_reduce_multigpu', 'all_reduce_multigpu(tensor_list, op=ReduceOp.SUM, group=group.WORLD, async_op=False)', {'_rank_not_in_group': _rank_not_in_group, 'AllreduceOptions': AllreduceOptions, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, 'tensor_list': tensor_list, 'op': op, 'group': group, 'async_op': async_op}, 1)

def all_reduce(tensor, op=ReduceOp.SUM, group=group.WORLD, async_op=False):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    After the call ``tensor`` is going to be bitwise identical in all processes.

    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.all_reduce', 'all_reduce(tensor, op=ReduceOp.SUM, group=group.WORLD, async_op=False)', {'_check_single_tensor': _check_single_tensor, '_rank_not_in_group': _rank_not_in_group, 'AllreduceOptions': AllreduceOptions, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, 'tensor': tensor, 'op': op, 'group': group, 'async_op': async_op}, 1)

def all_reduce_coalesced(tensors, op=ReduceOp.SUM, group=group.WORLD, async_op=False):
    """
    WARNING: at this time individual shape checking is not implemented across nodes.
    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the allreduce
    operation will proceed without complaint and return erroneous outputs. This lack
    of shape checking results in significant performance improvements but users of this
    function should take extra care to ensure that each node passes in tensors whose
    shapes match across nodes.

    Reduces each tensor in tensors (residing on the same device) across all machines
    in such a way that all get the final result.

    After the call each tensor in tensors is going to bitwise identical
    in all processes.

    Arguments:
        tensors (List[Tensor]): Input and output of the collective. The function
            operates in-place.
        op (Optional[ReduceOp]): One of the values from
            ``torch.distributed.ReduceOp`` enum. Specifies an operation used for
            element-wise reductions.
        group (Optional[ProcessGroup]): The process group to work on.
        async_op (Optional[bool]): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.all_reduce_coalesced', 'all_reduce_coalesced(tensors, op=ReduceOp.SUM, group=group.WORLD, async_op=False)', {'_check_tensor_list': _check_tensor_list, '_rank_not_in_group': _rank_not_in_group, 'AllreduceCoalescedOptions': AllreduceCoalescedOptions, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, 'tensors': tensors, 'op': op, 'group': group, 'async_op': async_op}, 1)

def reduce_multigpu(tensor_list, dst, op=ReduceOp.SUM, group=group.WORLD, async_op=False, dst_tensor=0):
    """
    Reduces the tensor data on multiple GPUs across all machines. Each tensor
    in ``tensor_list`` should reside on a separate GPU

    Only the GPU of ``tensor_list[dst_tensor]`` on the process with rank ``dst``
    is going to receive the final result.

    Only nccl backend is currently supported
    tensors should only be GPU tensors

    Arguments:
        tensor_list (List[Tensor]): Input and output GPU tensors of the
            collective. The function operates in-place.
            You also need to make sure that ``len(tensor_list)`` is the same for
            all the distributed processes calling this function.
        dst (int): Destination rank
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op
        dst_tensor (int, optional): Destination tensor rank within
                                    ``tensor_list``

    Returns:
        Async work handle, if async_op is set to True.
        None, otherwise

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.reduce_multigpu', 'reduce_multigpu(tensor_list, dst, op=ReduceOp.SUM, group=group.WORLD, async_op=False, dst_tensor=0)', {'_rank_not_in_group': _rank_not_in_group, 'ReduceOptions': ReduceOptions, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, '_get_group_rank': _get_group_rank, 'tensor_list': tensor_list, 'dst': dst, 'op': op, 'group': group, 'async_op': async_op, 'dst_tensor': dst_tensor}, 1)

def reduce(tensor, dst, op=ReduceOp.SUM, group=group.WORLD, async_op=False):
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        dst (int): Destination rank
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.reduce', 'reduce(tensor, dst, op=ReduceOp.SUM, group=group.WORLD, async_op=False)', {'_check_single_tensor': _check_single_tensor, '_rank_not_in_group': _rank_not_in_group, 'ReduceOptions': ReduceOptions, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, '_get_group_rank': _get_group_rank, 'tensor': tensor, 'dst': dst, 'op': op, 'group': group, 'async_op': async_op}, 1)

def all_gather_multigpu(output_tensor_lists, input_tensor_list, group=group.WORLD, async_op=False):
    """
    Gathers tensors from the whole group in a list.
    Each tensor in ``tensor_list`` should reside on a separate GPU

    Only nccl backend is currently supported
    tensors should only be GPU tensors

    Arguments:
        output_tensor_lists (List[List[Tensor]]): Output lists. It should
            contain correctly-sized tensors on each GPU to be used for output
            of the collective, e.g. ``output_tensor_lists[i]`` contains the
            all_gather result that resides on the GPU of
            ``input_tensor_list[i]``.

            Note that each element of ``output_tensor_lists`` has the size of
            ``world_size * len(input_tensor_list)``, since the function all
            gathers the result from every single GPU in the group. To interpret
            each element of ``output_tensor_lists[i]``, note that
            ``input_tensor_list[j]`` of rank k will be appear in
            ``output_tensor_lists[i][k * world_size + j]``

            Also note that ``len(output_tensor_lists)``, and the size of each
            element in ``output_tensor_lists`` (each element is a list,
            therefore ``len(output_tensor_lists[i])``) need to be the same
            for all the distributed processes calling this function.

        input_tensor_list (List[Tensor]): List of tensors(on different GPUs) to
            be broadcast from current process.
            Note that ``len(input_tensor_list)`` needs to be the same for
            all the distributed processes calling this function.

        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.all_gather_multigpu', 'all_gather_multigpu(output_tensor_lists, input_tensor_list, group=group.WORLD, async_op=False)', {'_rank_not_in_group': _rank_not_in_group, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, 'output_tensor_lists': output_tensor_lists, 'input_tensor_list': input_tensor_list, 'group': group, 'async_op': async_op}, 1)

def all_gather(tensor_list, tensor, group=group.WORLD, async_op=False):
    """
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor_list (list[Tensor]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.all_gather', 'all_gather(tensor_list, tensor, group=group.WORLD, async_op=False)', {'_check_tensor_list': _check_tensor_list, '_check_single_tensor': _check_single_tensor, '_rank_not_in_group': _rank_not_in_group, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, 'tensor_list': tensor_list, 'tensor': tensor, 'group': group, 'async_op': async_op}, 1)

def all_gather_coalesced(output_tensor_lists, input_tensor_list, group=group.WORLD, async_op=False):
    """
    Gathers input tensors from the whole group in a list in a coalesced manner.

    Arguments:
        output_tensor_lists (list[list[Tensor]]): Output list. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor_list (list[Tensor]): Tensors to be broadcast from
            current process. At least one tensor has to be non empty.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    Example:
        we have 2 process groups, 2 ranks.
        rank 0 passes:
            input_tensor_list = [[[1, 1], [1, 1]], [2], [3, 3]]
            output_tensor_lists =
               [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],
                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]
        rank 1 passes:
            input_tensor_list = [[[3, 3], [3, 3]], [5], [1, 1]]
            output_tensor_lists =
               [[[[-1, -1], [-1, -1]], [-1], [-1, -1]],
                [[[-1, -1], [-1, -1]], [-1], [-1, -1]]]
        both rank 0 and 1 get:
            output_tensor_lists =
               [[[1, 1], [1, 1]], [2], [3, 3]],
                [[3, 3], [3, 3]], [5], [1, 1]]].

    WARNING: at this time individual shape checking is not implemented across nodes.
    For example, if the rank 0 node passes [torch.rand(4), torch.rand(2)] and the
    rank 1 node passes [torch.rand(2), torch.rand(2), torch.rand(2)], the
    all_gather_coalesced operation will proceed without complaint and return
    erroneous outputs. This lack of shape checking results in significant
    performance improvements but users of this function should take extra care
    to ensure that each node passes in tensors whose shapes match across nodes.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.all_gather_coalesced', 'all_gather_coalesced(output_tensor_lists, input_tensor_list, group=group.WORLD, async_op=False)', {'_rank_not_in_group': _rank_not_in_group, '_check_tensor_list': _check_tensor_list, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, 'output_tensor_lists': output_tensor_lists, 'input_tensor_list': input_tensor_list, 'group': group, 'async_op': async_op}, 1)

def gather(tensor, gather_list=None, dst=0, group=group.WORLD, async_op=False):
    """
    Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        gather_list (list[Tensor], optional): List of appropriately-sized
            tensors to use for gathered data (default is None, must be specified
            on the destination rank)
        dst (int, optional): Destination rank (default is 0)
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.gather', 'gather(tensor, gather_list=None, dst=0, group=group.WORLD, async_op=False)', {'_check_single_tensor': _check_single_tensor, '_check_tensor_list': _check_tensor_list, '_rank_not_in_group': _rank_not_in_group, 'get_rank': get_rank, 'GatherOptions': GatherOptions, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, '_get_group_rank': _get_group_rank, 'tensor': tensor, 'gather_list': gather_list, 'dst': dst, 'group': group, 'async_op': async_op}, 1)

def scatter(tensor, scatter_list=None, src=0, group=group.WORLD, async_op=False):
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensor (Tensor): Output tensor.
        scatter_list (list[Tensor]): List of tensors to scatter (default is
            None, must be specified on the source rank)
        src (int): Source rank (default is 0)
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.scatter', 'scatter(tensor, scatter_list=None, src=0, group=group.WORLD, async_op=False)', {'_check_single_tensor': _check_single_tensor, '_check_tensor_list': _check_tensor_list, '_rank_not_in_group': _rank_not_in_group, 'get_rank': get_rank, 'ScatterOptions': ScatterOptions, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, '_get_group_rank': _get_group_rank, 'tensor': tensor, 'scatter_list': scatter_list, 'src': src, 'group': group, 'async_op': async_op}, 1)

def reduce_scatter_multigpu(output_tensor_list, input_tensor_lists, op=ReduceOp.SUM, group=group.WORLD, async_op=False):
    """
    Reduce and scatter a list of tensors to the whole group.  Only nccl backend
    is currently supported.

    Each tensor in ``output_tensor_list`` should reside on a separate GPU, as
    should each list of tensors in ``input_tensor_lists``.

    Arguments:
        output_tensor_list (List[Tensor]): Output tensors (on different GPUs)
            to receive the result of the operation.

            Note that ``len(output_tensor_list)`` needs to be the same for all
            the distributed processes calling this function.

        input_tensor_lists (List[List[Tensor]]): Input lists.  It should
            contain correctly-sized tensors on each GPU to be used for input of
            the collective, e.g. ``input_tensor_lists[i]`` contains the
            reduce_scatter input that resides on the GPU of
            ``output_tensor_list[i]``.

            Note that each element of ``input_tensor_lists`` has the size of
            ``world_size * len(output_tensor_list)``, since the function
            scatters the result from every single GPU in the group.  To
            interpret each element of ``input_tensor_lists[i]``, note that
            ``output_tensor_list[j]`` of rank k receives the reduce-scattered
            result from ``input_tensor_lists[i][k * world_size + j]``

            Also note that ``len(input_tensor_lists)``, and the size of each
            element in ``input_tensor_lists`` (each element is a list,
            therefore ``len(input_tensor_lists[i])``) need to be the same for
            all the distributed processes calling this function.

        group (ProcessGroup, optional): The process group to work on.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.reduce_scatter_multigpu', 'reduce_scatter_multigpu(output_tensor_list, input_tensor_lists, op=ReduceOp.SUM, group=group.WORLD, async_op=False)', {'_rank_not_in_group': _rank_not_in_group, 'ReduceScatterOptions': ReduceScatterOptions, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, 'output_tensor_list': output_tensor_list, 'input_tensor_lists': input_tensor_lists, 'op': op, 'group': group, 'async_op': async_op}, 1)

def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=group.WORLD, async_op=False):
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Arguments:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        group (ProcessGroup, optional): The process group to work on.
        async_op (bool, optional): Whether this op should be an async op.

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.reduce_scatter', 'reduce_scatter(output, input_list, op=ReduceOp.SUM, group=group.WORLD, async_op=False)', {'_check_single_tensor': _check_single_tensor, '_check_tensor_list': _check_tensor_list, '_rank_not_in_group': _rank_not_in_group, 'ReduceScatterOptions': ReduceScatterOptions, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, 'output': output, 'input_list': input_list, 'op': op, 'group': group, 'async_op': async_op}, 1)

def barrier(group=group.WORLD, async_op=False):
    """
    Synchronizes all processes.

    This collective blocks processes until the whole group enters this function,
    if async_op is False, or if async work handle is called on wait().

    Arguments:
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op

    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.barrier', 'barrier(group=group.WORLD, async_op=False)', {'_rank_not_in_group': _rank_not_in_group, 'GroupMember': GroupMember, '_check_default_pg': _check_default_pg, '_default_pg': _default_pg, 'group': group, 'async_op': async_op}, 1)

def new_group(ranks=None, timeout=default_pg_timeout, backend=None):
    """
    Creates a new distributed group.

    This function requires that all processes in the main group (i.e. all
    processes that are part of the distributed job) enter this function, even
    if they are not going to be members of the group. Additionally, groups
    should be created in the same order in all processes.

    Arguments:
        ranks (list[int]): List of ranks of group members.
        timeout (timedelta, optional): Timeout for operations executed against
            the process group. Default value equals 30 minutes.
            This is only applicable for the ``gloo`` backend.
        backend (str or Backend, optional): The backend to use. Depending on
            build-time configurations, valid values are ``gloo`` and ``nccl``.
            By default uses the same backend as the global group. This field
            should be given as a lowercase string (e.g., ``"gloo"``), which can
            also be accessed via :class:`Backend` attributes (e.g.,
            ``Backend.GLOO``).

    Returns:
        A handle of distributed group that can be given to collective calls.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.distributed_c10d.new_group', 'new_group(ranks=None, timeout=default_pg_timeout, backend=None)', {'_check_default_pg': _check_default_pg, '_pg_map': _pg_map, '_default_pg': _default_pg, 'Backend': Backend, '_new_process_group_helper': _new_process_group_helper, '_pg_group_ranks': _pg_group_ranks, 'ranks': ranks, 'timeout': timeout, 'backend': backend, 'default_pg_timeout': default_pg_timeout}, 1)

