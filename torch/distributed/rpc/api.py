import collections
import contextlib
import functools
import logging
import numbers
import sys
import threading
from datetime import timedelta
import torch
import torch.distributed as dist
from . import RpcBackendOptions, WorkerInfo, _cleanup_python_rpc_handler, _delete_all_user_rrefs, _destroy_rref_context, _get_current_rpc_agent, _invoke_remote_builtin, _invoke_remote_python_udf, _invoke_remote_torchscript, _invoke_rpc_builtin, _invoke_rpc_python_udf, _invoke_rpc_torchscript, _is_current_rpc_agent_set, _reset_current_rpc_agent, _set_and_start_rpc_agent, _set_rpc_timeout, backend_registry
from .internal import PythonUDF, RPCExecMode, _internal_rpc_pickler, _start_record_function
logger = logging.getLogger(__name__)
_ignore_rref_leak = True
_default_pickler = _internal_rpc_pickler

@contextlib.contextmanager
def _use_rpc_pickler(rpc_pickler):
    """
    rpc_pickler: (.internal._InternalRPCPickler) Overrides the default RPC pickler
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.distributed.rpc.api._use_rpc_pickler', '_use_rpc_pickler(rpc_pickler)', {'_internal_rpc_pickler': _internal_rpc_pickler, 'contextlib': contextlib, 'rpc_pickler': rpc_pickler}, 0)

def _require_initialized(func):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rpc.api._require_initialized', '_require_initialized(func)', {'functools': functools, '_is_current_rpc_agent_set': _is_current_rpc_agent_set, 'func': func}, 1)


class WaitAllWorkersStates(object):
    
    def __init__(self):
        self.intent_worker_names = set()
        self.proceed_signal = threading.Event()

_ALL_WORKER_NAMES = None
_wait_all_workers_dict_lock = threading.Lock()
_wait_all_workers_sequence_id = 0
_wait_all_workers_sequence_id_to_states = collections.defaultdict(WaitAllWorkersStates)

def _on_leader_follower_report_shutdown_intent(sequence_id, worker_name):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.distributed.rpc.api._on_leader_follower_report_shutdown_intent', '_on_leader_follower_report_shutdown_intent(sequence_id, worker_name)', {'_ALL_WORKER_NAMES': _ALL_WORKER_NAMES, '_wait_all_workers_sequence_id_to_states': _wait_all_workers_sequence_id_to_states, '_set_proceed_shutdown_signal': _set_proceed_shutdown_signal, 'sequence_id': sequence_id, 'worker_name': worker_name}, 0)

def _set_proceed_shutdown_signal(sequence_id):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.distributed.rpc.api._set_proceed_shutdown_signal', '_set_proceed_shutdown_signal(sequence_id)', {'_wait_all_workers_sequence_id_to_states': _wait_all_workers_sequence_id_to_states, 'sequence_id': sequence_id}, 0)

@_require_initialized
def _wait_all_workers():
    """
    Block until all local and remote RPC processes reach this method and wait
    for all outstanding work to complete. Every RPC process must call this
    method before exit to perform a graceful shutdown. This should be used to
    terminate the RPC framework, and there is no guarantee that the RPC
    framework will work after this method returns.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.distributed.rpc.api._wait_all_workers', '_wait_all_workers()', {'_ALL_WORKER_NAMES': _ALL_WORKER_NAMES, '_get_current_rpc_agent': _get_current_rpc_agent, '_wait_all_workers_dict_lock': _wait_all_workers_dict_lock, '_on_leader_follower_report_shutdown_intent': _on_leader_follower_report_shutdown_intent, 'rpc_sync': rpc_sync, '_wait_all_workers_sequence_id_to_states': _wait_all_workers_sequence_id_to_states, 'timedelta': timedelta, '_set_rpc_timeout': _set_rpc_timeout, 'rpc_async': rpc_async, '_set_proceed_shutdown_signal': _set_proceed_shutdown_signal, 'logger': logger, '_require_initialized': _require_initialized}, 0)

@_require_initialized
def shutdown(graceful=True):
    """
    Perform a shutdown of the RPC agent, and then destroy the RPC agent. This
    stops the local agent from accepting outstanding requests, and shuts
    down the RPC framework by terminating all RPC threads. If ``graceful=True``,
    this will block until all local and remote RPC processes reach this method
    and wait for all outstanding work to complete. Otherwise, if
    ``graceful=False``, this is a local shutdown, and it does not wait for other
    RPC processes to reach this method.

    Arguments:
        graceful (bool): Whether to do a graceful shutdown or not. If True,
                         this will 1) wait until there is no pending system
                         messages for ``UserRRefs`` and delete them; 2) block
                         until all local and remote RPC processes have reached
                         this method and wait for all outstanding work to
                         complete.

    Example::
        Make sure that ``MASTER_ADDRESS`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        >>> export MASTER_ADDRESS=localhost
        >>> export MASTER_port=5678

        Then run the following code in two different processes:

        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> # do some work
        >>> result = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(1), 1))
        >>> # ready to shutdown
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> # wait for worker 0 to finish work, and then shutdown.
        >>> rpc.shutdown()
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.distributed.rpc.api.shutdown', 'shutdown(graceful=True)', {'_wait_all_workers': _wait_all_workers, '_delete_all_user_rrefs': _delete_all_user_rrefs, '_get_current_rpc_agent': _get_current_rpc_agent, '_destroy_rref_context': _destroy_rref_context, '_ignore_rref_leak': _ignore_rref_leak, '_cleanup_python_rpc_handler': _cleanup_python_rpc_handler, '_reset_current_rpc_agent': _reset_current_rpc_agent, '_require_initialized': _require_initialized, 'graceful': graceful}, 0)

def _init_rpc_backend(backend=backend_registry.BackendType.PROCESS_GROUP, store=None, name=None, rank=-1, world_size=-1, rpc_backend_options=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.distributed.rpc.api._init_rpc_backend', '_init_rpc_backend(backend=backend_registry.BackendType.PROCESS_GROUP, store=None, name=None, rank=-1, world_size=-1, rpc_backend_options=None)', {'sys': sys, '_validate_rpc_args': _validate_rpc_args, '_is_current_rpc_agent_set': _is_current_rpc_agent_set, 'backend_registry': backend_registry, '_set_and_start_rpc_agent': _set_and_start_rpc_agent, 'backend': backend, 'store': store, 'name': name, 'rank': rank, 'world_size': world_size, 'rpc_backend_options': rpc_backend_options}, 0)

@_require_initialized
def get_worker_info(worker_name=None):
    """
    Get :class:`~torch.distributed.rpc.WorkerInfo` of a given worker name.
    Use this :class:`~torch.distributed.rpc.WorkerInfo` to avoid passing an
    expensive string on every invocation.

    Arguments:
        worker_name (str): the string name of a worker. If ``None``, return the
                           the id of the current worker. (default ``None``)

    Returns:
        :class:`~torch.distributed.rpc.WorkerInfo` instance for the given
        ``worker_name`` or :class:`~torch.distributed.rpc.WorkerInfo` of the
        current worker if ``worker_name`` is ``None``.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rpc.api.get_worker_info', 'get_worker_info(worker_name=None)', {'_get_current_rpc_agent': _get_current_rpc_agent, '_require_initialized': _require_initialized, 'worker_name': worker_name}, 1)

def _to_worker_info(name_or_info):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rpc.api._to_worker_info', '_to_worker_info(name_or_info)', {'WorkerInfo': WorkerInfo, 'get_worker_info': get_worker_info, 'name_or_info': name_or_info}, 1)

def _validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.distributed.rpc.api._validate_rpc_args', '_validate_rpc_args(backend, store, name, rank, world_size, rpc_backend_options)', {'backend_registry': backend_registry, 'dist': dist, 'numbers': numbers, 'RpcBackendOptions': RpcBackendOptions, 'backend': backend, 'store': store, 'name': name, 'rank': rank, 'world_size': world_size, 'rpc_backend_options': rpc_backend_options}, 0)

@_require_initialized
def remote(to, func, args=None, kwargs=None):
    """
    Make a remote call to run ``func`` on worker ``to`` and return an
    :class:`~torch.distributed.rpc.RRef` to the result value immediately.
    Worker ``to`` will be the owner of the returned
    :class:`~torch.distributed.rpc.RRef`, and the worker calling ``remote`` is
    a user. The owner manages the global reference count of its
    :class:`~torch.distributed.rpc.RRef`, and the owner
    :class:`~torch.distributed.rpc.RRef` is only destructed when globally there
    are no living references to it.

    Arguments:
        to (str or WorkerInfo): id or name of the destination worker.
        func (callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        A user :class:`~torch.distributed.rpc.RRef` instance to the result
        value. Use the blocking API :meth:`torch.distributed.rpc.RRef.to_here`
        to retrieve the result value locally.

    .. warning ::
        Using GPU tensors as arguments or return values of ``func`` is not
        supported since we don't support sending GPU tensors over the wire. You
        need to explicitly copy GPU tensors to CPU before using them as
        arguments or return values of ``func``.

    .. warning ::
        The ``remote`` API does not copy storages of argument tensors until
        sending them over the wire, which could be done by a different thread
        depending on the RPC backend type. The caller should make sure that the
        contents of those tensors stay intact until the returned RRef is
        confirmed by the owner, which can be checked using the
        :meth:`torch.distributed.rpc.RRef.confirmed_by_owner` API.

    Example::
        Make sure that ``MASTER_ADDRESS`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        >>> export MASTER_ADDRESS=localhost
        >>> export MASTER_port=5678

        Then run the following code in two different processes:

        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> rref1 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 3))
        >>> rref2 = rpc.remote("worker1", torch.add, args=(torch.ones(2), 1))
        >>> x = rref1.to_here() + rref2.to_here()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(t1, t2):
        >>>    return torch.add(t1, t2)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> rref = rpc.remote("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> rref.to_here()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rpc.api.remote', 'remote(to, func, args=None, kwargs=None)', {'torch': torch, '_to_worker_info': _to_worker_info, '_start_record_function': _start_record_function, 'RPCExecMode': RPCExecMode, 'get_worker_info': get_worker_info, '_invoke_remote_builtin': _invoke_remote_builtin, '_invoke_remote_torchscript': _invoke_remote_torchscript, '_default_pickler': _default_pickler, 'PythonUDF': PythonUDF, '_invoke_remote_python_udf': _invoke_remote_python_udf, '_require_initialized': _require_initialized, 'to': to, 'func': func, 'args': args, 'kwargs': kwargs}, 1)

def _invoke_rpc(to, func, rpc_type, args=None, kwargs=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rpc.api._invoke_rpc', '_invoke_rpc(to, func, rpc_type, args=None, kwargs=None)', {'torch': torch, '_to_worker_info': _to_worker_info, '_start_record_function': _start_record_function, 'get_worker_info': get_worker_info, '_invoke_rpc_builtin': _invoke_rpc_builtin, '_invoke_rpc_torchscript': _invoke_rpc_torchscript, '_default_pickler': _default_pickler, 'PythonUDF': PythonUDF, '_invoke_rpc_python_udf': _invoke_rpc_python_udf, 'to': to, 'func': func, 'rpc_type': rpc_type, 'args': args, 'kwargs': kwargs}, 1)

@_require_initialized
def rpc_sync(to, func, args=None, kwargs=None):
    """
    Make a blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe.

    Arguments:
        to (str or WorkerInfo): id or name of the destination worker.
        func (callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        Returns the result of running ``func`` with ``args`` and ``kwargs``.

    .. warning ::
        Using GPU tensors as arguments or return values of ``func`` is not
        supported since we don't support sending GPU tensors over the wire. You
        need to explicitly copy GPU tensors to CPU before using them as
        arguments or return values of ``func``.

    Example::
        Make sure that ``MASTER_ADDRESS`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        >>> export MASTER_ADDRESS=localhost
        >>> export MASTER_port=5678

        Then run the following code in two different processes:

        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> ret = rpc.rpc_sync("worker1", torch.add, args=(torch.ones(2), 3))
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(t1, t2):
        >>>    return torch.add(t1, t2)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> ret = rpc.rpc_sync("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

    """
    fut = _invoke_rpc(to, func, RPCExecMode.SYNC, args, kwargs)
    return fut.wait()

@_require_initialized
def rpc_async(to, func, args=None, kwargs=None):
    """
    Make a non-blocking RPC call to run function ``func`` on worker ``to``. RPC
    messages are sent and received in parallel to execution of Python code. This
    method is thread-safe. This method will immediately return a Future that can
    be awaited on.

    Arguments:
        to (str or WorkerInfo): id or name of the destination worker.
        func (callable): a callable function, such as Python callables, builtin
                         operators (e.g. :meth:`~torch.add`) and annotated
                         TorchScript functions.
        args (tuple): the argument tuple for the ``func`` invocation.
        kwargs (dict): is a dictionary of keyword arguments for the ``func``
                       invocation.

    Returns:
        Returns a Future object that can be waited
        on. When completed, the return value of ``func`` on ``args`` and
        ``kwargs`` can be retrieved from the Future object.

    .. warning ::
        Using GPU tensors as arguments or return values of ``func`` is not
        supported since we don't support sending GPU tensors over the wire. You
        need to explicitly copy GPU tensors to CPU before using them as
        arguments or return values of ``func``.

    .. warning ::
        The ``rpc_async`` API does not copy storages of argument tensors until
        sending them over the wire, which could be done by a different thread
        depending on the RPC backend type. The caller should make sure that the
        contents of those tensors stay intact until the returned Future
        completes.

    Example::
        Make sure that ``MASTER_ADDRESS`` and ``MASTER_PORT`` are set properly
        on both workers. Refer to :meth:`~torch.distributed.init_process_group`
        API for more details. For example,

        >>> export MASTER_ADDRESS=localhost
        >>> export MASTER_port=5678

        Then run the following code in two different processes:

        >>> # On worker 0:
        >>> import torch
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> fut1 = rpc.rpc_async("worker1", torch.add, args=(torch.ones(2), 3))
        >>> fut2 = rpc.rpc_async("worker1", min, args=(1, 2))
        >>> result = fut1.wait() + fut2.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()

        Below is an example of running a TorchScript function using RPC.

        >>> # On both workers:
        >>> @torch.jit.script
        >>> def my_script_add(t1, t2):
        >>>    return torch.add(t1, t2)

        >>> # On worker 0:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker0", rank=0, world_size=2)
        >>> fut = rpc.rpc_async("worker1", my_script_add, args=(torch.ones(2), 3))
        >>> ret = fut.wait()
        >>> rpc.shutdown()

        >>> # On worker 1:
        >>> import torch.distributed.rpc as rpc
        >>> rpc.init_rpc("worker1", rank=1, world_size=2)
        >>> rpc.shutdown()
    """
    return _invoke_rpc(to, func, RPCExecMode.ASYNC, args, kwargs)

