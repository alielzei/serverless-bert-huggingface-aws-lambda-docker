from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import datetime
import enum
import torch.distributed as dist
import torch.distributed.distributed_c10d as dc10d
from . import constants as rpc_constants
BackendValue = collections.namedtuple('BackendValue', ['construct_rpc_backend_options_handler', 'init_backend_handler'])

def _backend_type_repr(self):
    return 'BackendType.' + self.name
BackendType = enum.Enum(value='BackendType', names={})
BackendType.__repr__ = _backend_type_repr

def register_backend(backend_name, construct_rpc_backend_options_handler, init_backend_handler):
    """Registers a new RPC backend.

    Arguments:
        backend_name (str): backend string to identify the handler.
        construct_rpc_backend_options_handler (function):
            Handler that is invoked when
            rpc_backend.construct_rpc_backend_options(**dict) is called.
        init_backend_handler (function): Handler that is invoked when the
            `_init_rpc_backend()` function is called with a backend.
             This returns the agent.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rpc.backend_registry.register_backend', 'register_backend(backend_name, construct_rpc_backend_options_handler, init_backend_handler)', {'BackendValue': BackendValue, 'enum': enum, '_backend_type_repr': _backend_type_repr, 'backend_name': backend_name, 'construct_rpc_backend_options_handler': construct_rpc_backend_options_handler, 'init_backend_handler': init_backend_handler}, 1)

def construct_rpc_backend_options(backend, rpc_timeout=rpc_constants.DEFAULT_RPC_TIMEOUT, init_method=rpc_constants.DEFAULT_INIT_METHOD, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rpc.backend_registry.construct_rpc_backend_options', 'construct_rpc_backend_options(backend, rpc_timeout=rpc_constants.DEFAULT_RPC_TIMEOUT, init_method=rpc_constants.DEFAULT_INIT_METHOD, **kwargs)', {'datetime': datetime, 'backend': backend, 'rpc_timeout': rpc_timeout, 'init_method': init_method, 'kwargs': kwargs}, 1)

def init_backend(backend, *args, **kwargs):
    return backend.value.init_backend_handler(*args, **kwargs)

def _process_group_construct_rpc_backend_options_handler(rpc_timeout, init_method, num_send_recv_threads=rpc_constants.DEFAULT_NUM_SEND_RECV_THREADS, **kwargs):
    from . import ProcessGroupRpcBackendOptions
    return ProcessGroupRpcBackendOptions(rpc_timeout=rpc_timeout, init_method=init_method, num_send_recv_threads=num_send_recv_threads)

def _process_group_init_backend_handler(store, name, rank, world_size, rpc_backend_options):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rpc.backend_registry._process_group_init_backend_handler', '_process_group_init_backend_handler(store, name, rank, world_size, rpc_backend_options)', {'dist': dist, 'rpc_constants': rpc_constants, 'dc10d': dc10d, 'store': store, 'name': name, 'rank': rank, 'world_size': world_size, 'rpc_backend_options': rpc_backend_options}, 1)
register_backend('PROCESS_GROUP', _process_group_construct_rpc_backend_options_handler, _process_group_init_backend_handler)

