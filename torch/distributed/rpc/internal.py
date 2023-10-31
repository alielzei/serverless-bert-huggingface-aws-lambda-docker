import collections
import copyreg
from enum import Enum
import io
import pickle
import threading
import traceback
import torch
import torch.distributed as dist
_thread_local_tensor_tables = threading.local()


class RPCExecMode(Enum):
    SYNC = 'sync'
    ASYNC = 'async'
    REMOTE = 'remote'



class _InternalRPCPickler:
    """
    This class provides serialize() and deserialize() interfaces to serialize
    data to be "binary string + tensor table" format
    So for RPC python UDF function and args, non tensor data will be serialized
    into regular binary string, tensor data will be put into thread local tensor
    tables, this serialization format is consistent with builtin operator and args
    using JIT pickler. This format will make tensor handling in C++ much easier,
    e.g. attach tensor to distributed autograd graph in C++
    """
    
    def __init__(self):
        if torch._six.PY3:
            self._dispatch_table = copyreg.dispatch_table.copy()
            self._dispatch_table[torch.Tensor] = self._tensor_reducer
    
    @classmethod
    def _tensor_receiver(cls, tensor_index):
        global _thread_local_tensor_tables
        return _thread_local_tensor_tables.recv_tables[tensor_index]
    
    def _tensor_reducer(self, tensor):
        global _thread_local_tensor_tables
        _thread_local_tensor_tables.send_tables.append(tensor)
        tensor_index = len(_thread_local_tensor_tables.send_tables) - 1
        return (_InternalRPCPickler._tensor_receiver, (tensor_index, ))
    
    @classmethod
    def _rref_receiver(cls, rref_fork_data):
        return dist.rpc.RRef._deserialize(rref_fork_data)
    
    def _rref_reducer(self, rref):
        rref_fork_data = rref._serialize()
        return (_InternalRPCPickler._rref_receiver, (rref_fork_data, ))
    
    def serialize(self, obj):
        """
        Serialize non tensor data into binary string, tensor data into
        tensor table
        """
        f = io.BytesIO()
        p = pickle.Pickler(f)
        p.dispatch_table = self._dispatch_table
        p.dispatch_table[dist.rpc.RRef] = self._rref_reducer
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, 'send_tables'):
            old_send_tables = _thread_local_tensor_tables.send_tables
        else:
            old_send_tables = None
        _thread_local_tensor_tables.send_tables = []
        p.dump(obj)
        tensors = _thread_local_tensor_tables.send_tables
        if old_send_tables is not None:
            _thread_local_tensor_tables.send_tables = old_send_tables
        else:
            del _thread_local_tensor_tables.send_tables
        return (f.getvalue(), tensors)
    
    def deserialize(self, binary_data, tensor_table):
        """
        Deserilize binary string + tensor table to original obj
        """
        global _thread_local_tensor_tables
        if hasattr(_thread_local_tensor_tables, 'recv_tables'):
            old_recv_tables = _thread_local_tensor_tables.recv_tables
        else:
            old_recv_tables = None
        _thread_local_tensor_tables.recv_tables = tensor_table
        try:
            ret = pickle.loads(binary_data)
        except AttributeError as e:
            except_str = str(e) + ' Default RPC pickler does not serialize\n            function code. Ensure that UDFs are defined on both caller and\n            callee modules.'
            ret = AttributeError(except_str)
        if old_recv_tables is not None:
            _thread_local_tensor_tables.recv_tables = old_recv_tables
        else:
            del _thread_local_tensor_tables.recv_tables
        return ret

_internal_rpc_pickler = _InternalRPCPickler()

def serialize(obj):
    return _internal_rpc_pickler.serialize(obj)

def deserialize(binary_data, tensor_table):
    return _internal_rpc_pickler.deserialize(binary_data, tensor_table)

def _run_function(python_udf):
    """
    This function is exclusively called from C++.
    See ``torch/csrc/distributed/rpc/python_rpc_handler.cpp``.

    Runs a Python UDF and returns its return value.
    Wraps any exception in ``RemoteException`` if the function raises.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rpc.internal._run_function', '_run_function(python_udf)', {'traceback': traceback, 'RemoteException': RemoteException, 'python_udf': python_udf}, 1)

def _handle_exception(result):
    if isinstance(result, RemoteException):
        raise result.exception_type(result.msg)

def _start_record_function(exec_type, func_name, current_worker_name, dest_worker_name):
    """
    This function should be called from RPC/RRef functions to create a
    RecordFunction object for profiling. This function also runs the before
    callbacks that start the profiling, though the user is responsible for
    running the appropriate callbacks when the function to be profiled finishes.

    Arguments:
        exec_type (RPCExecMode): Type of RPC/RRef call
        func_name (str): Name of function being profiled.
        current_worker_name (str): Name of current worker.
        dest_worker_name (str): Name of the destination worker.

    Returns:
        An instance of `torch.autograd._RecordFunction`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rpc.internal._start_record_function', '_start_record_function(exec_type, func_name, current_worker_name, dest_worker_name)', {'torch': torch, 'exec_type': exec_type, 'func_name': func_name, 'current_worker_name': current_worker_name, 'dest_worker_name': dest_worker_name}, 1)
PythonUDF = collections.namedtuple('PythonUDF', ['func', 'args', 'kwargs'])
RemoteException = collections.namedtuple('RemoteException', ['msg', 'exception_type'])

