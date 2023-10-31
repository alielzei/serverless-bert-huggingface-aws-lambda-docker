from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
from google.protobuf.message import Message
from multiprocessing import Process
import os
from collections import defaultdict
import logging
import numpy as np
from past.builtins import basestring
import shutil
import socket
import tempfile
from caffe2.proto import caffe2_pb2
from caffe2.python import scope, utils
import caffe2.python._import_c_extension as C
logger = logging.getLogger(__name__)
Blobs = C.blobs
ResetBlob = C.reset_blob
CreateBlob = C.create_blob
CurrentWorkspace = C.current_workspace
DeserializeBlob = C.deserialize_blob
GlobalInit = C.global_init
HasBlob = C.has_blob
RegisteredOperators = C.registered_operators
SerializeBlob = C.serialize_blob
SwitchWorkspace = C.switch_workspace
RootFolder = C.root_folder
Workspaces = C.workspaces
BenchmarkNet = C.benchmark_net
BenchmarkNetOnce = C.benchmark_net_once
GetStats = C.get_stats
operator_tracebacks = defaultdict(dict)
is_asan = C.is_asan
has_cuda_support = C.has_cuda_support
has_hip_support = C.has_hip_support
has_gpu_support = C.has_gpu_support
if has_cuda_support:
    GpuDeviceType = caffe2_pb2.CUDA
    NumCudaDevices = C.num_cuda_devices
    NumGpuDevices = C.num_cuda_devices
    GetCUDAVersion = C.get_cuda_version
    GetCuDNNVersion = C.get_cudnn_version
    
    def GetGpuPeerAccessPattern():
        return np.asarray(C.get_cuda_peer_access_pattern())
    GetDeviceProperties = C.get_device_properties
    GetGPUMemoryInfo = C.get_gpu_memory_info
else:
    NumCudaDevices = lambda: 0
    GetCUDAVersion = lambda: 0
    GetCuDNNVersion = lambda: 0
if has_hip_support:
    GpuDeviceType = caffe2_pb2.HIP
    NumGpuDevices = C.num_hip_devices
    
    def GetGpuPeerAccessPattern():
        return np.asarray(C.get_hip_peer_access_pattern())
    GetDeviceProperties = C.get_device_properties
    GetGPUMemoryInfo = C.get_gpu_memory_info
if not has_gpu_support:
    GpuDeviceType = caffe2_pb2.CUDA
    NumGpuDevices = lambda: 0
    GetDeviceProperties = lambda x: None
    GetGpuPeerAccessPattern = lambda: np.array([])
    GetGPUMemoryInfo = lambda: None
IsNUMAEnabled = C.is_numa_enabled
GetNumNUMANodes = C.get_num_numa_nodes
GetBlobNUMANode = C.get_blob_numa_node
GetBlobSizeBytes = C.get_blob_size_bytes

def FillRandomNetworkInputs(net, input_dims, input_types):
    C.fill_random_network_inputs(net.Proto().SerializeToString(), input_dims, input_types)

def _GetFreeFlaskPort():
    """Get a free flask port."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace._GetFreeFlaskPort', '_GetFreeFlaskPort()', {'socket': socket}, 1)

def StartMint(root_folder=None, port=None):
    """Start a mint instance.

    TODO(Yangqing): this does not work well under ipython yet. According to
        https://github.com/ipython/ipython/issues/5862
    writing up some fix is a todo item.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.StartMint', 'StartMint(root_folder=None, port=None)', {'C': C, '_GetFreeFlaskPort': _GetFreeFlaskPort, 'Process': Process, 'socket': socket, 'root_folder': root_folder, 'port': port}, 1)

def StringifyProto(obj):
    """Stringify a protocol buffer object.

  Inputs:
    obj: a protocol buffer object, or a Pycaffe2 object that has a Proto()
        function.
  Outputs:
    string: the output protobuf string.
  Raises:
    AttributeError: if the passed in object does not have the right attribute.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.StringifyProto', 'StringifyProto(obj)', {'basestring': basestring, 'Message': Message, 'obj': obj}, 1)

def ResetWorkspace(root_folder=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.ResetWorkspace', 'ResetWorkspace(root_folder=None)', {'C': C, 'os': os, 'root_folder': root_folder}, 1)

def CreateNet(net, overwrite=False, input_blobs=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.CreateNet', 'CreateNet(net, overwrite=False, input_blobs=None)', {'C': C, 'CallWithExceptionIntercept': CallWithExceptionIntercept, 'GetNetName': GetNetName, 'StringifyProto': StringifyProto, 'net': net, 'overwrite': overwrite, 'input_blobs': input_blobs}, 1)

def Predictor(init_net, predict_net):
    return C.Predictor(StringifyProto(init_net), StringifyProto(predict_net))

def GetOperatorCost(operator, blobs):
    return C.get_operator_cost(StringifyProto(operator), blobs)

def RunOperatorOnce(operator):
    return C.run_operator_once(StringifyProto(operator))

def RunOperatorMultiple(operator, num_runs):
    return C.run_operator_multiple(StringifyProto(operator), num_runs)

def RunOperatorsOnce(operators):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.RunOperatorsOnce', 'RunOperatorsOnce(operators)', {'RunOperatorOnce': RunOperatorOnce, 'operators': operators}, 1)

def ClearGlobalNetObserver():
    return C.clear_global_net_observer()

def CallWithExceptionIntercept(func, op_id_fetcher, net_name, *args, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.CallWithExceptionIntercept', 'CallWithExceptionIntercept(func, op_id_fetcher, net_name, *args, **kwargs)', {'operator_tracebacks': operator_tracebacks, 'logger': logger, 'func': func, 'op_id_fetcher': op_id_fetcher, 'net_name': net_name, 'args': args, 'kwargs': kwargs}, 1)

def RunNetOnce(net):
    return CallWithExceptionIntercept(C.run_net_once, C.Workspace.current._last_failed_op_net_position, GetNetName(net), StringifyProto(net))

def RunNet(name, num_iter=1, allow_fail=False):
    """Runs a given net.

    Inputs:
      name: the name of the net, or a reference to the net.
      num_iter: number of iterations to run
      allow_fail: if True, does not assert on net exec failure but returns False
    Returns:
      True or an exception.
    """
    return CallWithExceptionIntercept(C.run_net, C.Workspace.current._last_failed_op_net_position, GetNetName(name), StringifyNetName(name), num_iter, allow_fail)

def RunPlan(plan_or_step):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.RunPlan', 'RunPlan(plan_or_step)', {'C': C, 'StringifyProto': StringifyProto, 'plan_or_step': plan_or_step}, 1)

def RunPlanInBackground(plan_or_step):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.RunPlanInBackground', 'RunPlanInBackground(plan_or_step)', {'C': C, 'StringifyProto': StringifyProto, 'plan_or_step': plan_or_step}, 1)

def InferShapesAndTypes(nets, blob_dimensions=None, nets_proto=False, blob_types=None):
    """Infers the shapes and types for the specified nets.

    Inputs:
      nets: the list of nets
      blob_dimensions (optional): a dictionary of blobs and their dimensions.
          If not specified, the workspace blobs are used.
      nets_proto (optional): a boolean flag indicating whether the protobuffer
          representation is passed to the routine.
    Returns:
      A tuple of (shapes, types) dictionaries keyed by blob name.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.InferShapesAndTypes', 'InferShapesAndTypes(nets, blob_dimensions=None, nets_proto=False, blob_types=None)', {'StringifyProto': StringifyProto, 'C': C, 'caffe2_pb2': caffe2_pb2, 'nets': nets, 'blob_dimensions': blob_dimensions, 'nets_proto': nets_proto, 'blob_types': blob_types}, 2)

def _StringifyName(name, expected_type):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace._StringifyName', '_StringifyName(name, expected_type)', {'basestring': basestring, 'name': name, 'expected_type': expected_type}, 1)

def StringifyBlobName(name):
    return _StringifyName(name, 'BlobReference')

def StringifyNetName(name):
    return _StringifyName(name, 'Net')

def GetNetName(net):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.GetNetName', 'GetNetName(net)', {'basestring': basestring, 'caffe2_pb2': caffe2_pb2, 'net': net}, 1)

def FeedBlob(name, arr, device_option=None):
    """Feeds a blob into the workspace.

    Inputs:
      name: the name of the blob.
      arr: either a TensorProto object or a numpy array object to be fed into
          the workspace.
      device_option (optional): the device option to feed the data with.
    Returns:
      True or False, stating whether the feed is successful.
    """
    ws = C.Workspace.current
    return _Workspace_feed_blob(ws, name, arr, device_option)

def FetchBlobs(names):
    """Fetches a list of blobs from the workspace.

    Inputs:
        names: list of names of blobs - strings or BlobReferences
    Returns:
        list of fetched blobs
    """
    return [FetchBlob(name) for name in names]

def FetchBlob(name):
    """Fetches a blob from the workspace.

    Inputs:
      name: the name of the blob - a string or a BlobReference
    Returns:
      Fetched blob (numpy array or string) if successful
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.FetchBlob', 'FetchBlob(name)', {'C': C, 'StringifyBlobName': StringifyBlobName, 'name': name}, 1)

def FetchTorch(name):
    ws = C.Workspace.current
    return ws.blobs[name].to_torch()
Int8Tensor = collections.namedtuple('Int8Tensor', ['data', 'scale', 'zero_point'])

def FetchInt8Blob(name):
    """Fetches an Int8 blob from the workspace. It shared backend implementation
    with FetchBlob but it is recommended when fetching Int8 Blobs

    Inputs:
      name: the name of the Int8 blob - a string or a BlobReference
    Returns:
      data: int8 numpy array, data
      scale: float, fake quantization scale
      zero_point: int, fake quantization offset
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.FetchInt8Blob', 'FetchInt8Blob(name)', {'C': C, 'StringifyBlobName': StringifyBlobName, 'Int8Tensor': Int8Tensor, 'name': name}, 1)

def FetchInt8BlobRealVal(name):
    """Fetches an Int8 blob from the workspace and return its real value representation.

    Inputs:
      name: the name of the Int8 blob - a string or a BlobReference
    Returns:
      real value representation of int8 numpy array
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.FetchInt8BlobRealVal', 'FetchInt8BlobRealVal(name)', {'C': C, 'StringifyBlobName': StringifyBlobName, 'Int8Tensor': Int8Tensor, 'np': np, 'name': name}, 1)

def _Workspace_fetch_int8_blob(ws, name):
    """Fetches an Int8 blob from the workspace. It shared backend implementation
    with FetchBlob but it is recommended when fetching Int8 Blobs

    Inputs:
      name: the name of the Int8 blob - a string or a BlobReference
    Returns:
      data: int8 numpy array, data
      scale: float, fake quantization scale
      zero_point: int, fake quantization offset
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace._Workspace_fetch_int8_blob', '_Workspace_fetch_int8_blob(ws, name)', {'StringifyBlobName': StringifyBlobName, 'Int8Tensor': Int8Tensor, 'ws': ws, 'name': name}, 1)
C.Workspace.fetch_int8_blob = _Workspace_fetch_int8_blob

def ApplyTransform(transform_key, net):
    """Apply a Transform to a NetDef protobuf object, and returns the new
    transformed NetDef.

    Inputs:
      transform_key: the name of the transform, as it is stored in the registry
      net: a NetDef protobuf object
    Returns:
      Transformed NetDef protobuf object.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.ApplyTransform', 'ApplyTransform(transform_key, net)', {'caffe2_pb2': caffe2_pb2, 'C': C, 'transform_key': transform_key, 'net': net}, 1)

def ApplyTransformIfFaster(transform_key, net, init_net, **kwargs):
    """Apply a Transform to a NetDef protobuf object, and returns the new
    transformed NetDef, only if it runs faster than the original.

    The runs are performed on the current active workspace (gWorkspace).
    You should initialize that workspace before making a call to this function.

    Inputs:
      transform_key: the name of the transform, as it is stored in the registry
      net: a NetDef protobuf object
      init_net: The net to initialize the workspace.
      warmup_runs (optional):
        Determines how many times the net is run before testing.
        Will be 5 by default.
      main_runs (optional):
        Determines how many times the net is run during testing.
        Will be 10 by default.
      improvement_threshold (optional):
        Determines the factor which the new net needs to be faster
        in order to replace the old. Will be 1.01 by default.

    Returns:
      Either a Transformed NetDef protobuf object, or the original netdef.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.ApplyTransformIfFaster', 'ApplyTransformIfFaster(transform_key, net, init_net, **kwargs)', {'caffe2_pb2': caffe2_pb2, 'C': C, 'transform_key': transform_key, 'net': net, 'init_net': init_net, 'kwargs': kwargs}, 1)

def GetNameScope():
    """Return the current namescope string. To be used to fetch blobs"""
    return scope.CurrentNameScope()


class _BlobDict(object):
    """Provides python dict compatible way to do fetching and feeding"""
    
    def __getitem__(self, key):
        return FetchBlob(key)
    
    def __setitem__(self, key, value):
        return FeedBlob(key, value)
    
    def __len__(self):
        return len(C.blobs())
    
    def __iter__(self):
        return C.blobs().__iter__()
    
    def __contains__(self, item):
        return C.has_blob(item)

blobs = _BlobDict()
_immediate_mode = False
_immediate_workspace_name = '_CAFFE2_IMMEDIATE'
_immediate_root_folder = ''

def IsImmediate():
    return _immediate_mode

@contextlib.contextmanager
def WorkspaceGuard(workspace_name):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.workspace.WorkspaceGuard', 'WorkspaceGuard(workspace_name)', {'CurrentWorkspace': CurrentWorkspace, 'SwitchWorkspace': SwitchWorkspace, 'contextlib': contextlib, 'workspace_name': workspace_name}, 0)

def StartImmediate(i_know=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.StartImmediate', 'StartImmediate(i_know=False)', {'IsImmediate': IsImmediate, 'StopImmediate': StopImmediate, 'WorkspaceGuard': WorkspaceGuard, '_immediate_workspace_name': _immediate_workspace_name, 'tempfile': tempfile, 'ResetWorkspace': ResetWorkspace, 'i_know': i_know}, 1)

def StopImmediate():
    """Stops an immediate mode run."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace.StopImmediate', 'StopImmediate()', {'IsImmediate': IsImmediate, 'WorkspaceGuard': WorkspaceGuard, '_immediate_workspace_name': _immediate_workspace_name, 'ResetWorkspace': ResetWorkspace, 'shutil': shutil}, 1)

def ImmediateBlobs():
    with WorkspaceGuard(_immediate_workspace_name):
        return Blobs()

def RunOperatorImmediate(op):
    with WorkspaceGuard(_immediate_workspace_name):
        RunOperatorOnce(op)

def FetchImmediate(*args, **kwargs):
    with WorkspaceGuard(_immediate_workspace_name):
        return FetchBlob(*args, **kwargs)

def FeedImmediate(*args, **kwargs):
    with WorkspaceGuard(_immediate_workspace_name):
        return FeedBlob(*args, **kwargs)

def _Workspace_create_net_with_exception_intercept(ws, net, overwrite=False):
    return CallWithExceptionIntercept(ws._create_net, ws._last_failed_op_net_position, GetNetName(net), StringifyProto(net), overwrite)

def _Workspace_run(ws, obj):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace._Workspace_run', '_Workspace_run(ws, obj)', {'caffe2_pb2': caffe2_pb2, 'CallWithExceptionIntercept': CallWithExceptionIntercept, 'GetNetName': GetNetName, 'ws': ws, 'obj': obj}, 1)

def _Workspace_feed_blob(ws, name, arr, device_option=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace._Workspace_feed_blob', '_Workspace_feed_blob(ws, name, arr, device_option=None)', {'caffe2_pb2': caffe2_pb2, 'utils': utils, 'np': np, 'scope': scope, 'logger': logger, 'StringifyBlobName': StringifyBlobName, 'ws': ws, 'name': name, 'arr': arr, 'device_option': device_option}, 1)

def _Workspace_remove_blob(ws, blob):
    ws._remove_blob(str(blob))
Workspace = C.Workspace
Workspace.create_net = _Workspace_create_net_with_exception_intercept
Workspace.run = _Workspace_run
Workspace.feed_blob = _Workspace_feed_blob
Workspace.remove_blob = _Workspace_remove_blob

def _Blob_feed(blob, arg, device_option=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace._Blob_feed', '_Blob_feed(blob, arg, device_option=None)', {'StringifyProto': StringifyProto, 'blob': blob, 'arg': arg, 'device_option': device_option}, 1)
C.Blob.feed = _Blob_feed

def _Tensor_to_torch(tensor):
    """
    PyTorch tensor interop (TensorCPU methods)

    Can be accessed as:
      workspace.Workspace.current.blobs['foo'].tensor().to_torch()
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace._Tensor_to_torch', '_Tensor_to_torch(tensor)', {'tensor': tensor}, 1)
C.TensorCPU.to_torch = _Tensor_to_torch

def _Blob_to_torch(blob):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.workspace._Blob_to_torch', '_Blob_to_torch(blob)', {'blob': blob}, 1)
C.Blob.to_torch = _Blob_to_torch

