from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from collections import OrderedDict
from future.utils import viewitems, viewkeys, viewvalues
import logging
import copy
from multiprocessing import cpu_count
from caffe2.python import model_helper, dyndep, scope, workspace, core, memonger, utils
from caffe2.proto import caffe2_pb2
import numpy as np
import warnings
dyndep.InitOpsLibrary('@/caffe2/caffe2/contrib/gloo:gloo_ops')
if workspace.NumGpuDevices() > 0:
    dyndep.InitOpsLibrary('@/caffe2/caffe2/contrib/nccl:nccl_ops')
    dyndep.InitOpsLibrary('@/caffe2/caffe2/contrib/gloo:gloo_ops_gpu')
log = logging.getLogger('data_parallel_model')
log.setLevel(logging.INFO)
_DEFAULT_TIMEOUT_SEC = 30
_DEFAULT_BARRIER_NET_TIMEOUT_SEC = 300

def Parallelize_GPU(*args, **kwargs):
    kwargs['cpu_device'] = False
    Parallelize(*args, **kwargs)

def Parallelize_CPU(*args, **kwargs):
    kwargs['cpu_device'] = True
    Parallelize(*args, **kwargs)

def Parallelize_iDeep(*args, **kwargs):
    kwargs['ideep'] = True
    Parallelize(*args, **kwargs)

def Parallelize(model_helper_obj, input_builder_fun, forward_pass_builder_fun, param_update_builder_fun=None, optimizer_builder_fun=None, post_sync_builder_fun=None, pre_grad_net_transformer_fun=None, net_transformer_fun=None, devices=None, rendezvous=None, net_type='dag', broadcast_computed_params=True, optimize_gradient_memory=False, dynamic_memory_management=False, blobs_to_keep=None, use_nccl=False, max_concurrent_distributed_ops=16, cpu_device=False, ideep=False, num_threads_per_device=4, shared_model=False, combine_spatial_bn=False, barrier_net_timeout_sec=_DEFAULT_BARRIER_NET_TIMEOUT_SEC):
    """
    Function to create a model that can run on many GPUs or CPUs.
      model_helper_obj: an object of ModelHelper
      input_builder_fun:
                         Function that adds the input operators
                         Note: Remember to instantiate reader outside of this
                         function so all devices share same reader object.
                         Signature:  input_builder_fun(model)
      forward_pass_builder_fun:
                        Function to add the operators to the model.
                        Must return list of loss-blob references that
                        are used to build the gradient. Loss scale parameter
                        is passed, as you should scale the loss of your model
                        by 1.0 / the total number of devices.
                        Signature: forward_pass_builder_fun(model, loss_scale)
      param_update_builder_fun:
                        Function that adds operators that are run after
                        gradient update, such as updating the weights and
                        weight decaying. This is called for each GPU separately.
                        Signature: param_update_builder_fun(model)
      optimizer_builder_fun:
                        Alternative to param_update_builder_fun, allows one
                        to add an optimizer for the whole model. Called only
                        once, without name or devicescope.
      net_transformer_fun:
                        Optional function to transform the network after the
                        network is built. It will be called once (NOT once per
                        GPU.)
                        Signature:
                        net_transformer_fun(
                            model, num_devices, device_prefix, device_type)
      pre_grad_net_transformer_fun:
                        Optional function to transform the network similar to
                        net_transformer_fun, but happens before gradient ops
                        been add.
                        Signature: pre_grad_net_transformer_fun(model)
      post_sync_builder_fun:
                        Function applied after initial parameter sync has been
                        completed, such as keeping multi-precision parameters
                        in sync.
                        Signature: post_sync_builder_fun(model)
      devices:          List of GPU ids, such as [0, 1, 2, 3],
      rendezvous:       used for rendezvous in distributed computation, if None
                        then only one node is used. To create rendezvous,
                        use <TBD>.
      net_type:         Network type
      optimize_gradient_memory: whether to apply 'memonger' to share blobs
      shared_model      (only for CPU) use same parameters on each device
                        in gradient computation to reduce memory footprint.
      dynamic_memory_management: Whether to apply dynamic memory optimization
                        by freeing unused blobs. The underlying (de)allocation
                        uses cached allocator. For GPU training PLEASE MAKE SURE
                        caffe2_cuda_memory_pool is set.
      blobs_to_keep :   A list of blob names to keep and don't free during
                        dynamic memory optimization (for example loss blob).
      cpu_device        Use CPU instead of GPU.
      ideep             Use ideep.
      combine_spatial_bn:
                        When set to True, applies batch normalization across
                        all devices within the node. If False, batch
                        normalization will be done separately for each device.
                        This option is currently only supported on the CPU.
      barrier_net_timeout_sec:
                        The timeout in seconds of the barrier net, which is run
                        to synchronize shards before a training epoch starts.
                        Defaults to 300 seconds.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.Parallelize', "Parallelize(model_helper_obj, input_builder_fun, forward_pass_builder_fun, param_update_builder_fun=None, optimizer_builder_fun=None, post_sync_builder_fun=None, pre_grad_net_transformer_fun=None, net_transformer_fun=None, devices=None, rendezvous=None, net_type='dag', broadcast_computed_params=True, optimize_gradient_memory=False, dynamic_memory_management=False, blobs_to_keep=None, use_nccl=False, max_concurrent_distributed_ops=16, cpu_device=False, ideep=False, num_threads_per_device=4, shared_model=False, combine_spatial_bn=False, barrier_net_timeout_sec=_DEFAULT_BARRIER_NET_TIMEOUT_SEC)", {'scope': scope, 'caffe2_pb2': caffe2_pb2, 'workspace': workspace, 'cpu_count': cpu_count, 'log': log, 'model_helper': model_helper, 'copy': copy, 'core': core, '_ValidateParams': _ValidateParams, '_GroupByDevice': _GroupByDevice, 'viewkeys': viewkeys, '_AddGradientOperators': _AddGradientOperators, '_InferBlobDevice': _InferBlobDevice, '_InterleaveOps': _InterleaveOps, '_CPUInterDeviceBatchNormalization': _CPUInterDeviceBatchNormalization, '_GPUInterDeviceBatchNormalization': _GPUInterDeviceBatchNormalization, '_BroadcastComputedParams': _BroadcastComputedParams, '_GetReverseOrderedGrads': _GetReverseOrderedGrads, '_AllReduceBlobs': _AllReduceBlobs, '_PruneParametersForSharing': _PruneParametersForSharing, '_ComputeBlobsToSync': _ComputeBlobsToSync, '_AnalyzeOperators': _AnalyzeOperators, '_SyncAllParams': _SyncAllParams, '_OptimizeGradientMemorySimple': _OptimizeGradientMemorySimple, '_AddDynamicMemoryOptimization': _AddDynamicMemoryOptimization, '_AddBarrierToModelNets': _AddBarrierToModelNets, '_RemapParameterBlobsForSharedModel': _RemapParameterBlobsForSharedModel, 'model_helper_obj': model_helper_obj, 'input_builder_fun': input_builder_fun, 'forward_pass_builder_fun': forward_pass_builder_fun, 'param_update_builder_fun': param_update_builder_fun, 'optimizer_builder_fun': optimizer_builder_fun, 'post_sync_builder_fun': post_sync_builder_fun, 'pre_grad_net_transformer_fun': pre_grad_net_transformer_fun, 'net_transformer_fun': net_transformer_fun, 'devices': devices, 'rendezvous': rendezvous, 'net_type': net_type, 'broadcast_computed_params': broadcast_computed_params, 'optimize_gradient_memory': optimize_gradient_memory, 'dynamic_memory_management': dynamic_memory_management, 'blobs_to_keep': blobs_to_keep, 'use_nccl': use_nccl, 'max_concurrent_distributed_ops': max_concurrent_distributed_ops, 'cpu_device': cpu_device, 'ideep': ideep, 'num_threads_per_device': num_threads_per_device, 'shared_model': shared_model, 'combine_spatial_bn': combine_spatial_bn, 'barrier_net_timeout_sec': barrier_net_timeout_sec, '_DEFAULT_BARRIER_NET_TIMEOUT_SEC': _DEFAULT_BARRIER_NET_TIMEOUT_SEC}, 1)

def Parallelize_GPU_BMUF(*args, **kwargs):
    kwargs['cpu_device'] = False
    Parallelize_BMUF(*args, **kwargs)

def Parallelize_CPU_BMUF(*args, **kwargs):
    kwargs['cpu_device'] = True
    Parallelize_BMUF(*args, **kwargs)

def Parallelize_BMUF(model_helper_obj, input_builder_fun, forward_pass_builder_fun, param_update_builder_fun, block_learning_rate=1.0, block_momentum=None, devices=None, rendezvous=None, net_type='dag', master_device=None, use_nccl=False, nesterov=False, optimize_gradient_memory=False, reset_momentum_sgd=False, warmup_iterations=None, max_concurrent_distributed_ops=4, add_blobs_to_sync=None, num_threads_per_device=4, cpu_device=False, barrier_net_timeout_sec=_DEFAULT_BARRIER_NET_TIMEOUT_SEC):
    """
    Function to create model that run on many GPUs and creates a net for
    parameter_updates that can be run independently for number of iterations
    then followed by another net that runs once to compute the final parameter
    updates according to block wise model update filtering rule described
    in : Scalable Training of Deep Learning Machines by Incremental Block
    Training with Intra-block Parallel Optimization and Blockwise Model-Update
    Filtering (ICASSP 2016).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.Parallelize_BMUF', "Parallelize_BMUF(model_helper_obj, input_builder_fun, forward_pass_builder_fun, param_update_builder_fun, block_learning_rate=1.0, block_momentum=None, devices=None, rendezvous=None, net_type='dag', master_device=None, use_nccl=False, nesterov=False, optimize_gradient_memory=False, reset_momentum_sgd=False, warmup_iterations=None, max_concurrent_distributed_ops=4, add_blobs_to_sync=None, num_threads_per_device=4, cpu_device=False, barrier_net_timeout_sec=_DEFAULT_BARRIER_NET_TIMEOUT_SEC)", {'scope': scope, 'caffe2_pb2': caffe2_pb2, 'model_helper': model_helper, 'workspace': workspace, 'log': log, 'core': core, 'copy': copy, '_ForEachDevice': _ForEachDevice, '_ValidateParams': _ValidateParams, '_GroupByDevice': _GroupByDevice, 'viewkeys': viewkeys, '_AddGradientOperators': _AddGradientOperators, '_InferBlobDevice': _InferBlobDevice, '_SyncAllParams': _SyncAllParams, '_AllReduceBlobs': _AllReduceBlobs, 'AddBlobSync': AddBlobSync, '_OptimizeGradientMemorySimple': _OptimizeGradientMemorySimple, '_AddBarrierToModelNets': _AddBarrierToModelNets, 'model_helper_obj': model_helper_obj, 'input_builder_fun': input_builder_fun, 'forward_pass_builder_fun': forward_pass_builder_fun, 'param_update_builder_fun': param_update_builder_fun, 'block_learning_rate': block_learning_rate, 'block_momentum': block_momentum, 'devices': devices, 'rendezvous': rendezvous, 'net_type': net_type, 'master_device': master_device, 'use_nccl': use_nccl, 'nesterov': nesterov, 'optimize_gradient_memory': optimize_gradient_memory, 'reset_momentum_sgd': reset_momentum_sgd, 'warmup_iterations': warmup_iterations, 'max_concurrent_distributed_ops': max_concurrent_distributed_ops, 'add_blobs_to_sync': add_blobs_to_sync, 'num_threads_per_device': num_threads_per_device, 'cpu_device': cpu_device, 'barrier_net_timeout_sec': barrier_net_timeout_sec, '_DEFAULT_BARRIER_NET_TIMEOUT_SEC': _DEFAULT_BARRIER_NET_TIMEOUT_SEC}, 1)

def CreateNet(model, overwrite=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.CreateNet', 'CreateNet(model, overwrite=False)', {'workspace': workspace, 'model': model, 'overwrite': overwrite}, 0)

def RunInitNet(model):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.RunInitNet', 'RunInitNet(model)', {'workspace': workspace, 'CreateNet': CreateNet, 'model': model}, 0)

def RunWarmup(model):
    workspace.RunNet(model.net, model._warmup_iterations)
    workspace.RunNetOnce(model._warmup_broadcast)

def RunNet(model, num_iterations):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.RunNet', 'RunNet(model, num_iterations)', {'workspace': workspace, 'model': model, 'num_iterations': num_iterations}, 0)

def _AddBarrierToModelNets(model, barrier_net_timeout_sec):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._AddBarrierToModelNets', '_AddBarrierToModelNets(model, barrier_net_timeout_sec)', {'core': core, '_CreateBarrierNet': _CreateBarrierNet, 'model': model, 'barrier_net_timeout_sec': barrier_net_timeout_sec}, 0)

def _CreateBarrierNet(model, init_net, name_prefix, timeout_sec):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._CreateBarrierNet', '_CreateBarrierNet(model, init_net, name_prefix, timeout_sec)', {'log': log, '_CreateOrCloneCommonWorld': _CreateOrCloneCommonWorld, 'core': core, 'model': model, 'init_net': init_net, 'name_prefix': name_prefix, 'timeout_sec': timeout_sec}, 1)

def Synchronize(model, timeout_sec=_DEFAULT_BARRIER_NET_TIMEOUT_SEC):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.Synchronize', 'Synchronize(model, timeout_sec=_DEFAULT_BARRIER_NET_TIMEOUT_SEC)', {'warnings': warnings, 'core': core, '_CreateBarrierNet': _CreateBarrierNet, 'workspace': workspace, 'log': log, 'model': model, 'timeout_sec': timeout_sec, '_DEFAULT_BARRIER_NET_TIMEOUT_SEC': _DEFAULT_BARRIER_NET_TIMEOUT_SEC}, 1)

def ConvertNetForDevice(net, device=None):
    """
    Converts all blobs in the net to have namescope gpu_X, and correct
    device scope. You can use this to enable AppendNet with a
    forward_pass_builder_fun:

       def builder_fun(model):
          ...
          model.net.AppendNet(
             data_parallel_model.ConvertNetForDevice(othermodel.net))
          model.param_init_net.AppendNet(
             data_parallel_model.ConvertNetForDevice(othermodel.param_init_net))
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.ConvertNetForDevice', 'ConvertNetForDevice(net, device=None)', {'copy': copy, 'scope': scope, 'core': core, 'caffe2_pb2': caffe2_pb2, 'net': net, 'device': device}, 1)

def _ForEachDevice(devices, f, device_type, device_prefix, scoped=False, *args, **kwargs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._ForEachDevice', '_ForEachDevice(devices, f, device_type, device_prefix, scoped=False, *args, **kwargs)', {'core': core, 'devices': devices, 'f': f, 'device_type': device_type, 'device_prefix': device_prefix, 'scoped': scoped, 'args': args, 'kwargs': kwargs}, 0)

def _AddGradientOperators(devices, model, losses_by_gpu):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._AddGradientOperators', '_AddGradientOperators(devices, model, losses_by_gpu)', {'core': core, 'devices': devices, 'model': model, 'losses_by_gpu': losses_by_gpu}, 1)

def ExtractPredictorNet(model, inputs, outputs, device):
    """
    Returns (net, params) that can be exported to be used as a prediction
    net.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.ExtractPredictorNet', 'ExtractPredictorNet(model, inputs, outputs, device)', {'model_helper': model_helper, 'model': model, 'inputs': inputs, 'outputs': outputs, 'device': device}, 2)

def GetCheckpointParams(model):
    """
    Returns a set of blobs that are needed for a complete check point.
    They are blobs for the first gpu and iteration blobs.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.GetCheckpointParams', 'GetCheckpointParams(model)', {'_ComputeBlobsToSync': _ComputeBlobsToSync, 'model': model}, 1)

def FinalizeAfterCheckpoint(model, blobs=None, cpu_mode=False):
    """
    This function should be called after loading parameters from a
    checkpoint / initial parameters file.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.FinalizeAfterCheckpoint', 'FinalizeAfterCheckpoint(model, blobs=None, cpu_mode=False)', {'_ComputeBlobsToSync': _ComputeBlobsToSync, 'stripBlobName': stripBlobName, 'log': log, 'core': core, 'scope': scope, '_SyncAllParams': _SyncAllParams, 'workspace': workspace, 'model': model, 'blobs': blobs, 'cpu_mode': cpu_mode}, 0)

def GetLearningRateBlobNames(model):
    """
    Returns a list of learning rates blob names used in the optimizer.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.GetLearningRateBlobNames', 'GetLearningRateBlobNames(model)', {'caffe2_pb2': caffe2_pb2, 'core': core, 'model': model}, 1)

def _Broadcast(devices, model, net, param, use_nccl=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._Broadcast', '_Broadcast(devices, model, net, param, use_nccl=False)', {'_IsGPUBlob': _IsGPUBlob, 'core': core, 'viewvalues': viewvalues, 'workspace': workspace, '_IsIDEEPBlob': _IsIDEEPBlob, 'caffe2_pb2': caffe2_pb2, 'devices': devices, 'model': model, 'net': net, 'param': param, 'use_nccl': use_nccl}, 1)

def _AllReduce(devices, model, net, param, use_nccl=False, control_input=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._AllReduce', '_AllReduce(devices, model, net, param, use_nccl=False, control_input=None)', {'viewvalues': viewvalues, 'caffe2_pb2': caffe2_pb2, 'workspace': workspace, 'core': core, '_Broadcast': _Broadcast, 'devices': devices, 'model': model, 'net': net, 'param': param, 'use_nccl': use_nccl, 'control_input': control_input}, 1)

def _SyncAllParams(devices, model, init_net, net, rendezvous, unique_param_names, max_concurrent_distributed_ops=4):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._SyncAllParams', '_SyncAllParams(devices, model, init_net, net, rendezvous, unique_param_names, max_concurrent_distributed_ops=4)', {'_SyncAllParamsSingleHost': _SyncAllParamsSingleHost, '_SyncAllParamsDistributed': _SyncAllParamsDistributed, 'devices': devices, 'model': model, 'init_net': init_net, 'net': net, 'rendezvous': rendezvous, 'unique_param_names': unique_param_names, 'max_concurrent_distributed_ops': max_concurrent_distributed_ops}, 0)

def AddBlobSync(model, blobs, net=None):
    """
    Sync a blob across devices and hosts
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.AddBlobSync', 'AddBlobSync(model, blobs, net=None)', {'core': core, '_SyncAllParams': _SyncAllParams, 'model': model, 'blobs': blobs, 'net': net}, 1)

def AddDistributedBlobSync(model, blobs):
    """
    Sync blobs across machines (but not across devices)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.AddDistributedBlobSync', 'AddDistributedBlobSync(model, blobs)', {'_CreateOrCloneCommonWorld': _CreateOrCloneCommonWorld, 'model': model, 'blobs': blobs}, 1)

def _SyncAllParamsDistributed(devices, model, init_net, net, rendezvous, unique_param_names, max_concurrent_distributed_ops):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._SyncAllParamsDistributed', '_SyncAllParamsDistributed(devices, model, init_net, net, rendezvous, unique_param_names, max_concurrent_distributed_ops)', {'core': core, 'caffe2_pb2': caffe2_pb2, 'CollectivesConcurrencyControl': CollectivesConcurrencyControl, 'viewvalues': viewvalues, '_IsGPUBlob': _IsGPUBlob, '_IsIDEEPBlob': _IsIDEEPBlob, '_Broadcast': _Broadcast, 'devices': devices, 'model': model, 'init_net': init_net, 'net': net, 'rendezvous': rendezvous, 'unique_param_names': unique_param_names, 'max_concurrent_distributed_ops': max_concurrent_distributed_ops}, 0)

def _SyncAllParamsSingleHost(devices, model, net, unique_param_names):
    for param in unique_param_names:
        _Broadcast(devices, model, net, param)

def _AllReduceBlobs(blob_names, devices, model, net, rendezvous, use_nccl, max_concurrent_distributed_ops):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._AllReduceBlobs', '_AllReduceBlobs(blob_names, devices, model, net, rendezvous, use_nccl, max_concurrent_distributed_ops)', {'_AllReduceBlobsSingleHost': _AllReduceBlobsSingleHost, '_AllReduceBlobsDistributed': _AllReduceBlobsDistributed, 'blob_names': blob_names, 'devices': devices, 'model': model, 'net': net, 'rendezvous': rendezvous, 'use_nccl': use_nccl, 'max_concurrent_distributed_ops': max_concurrent_distributed_ops}, 0)

def _PruneParametersForSharing(model):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._PruneParametersForSharing', '_PruneParametersForSharing(model)', {'model': model}, 0)

def _RemapParameterBlobsForSharedModel(model, all_params):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._RemapParameterBlobsForSharedModel', '_RemapParameterBlobsForSharedModel(model, all_params)', {'log': log, 'stripBlobName': stripBlobName, 'model': model, 'all_params': all_params}, 0)


class CollectivesConcurrencyControl(object):
    """
    Creates common worlds (up to max_concurrent_context) and manage the
    sequential execution of collectives that shares the same context with
    cyclic control inputs.
    """
    
    def __init__(self, name, max_concurrent_context, param_init_net, rendezvous):
        self.name = name
        self.param_init_net = param_init_net
        self.max_concurrent_context = max_concurrent_context
        self.counter = 0
        self.common_worlds = []
        self.control_inputs = []
        self.rendezvous = rendezvous
    
    def get_control_and_context(self, control_output_blob):
        (common_world, control_input) = [None, None]
        current_slot = self.counter % self.max_concurrent_context
        if len(self.common_worlds) < self.max_concurrent_context:
            common_world = _CreateOrCloneCommonWorld(self.param_init_net, '{}_{}_cw'.format(self.name, current_slot), rendezvous=self.rendezvous)
            self.common_worlds.append(common_world)
            self.control_inputs.append(control_output_blob)
        else:
            common_world = self.common_worlds[current_slot]
            control_input = self.control_inputs[current_slot]
            self.control_inputs[current_slot] = control_output_blob
        self.counter += 1
        return (common_world, control_input)


def _AllReduceBlobsDistributed(blob_names, devices, model, net, rendezvous, max_concurrent_distributed_ops):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._AllReduceBlobsDistributed', '_AllReduceBlobsDistributed(blob_names, devices, model, net, rendezvous, max_concurrent_distributed_ops)', {'core': core, 'CollectivesConcurrencyControl': CollectivesConcurrencyControl, 'viewvalues': viewvalues, '_Broadcast': _Broadcast, 'blob_names': blob_names, 'devices': devices, 'model': model, 'net': net, 'rendezvous': rendezvous, 'max_concurrent_distributed_ops': max_concurrent_distributed_ops}, 0)

def _AllReduceBlobsSingleHost(blob_names, devices, model, net, use_nccl):
    """Performs NCCL AllReduce to distribute blobs to all the GPUs."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._AllReduceBlobsSingleHost', '_AllReduceBlobsSingleHost(blob_names, devices, model, net, use_nccl)', {'core': core, 'viewvalues': viewvalues, '_IsGPUBlob': _IsGPUBlob, '_AllReduce': _AllReduce, 'viewitems': viewitems, '_IsIDEEPBlob': _IsIDEEPBlob, 'caffe2_pb2': caffe2_pb2, '_Broadcast': _Broadcast, 'blob_names': blob_names, 'devices': devices, 'model': model, 'net': net, 'use_nccl': use_nccl}, 1)

def _BroadcastComputedParams(devices, model, rendezvous, use_nccl=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._BroadcastComputedParams', '_BroadcastComputedParams(devices, model, rendezvous, use_nccl=False)', {'_BroadcastComputedParamsSingleHost': _BroadcastComputedParamsSingleHost, '_BroadcastComputedParamsDistributed': _BroadcastComputedParamsDistributed, 'devices': devices, 'model': model, 'rendezvous': rendezvous, 'use_nccl': use_nccl}, 0)

def _BroadcastComputedParamsDistributed(devices, model, rendezvous, use_nccl=False):
    _BroadcastComputedParamsSingleHost(devices, model, use_nccl)
    log.warn('Distributed broadcast of computed params is not implemented yet')

def _BroadcastComputedParamsSingleHost(devices, model, use_nccl=False):
    """
    Average computed params over all devices
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._BroadcastComputedParamsSingleHost', '_BroadcastComputedParamsSingleHost(devices, model, use_nccl=False)', {'_Broadcast': _Broadcast, 'devices': devices, 'model': model, 'use_nccl': use_nccl}, 1)

def _GetReverseOrderedGrads(model):
    """
    Returns the gradients in reverse order (namespace stripped),
    for the optimal synchronization order.
    """
    return list(reversed(model._grad_names))

def stripBlobName(param):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.stripBlobName', 'stripBlobName(param)', {'core': core, 'stripBlobName': stripBlobName, 'scope': scope, 'param': param}, 1)

def _AnalyzeOperators(model):
    """
    Look at all the operators and check that they do not cross device scopes
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._AnalyzeOperators', '_AnalyzeOperators(model)', {'core': core, 'model': model}, 0)

def _InferBlobDevice(model):
    """
    Assign blob to device option based on the operator outputing it
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._InferBlobDevice', '_InferBlobDevice(model)', {'caffe2_pb2': caffe2_pb2, 'model': model}, 0)

def _IsIDEEPBlob(model, blob_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._IsIDEEPBlob', '_IsIDEEPBlob(model, blob_name)', {'caffe2_pb2': caffe2_pb2, 'model': model, 'blob_name': blob_name}, 1)

def _IsGPUBlob(model, blob_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._IsGPUBlob', '_IsGPUBlob(model, blob_name)', {'core': core, 'model': model, 'blob_name': blob_name}, 1)

def _GroupByDevice(model, devices, params, non_data_params):
    """
    Groups blobs by device, returning a map of [blobname] = {0: BlobRef, 1: ..}.
    Returns ordered dictionary, ensuring the original order.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._GroupByDevice', '_GroupByDevice(model, devices, params, non_data_params)', {'OrderedDict': OrderedDict, 'core': core, 'stripBlobName': stripBlobName, 'model': model, 'devices': devices, 'params': params, 'non_data_params': non_data_params}, 1)

def _ValidateParams(params):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._ValidateParams', '_ValidateParams(params)', {'params': params}, 0)

def _ComputeBlobsToSync(model):
    """
    We sync all blobs that are generated by param init net and
    are 'data parallel', i.e assigned to a device
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._ComputeBlobsToSync', '_ComputeBlobsToSync(model)', {'stripBlobName': stripBlobName, 'scope': scope, 'core': core, 'model': model}, 2)

def _OptimizeGradientMemorySimple(model, losses_by_gpu, devices):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._OptimizeGradientMemorySimple', '_OptimizeGradientMemorySimple(model, losses_by_gpu, devices)', {'log': log, 'memonger': memonger, 'viewvalues': viewvalues, 'model': model, 'losses_by_gpu': losses_by_gpu, 'devices': devices}, 0)

def _AddDynamicMemoryOptimization(model, blobs_to_keep, devices):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._AddDynamicMemoryOptimization', '_AddDynamicMemoryOptimization(model, blobs_to_keep, devices)', {'viewvalues': viewvalues, 'memonger': memonger, 'model': model, 'blobs_to_keep': blobs_to_keep, 'devices': devices}, 0)

def OptimizeGradientMemory(model, input_shapes, excluded_blobs, recycle_activations):
    """
    Optimize memory usage of the backward pass by recycling blobs for gradient
    inputs that have been 'used'.
    input_shapes:  dict of blob name to shape for the inputs of the model.
                   Pass empty dictionary if not known.
    excluded_blobs: list of blobs that cannot be recycled. These are blobs
                   that you will access externally.
    recycle_activations: whether to also recycle forward pass activations
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model.OptimizeGradientMemory', 'OptimizeGradientMemory(model, input_shapes, excluded_blobs, recycle_activations)', {'viewitems': viewitems, 'workspace': workspace, 'memonger': memonger, 'viewvalues': viewvalues, 'model': model, 'input_shapes': input_shapes, 'excluded_blobs': excluded_blobs, 'recycle_activations': recycle_activations}, 0)

def _CreateOrCloneCommonWorld(net, common_world_blob, rendezvous, name=None, timeout_sec=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._CreateOrCloneCommonWorld', '_CreateOrCloneCommonWorld(net, common_world_blob, rendezvous, name=None, timeout_sec=None)', {'_DEFAULT_TIMEOUT_SEC': _DEFAULT_TIMEOUT_SEC, 'net': net, 'common_world_blob': common_world_blob, 'rendezvous': rendezvous, 'name': name, 'timeout_sec': timeout_sec}, 1)

def _RunComparison(model, blob_name, device=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._RunComparison', '_RunComparison(model, blob_name, device=None)', {'core': core, 'np': np, 'workspace': workspace, 'model': model, 'blob_name': blob_name, 'device': device}, 1)

def _InterleaveOps(model):
    """
    Data Parallel Model creates a net with ops in one device grouped together.
    This will interleave the ops so that each op for each device is next
    to each other in the net. Kind of like combining decks of cards. This
    ensures that progress is made along the critical path roughly concurrently
    for each device, which is important due to the extra intra-node
    synchronization required for multi-device batch normalization.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._InterleaveOps', '_InterleaveOps(model)', {'model': model}, 0)

def _CPUInterDeviceBatchNormalization(model):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._CPUInterDeviceBatchNormalization', '_CPUInterDeviceBatchNormalization(model)', {'core': core, 'stripBlobName': stripBlobName, 'utils': utils, 'model': model}, 1)

def _GPUInterDeviceBatchNormalization(model):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.data_parallel_model._GPUInterDeviceBatchNormalization', '_GPUInterDeviceBatchNormalization(model)', {'core': core, 'caffe2_pb2': caffe2_pb2, 'stripBlobName': stripBlobName, 'utils': utils, 'model': model}, 1)

