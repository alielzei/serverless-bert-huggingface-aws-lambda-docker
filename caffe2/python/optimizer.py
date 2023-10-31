from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import namedtuple, defaultdict
from past.builtins import basestring
import logging
import copy
import numpy as np
from caffe2.python import core, scope, utils, workspace
from caffe2.python.modeling import parameter_info
from caffe2.proto import caffe2_pb2
_LEARNING_RATE_INJECTION = 'lr_injection'
AuxOptimizerParams = namedtuple('AuxOptimizerParams', ['local', 'shared'])
_optimizer_instance_count = defaultdict(int)
FP16_ENGINES = ['SIMD_Q_FP16', 'SIMD_Q_STOC_FP16', 'SIMD_Q_STOC_MKL_FP16']
logger = logging.getLogger(__name__)


class Optimizer(object):
    
    def __init__(self):
        self._aux_params = AuxOptimizerParams(local=[], shared=[])
        self._instance_num = _optimizer_instance_count[self.__class__.__name__]
        _optimizer_instance_count[self.__class__.__name__] += 1
        self._lr_multiplier = None
        self._local_lr_multiplier = None
        self._local_lr_multiplier_on_gpu = False
    "\n    Adds optimization operators to the net for given parameter and its gradient\n    Parameter is specified by either 'param' being a ParameterInfo object.\n    In this case  param.grad has to be set\n\n    Or by 'param' being a BlobReference and 'grad' being a BlobReference for its\n    gradient.\n    "
    
    def __call__(self, net, param_init_net, param, grad=None):
        if grad is None:
            assert isinstance(param, parameter_info.ParameterInfo), 'Expected parameter to be of type ParameterInfo, got {}'.format(param)
            assert param.grad is not None
        else:
            if isinstance(param, basestring):
                param = core.BlobReference(param)
            param = parameter_info.ParameterInfo(param_id=None, param=param, grad=grad)
        self._run(net, param_init_net, param)
    
    def _run(self, net, param_init_net, param_info):
        raise Exception('Not Implemented')
    
    def get_cpu_blob_name(self, base_str, node_name=''):
        classname = self.__class__.__name__
        return '%s_%d_%s%s_cpu' % (classname, self._instance_num, base_str, node_name)
    
    def get_gpu_blob_name(self, base_str, gpu_id, node_name):
        classname = self.__class__.__name__
        return '%s_%d_%s%s_gpu%d' % (classname, self._instance_num, base_str, node_name, gpu_id)
    
    @property
    def attributes(self):
        attr = copy.deepcopy(self.__dict__)
        del attr['_instance_num']
        return attr
    
    def make_unique_blob_name(self, base_str):
        """
        Returns a blob name that will be unique to the current device
        and optimizer instance.
        """
        current_scope = scope.CurrentDeviceScope()
        if current_scope is None:
            return self.get_cpu_blob_name(base_str)
        if core.IsGPUDeviceType(current_scope.device_type):
            return self.get_gpu_blob_name(base_str, current_scope.device_id, current_scope.node_name)
        else:
            return self.get_cpu_blob_name(base_str, current_scope.node_name)
    
    def build_lr(self, net, param_init_net, base_learning_rate, learning_rate_blob=None, policy='fixed', iter_val=0, **kwargs):
        if learning_rate_blob is None:
            learning_rate_blob = self.make_unique_blob_name('lr')
        iteration = utils.BuildUniqueMutexIter(param_init_net, net, iter_val=iter_val)
        if not net.BlobIsDefined(learning_rate_blob):
            lr = net.LearningRate([iteration], learning_rate_blob, base_lr=-base_learning_rate, policy=policy, **kwargs)
        else:
            lr = net.GetBlobRef(learning_rate_blob)
        if self._lr_multiplier is not None:
            lr_multiplier = net.CopyFromCPUInput(self._lr_multiplier, self.make_unique_blob_name('lr_multiplier'))
            lr = net.Mul([lr, lr_multiplier], self.make_unique_blob_name('scaled_lr'), broadcast=1)
        if self._local_lr_multiplier is not None:
            current_scope = scope.CurrentDeviceScope()
            if (current_scope is not None and core.IsGPUDeviceType(current_scope.device_type) and not self._local_lr_multiplier_on_gpu):
                local_lr_multiplier = net.CopyFromCPUInput(self._local_lr_multiplier, self.make_unique_blob_name('local_lr_multiplier'))
            else:
                local_lr_multiplier = self._local_lr_multiplier
            lr = net.Mul([lr, local_lr_multiplier], self.make_unique_blob_name('local_scaled_lr'), broadcast=1)
        return (lr, iteration)
    
    def add_lr_multiplier(self, lr_multiplier):
        """
        Set the global learning rate multiplier. If a multiplier already
        existed, this will overwrite the existing multiplier. The multiplier is
        used for all future calls to _run(), unless it is overwritten.
        """
        self._lr_multiplier = lr_multiplier
    
    def _add_local_lr_multiplier(self, local_lr_multiplier, is_gpu_blob=False):
        """
        Set the local learning rate multiplier. This local multiplier is
        multiplied with the global learning rate multiplier if it exists. As
        with the global learning rate multiplier, this multiplier will be
        used for all future calls to _run(), so please call
        _clear_local_lr_multiplier() at the beginning of the optimizer's _run()
        before optionally calling this function.
        """
        self._local_lr_multiplier = local_lr_multiplier
        self._local_lr_multiplier_on_gpu = is_gpu_blob
    
    def _clear_local_lr_multiplier(self):
        self._local_lr_multiplier = None
        self._local_lr_multiplier_on_gpu = False
    
    @staticmethod
    def dedup(net, sparse_dedup_aggregator, grad):
        assert isinstance(grad, core.GradientSlice), 'Dedup only works for sparse gradient, got {}'.format(grad)
        if sparse_dedup_aggregator:
            return net.DeduplicateGradientSlices(grad, aggregator=sparse_dedup_aggregator)
        else:
            return grad
    
    def get_auxiliary_parameters(self):
        """Returns a list of auxiliary parameters.

        Returns:
            aux_params: A namedtuple, AuxParams.

            aux_params.local stores a list of blobs. Each blob is a local
            auxiliary parameter. A local auxiliary parameter is a parameter in
            parallel to a learning rate parameter. Take adagrad as an example,
            the local auxiliary parameter is the squared sum parameter, because
            every learning rate has a squared sum associated with it.

            aux_params.shared also stores a list of blobs. Each blob is a shared
            auxiliary parameter. A shared auxiliary parameter is a parameter
            that is shared across all the learning rate parameters. Take adam as
            an example, the iteration parameter is a shared parameter, because
            all the learning rates share the same iteration parameter.
        """
        return self._aux_params
    
    def scale_learning_rate(self, *args, **kwargs):
        raise NotImplementedError('Optimizer Need to Implement `scale_learning_rate` method.')
    
    def create_lars_inputs(self, param_init_net, weight_decay, trust, lr_max):
        wd = param_init_net.ConstantFill([], 'weight_decay', shape=[1], value=weight_decay)
        trust = param_init_net.ConstantFill([], 'trust', shape=[1], value=trust)
        lr_max = param_init_net.ConstantFill([], 'lr_max', shape=[1], value=lr_max)
        return (wd, trust, lr_max)



class SgdOptimizer(Optimizer):
    
    def __init__(self, base_learning_rate=0.01, policy='fixed', momentum=0.0, nesterov=1, sparse_dedup_aggregator=None, lars=None, **kwargs):
        super(SgdOptimizer, self).__init__()
        self.base_learning_rate = base_learning_rate
        self.policy = policy
        self.momentum = momentum
        self.nesterov = nesterov
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.lars = lars
        self.init_kwargs = kwargs
    
    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad
        if self.base_learning_rate == 0:
            return
        assert self.base_learning_rate > 0, 'Expect positive base learning rate, got {}'.format(self.base_learning_rate)
        self._clear_local_lr_multiplier()
        if (self.lars is not None and not isinstance(grad, core.GradientSlice)):
            assert self.lars >= 0, 'Lars offset must be nonnegative, got {}'.format(self.lars)
            (wd, trust, lr_max) = self.create_lars_inputs(param_init_net, 0.0, 1.0, np.finfo(np.float32).max)
            lr_lars_multiplier = net.Lars([param, grad, wd, trust, lr_max], self.make_unique_blob_name(str(param) + '_lars'), offset=self.lars, lr_min=0.0)
            current_scope = scope.CurrentDeviceScope()
            self._add_local_lr_multiplier(lr_lars_multiplier, is_gpu_blob=(current_scope is not None and core.IsGPUDeviceType(current_scope.device_type)))
        lr_sign = (-1 if self.momentum else 1)
        (lr, _) = self.build_lr(net, param_init_net, base_learning_rate=self.base_learning_rate * lr_sign, policy=self.policy, **self.init_kwargs)
        dev = scope.CurrentDeviceScope()
        if dev is None:
            dev = core.DeviceOption(caffe2_pb2.CPU)
        ONE = param_init_net.ConstantFill([], 'ONE_{}_{}{}'.format(dev.device_type, dev.device_id, dev.node_name), shape=[1], value=1.0)
        self._aux_params.shared.append(ONE)
        if self.momentum > 0:
            momentum_data = param_init_net.ConstantFill(param, str(param) + '_momentum', value=0.0)
            self._aux_params.local.append(momentum_data)
        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            if self.momentum > 0.0:
                net.SparseMomentumSGDUpdate([grad.values, momentum_data, lr, param, grad.indices], [grad.values, momentum_data, param], momentum=self.momentum, nesterov=self.nesterov)
            else:
                net.ScatterWeightedSum([param, ONE, grad.indices, grad.values, lr], param)
        elif self.momentum > 0.0:
            net.MomentumSGDUpdate([grad, momentum_data, lr, param], [grad, momentum_data, param], momentum=self.momentum, nesterov=self.nesterov)
        else:
            coeff = lr
            net.WeightedSum([param, ONE, grad, coeff], param)
    
    def scale_learning_rate(self, scale):
        self.base_learning_rate *= scale
        return



class MultiPrecisionSgdOptimizer(SgdOptimizer):
    
    def __init__(self, base_learning_rate=0.1, momentum=0.0, policy='fixed', nesterov=1, sparse_dedup_aggregator=None, **kwargs):
        super(MultiPrecisionSgdOptimizer, self).__init__(base_learning_rate=base_learning_rate, policy=policy, momentum=momentum, nesterov=nesterov, sparse_dedup_aggregator=sparse_dedup_aggregator, **kwargs)
    
    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        param_fp32 = (param_info.blob_copy[core.DataType.FLOAT] if param_info.blob_copy is not None else None)
        if param_fp32 is None:
            return SgdOptimizer._run(self, net, param_init_net, param_info)
        grad = param_info.grad
        if self.base_learning_rate == 0:
            return
        assert self.base_learning_rate > 0, 'Expect positive base learning rate, got {}'.format(self.base_learning_rate)
        (lr, _) = self.build_lr(net, param_init_net, base_learning_rate=-self.base_learning_rate, policy=self.policy, **self.init_kwargs)
        momentum_data = param_init_net.ConstantFill(param_fp32, str(param) + '_momentum', value=0.0)
        self._aux_params.local.append(momentum_data)
        assert not isinstance(grad, core.GradientSlice), 'MultiPrecisionSgd does not support sparse gradients'
        grad_fp32 = net.HalfToFloat(grad, grad + '_fp32')
        net.MomentumSGDUpdate([grad_fp32, momentum_data, lr, param_fp32], [grad_fp32, momentum_data, param_fp32], momentum=self.momentum, nesterov=self.nesterov)
        net.FloatToHalf(param_fp32, param)



class FP16SgdOptimizer(SgdOptimizer):
    
    def __init__(self, base_learning_rate=0.1, momentum=0.0, policy='fixed', nesterov=1, weight_decay=0.0001, sparse_dedup_aggregator=None, **kwargs):
        super(FP16SgdOptimizer, self).__init__(base_learning_rate=base_learning_rate, policy=policy, momentum=momentum, nesterov=nesterov, sparse_dedup_aggregator=sparse_dedup_aggregator, **kwargs)
        self.weight_decay = weight_decay
    
    def _run(self, net, param_init_net, param_info, fp32_update=False):
        fp32_update_flag = 0
        param_name = str(param_info.blob)
        if param_name.find('spatbn') != -1:
            fp32_update = True
        if fp32_update:
            fp32_update_flag = 1
            param = param_info.blob
            param_fp32 = param_info.blob
        elif param_info.blob_copy is None:
            fp32_update_flag = 1
            param = param_info.blob
            param_fp32 = param_info.blob
        elif core.DataType.FLOAT in param_info.blob_copy:
            param = param_info.blob
            param_fp32 = param_info.blob_copy[core.DataType.FLOAT]
        elif core.DataType.FLOAT16 in param_info.blob_copy:
            param = param_info.blob_copy[core.DataType.FLOAT16]
            param_fp32 = param_info.blob
        else:
            assert False, 'Unrecognized parameter format to be updated by FP16 Optimizer. Parameter: {}'.format(param_info.name)
        grad = param_info.grad
        if self.base_learning_rate == 0:
            return
        assert self.base_learning_rate > 0, 'Expect positive base learning rate, got {}'.format(self.base_learning_rate)
        (lr, _) = self.build_lr(net, param_init_net, base_learning_rate=-self.base_learning_rate, policy=self.policy, **self.init_kwargs)
        momentum_data_fp32 = param_init_net.ConstantFill(param_fp32, str(param) + '_momentum_fp32', value=0.0)
        momentum_data = param_init_net.FloatToHalf(momentum_data_fp32, str(param) + '_momentum')
        self._aux_params.local.append(momentum_data)
        assert not isinstance(grad, core.GradientSlice), 'FP16Sgd does not support sparse gradients'
        if fp32_update_flag == 0:
            net.FP16MomentumSGDUpdate([grad, momentum_data, lr, param], [grad, momentum_data, param], momentum=self.momentum, nesterov=self.nesterov, weight_decay=self.weight_decay)
        else:
            net.FP32MomentumSGDUpdate([grad, momentum_data_fp32, lr, param], [grad, momentum_data_fp32, param], momentum=self.momentum, nesterov=self.nesterov, weight_decay=self.weight_decay)



class WeightDecayBuilder(Optimizer):
    
    def __init__(self, weight_decay):
        self.weight_decay = weight_decay
    
    def _run(self, net, param_init_net, param_info):
        dev = scope.CurrentDeviceScope()
        if dev is None:
            dev = core.DeviceOption(caffe2_pb2.CPU)
        ONE = param_init_net.ConstantFill([], 'ONE_{}_{}'.format(dev.device_type, dev.device_id), shape=[1], value=1.0)
        WD = param_init_net.ConstantFill([], 'wd_{}_{}'.format(dev.device_type, dev.device_id), shape=[1], value=self.weight_decay)
        if isinstance(param_info.grad, core.GradientSlice):
            raise ValueError('Weight decay does not yet support sparse gradients')
        else:
            net.WeightedSum([param_info.grad, ONE, param_info.blob, WD], param_info.grad)



class AdagradOptimizer(Optimizer):
    
    def __init__(self, alpha=0.01, epsilon=0.0001, decay=1, policy='fixed', sparse_dedup_aggregator=None, rowWise=False, engine='', lars=None, output_effective_lr=False, output_effective_lr_and_update=False, pruning_options=None, **kwargs):
        super(AdagradOptimizer, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay = decay
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.rowWise = rowWise
        self.engine = engine
        self.lars = lars
        self.output_effective_lr = output_effective_lr
        self.output_effective_lr_and_update = output_effective_lr_and_update
        self.init_kwargs = kwargs
        self._process_pruning_options(pruning_options)
    
    def _process_pruning_options(self, pruning_options):
        self.use_mask = False
        if pruning_options is None:
            pruning_options = {}
        else:
            assert isinstance(pruning_options, dict), 'pruning_options can only be provided as a dictionary, currently: {}'.format(pruning_options)
        self.mask_tensor = pruning_options.get('mask_tensor', None)
        self.mask_db_path = pruning_options.get('mask_db_path', None)
        self.mask_db_type = pruning_options.get('mask_db_type', None)
        self.mask_blob_name = pruning_options.get('mask_blob_name', None)
        if self.mask_tensor is not None:
            assert type(self.mask_tensor) is np.ndarray, 'mask_tensor must be a numpy array!'
            assert self.mask_db_path is None, 'mask can be provided through either a numpy array or a db path, not both'
            assert self.mask_db_type is None, 'mask can be provided through either a numpy array or a db path, not both'
            assert self.mask_blob_name is None, 'mask can be provided through either a numpy array or a db path, not both'
            self.use_mask = True
        if (self.mask_db_path is not None or self.mask_db_type is not None or self.mask_blob_name is not None):
            assert self.mask_db_path is not None, 'when mask is provided through db, db path, db type, and blob name are all needed'
            assert self.mask_db_type is not None, 'when mask is provided through db, db path, db type, and blob name are all needed'
            assert self.mask_blob_name is not None, 'when mask is provided through db, db path, db type, and blob name are all needed'
            assert self.mask_tensor is None, 'mask can be provided through either a numpy array or a db path, not both'
            self.use_mask = True
    
    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad
        if self.alpha <= 0:
            return
        self._clear_local_lr_multiplier()
        if (self.lars is not None and not isinstance(grad, core.GradientSlice)):
            assert self.lars >= 0, 'Lars offset must be nonnegative, got {}'.format(self.lars)
            (wd, trust, lr_max) = self.create_lars_inputs(param_init_net, 0.0, 1.0, np.finfo(np.float32).max)
            lr_lars_multiplier = net.Lars([param, grad, wd, trust, lr_max], self.make_unique_blob_name(str(param) + '_lars'), offset=self.lars, lr_min=0.0)
            current_scope = scope.CurrentDeviceScope()
            self._add_local_lr_multiplier(lr_lars_multiplier, is_gpu_blob=(current_scope is not None and core.IsGPUDeviceType(current_scope.device_type)))
        (lr, _) = self.build_lr(net, param_init_net, base_learning_rate=self.alpha, policy=self.policy, **self.init_kwargs)
        if self.rowWise:
            logger.info('Using engine {} for rowWise Adagrad'.format(self.engine))
            (shapes, types) = workspace.InferShapesAndTypes([param_init_net])
            if str(param) not in shapes:
                shape = param_init_net.Shape(param, str(param) + '_shape')
                num_rows = param_init_net.Slice([shape], str(shape) + '_numrows', starts=[0], ends=[1])
                param_squared_sum = param_init_net.ConstantFill(num_rows, str(param) + '_avg_squared_sum', input_as_shape=1, value=0.0)
            else:
                param_squared_sum = param_init_net.ConstantFill([], str(param) + '_avg_squared_sum', shape=[shapes[str(param)][0]], value=0.0)
        else:
            logger.info('Using engine {} for regular Adagrad'.format(self.engine))
            if self.engine in FP16_ENGINES:
                (shapes, types) = workspace.InferShapesAndTypes([param_init_net])
                assert str(param) in shapes, shapes
                shape = shapes[str(param)]
                param_squared_sum = param_init_net.Float16ConstantFill([], str(param) + '_squared_sum', value=0.0, shape=shape)
            else:
                param_squared_sum = param_init_net.ConstantFill([param], str(param) + '_squared_sum', value=0.0)
        if self.use_mask is True:
            if self.mask_tensor is not None:
                if not isinstance(grad, core.GradientSlice):
                    mask_blob = param_init_net.GivenTensorFill([], [str(param) + '_mask'], values=self.mask_tensor, shape=self.mask_tensor.shape)
                else:
                    self.mask_tensor = self.mask_tensor.astype(np.uint8)
                    mask_blob = param_init_net.GivenTensorBoolFill([], [str(param) + '_mask'], values=self.mask_tensor, shape=self.mask_tensor.shape)
                    mask_blob = param_init_net.Cast(mask_blob, to=core.DataType.UINT8)
                    mask_changed_blob = param_init_net.ConstantFill([], [str(param) + '_mask_changed_blob'], value=False, dtype=core.DataType.BOOL, shape=[1])
            elif (self.mask_db_path is not None or self.mask_db_type is not None or self.mask_blob_name is not None):
                mask_blob = param_init_net.Load([], self.mask_blob_name, db=self.mask_db_path, db_type=self.mask_db_type, absolute_path=True)
                if isinstance(grad, core.GradientSlice):
                    mask_changed_blob = param_init_net.ConstantFill([], [str(param) + '_mask_changed_blob'], value=False, dtype=core.DataType.BOOL, shape=[1])
            else:
                raise NotImplementedError('If mask is used, it needs to be provided through a numpy array or a db file')
        self._aux_params.local.append(param_squared_sum)
        if self.rowWise:
            assert isinstance(grad, core.GradientSlice), 'If SparseAdagrad with rowWise=True, gradient must be a gradientslice. PLease ensure that rowWise is not enabled for the dense Adagrad optimizer, as it is not supported.'
        if isinstance(grad, core.GradientSlice):
            assert self.decay == 1.0, 'Decay is not implemented for SparseAdagrad and must be set to 1'
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            input_args = [param, param_squared_sum, grad.indices, grad.values, lr]
            if self.rowWise:
                if self.use_mask is True:
                    op = 'MaskedRowWiseSparseAdagrad'
                    input_args += [mask_blob, mask_changed_blob]
                else:
                    op = 'RowWiseSparseAdagrad'
            elif self.use_mask is True:
                op = 'MaskedSparseAdagrad'
                input_args += [mask_blob, mask_changed_blob]
            else:
                op = 'SparseAdagrad'
            net.__getattr__(op)(input_args, [param, param_squared_sum], epsilon=self.epsilon, engine=self.engine)
        else:
            output_args = [param, param_squared_sum]
            if self.output_effective_lr_and_update:
                assert self.use_mask is False, "MaskedAdagrad doesn't support outputting effective_lr_and_update"
                output_args.append(str(param) + '_effective_lr')
                output_args.append(str(param) + '_update')
            elif self.output_effective_lr:
                assert self.use_mask is False, "MaskedAdagrad doesn't support outputting effective_lr"
                output_args.append(str(param) + '_effective_lr')
            if self.use_mask:
                net.MaskedAdagrad([param, param_squared_sum, grad, lr, mask_blob], output_args, epsilon=self.epsilon, decay=float(self.decay), engine=self.engine)
            else:
                net.Adagrad([param, param_squared_sum, grad, lr], output_args, epsilon=self.epsilon, decay=float(self.decay), engine=self.engine)
    
    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return



class WngradOptimizer(Optimizer):
    
    def __init__(self, alpha=1.0, epsilon=1e-09, policy='fixed', sparse_dedup_aggregator=None, engine='', moment_init=100.0, lars=None, output_effective_lr=False, output_effective_lr_and_update=False, **kwargs):
        super(WngradOptimizer, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.engine = engine
        self.moment_init = moment_init
        self.lars = lars
        self.output_effective_lr = output_effective_lr
        self.output_effective_lr_and_update = output_effective_lr_and_update
        self.init_kwargs = kwargs
    
    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad
        if self.alpha <= 0:
            return
        self._clear_local_lr_multiplier()
        if (self.lars is not None and not isinstance(grad, core.GradientSlice)):
            assert self.lars >= 0, 'Lars offset must be nonnegative, got {}'.format(self.lars)
            (wd, trust, lr_max) = self.create_lars_inputs(param_init_net, 0.0, 1.0, np.finfo(np.float32).max)
            lr_lars_multiplier = net.Lars([param, grad, wd, trust, lr_max], self.make_unique_blob_name(str(param) + '_lars'), offset=self.lars, lr_min=0.0)
            current_scope = scope.CurrentDeviceScope()
            self._add_local_lr_multiplier(lr_lars_multiplier, is_gpu_blob=(current_scope is not None and core.IsGPUDeviceType(current_scope.device_type)))
        (lr, _) = self.build_lr(net, param_init_net, base_learning_rate=self.alpha, policy=self.policy, **self.init_kwargs)
        moment = param_init_net.ConstantFill([], str(param) + '_moment', shape=[1], value=self.moment_init)
        self._aux_params.local.append(moment)
        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            net.SparseWngrad([param, moment, grad.indices, grad.values, lr], [param, moment], epsilon=self.epsilon, engine=self.engine)
        else:
            output_args = [param, moment]
            if self.output_effective_lr_and_update:
                output_args.append(str(param) + '_effective_lr')
                output_args.append(str(param) + '_update')
            elif self.output_effective_lr:
                output_args.append(str(param) + '_effective_lr')
            net.Wngrad([param, moment, grad, lr], output_args, epsilon=self.epsilon, engine=self.engine)
    
    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return



class AdadeltaOptimizer(Optimizer):
    
    def __init__(self, alpha=0.01, epsilon=0.0001, decay=0.95, policy='fixed', sparse_dedup_aggregator=None, engine='', **kwargs):
        """Constructor function to add Adadelta Optimizer

        Args:
            alpha: learning rate
            epsilon: attribute of Adadelta to avoid numerical issues
            decay: attribute of Adadelta to decay the squared gradient sum
            policy: specifies how learning rate should be applied, options are
              "fixed", "step", "exp", etc.
            sparse_dedup_aggregator: specifies deduplication strategy for
              gradient slices. Works while using sparse gradients. Options
              include "mean" and "sum".
            engine: the engine used, options include "", "CUDNN", etc.
        """
        super(AdadeltaOptimizer, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay = decay
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.engine = engine
        self.init_kwargs = kwargs
    
    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad
        if self.alpha <= 0:
            return
        (lr, _) = self.build_lr(net, param_init_net, base_learning_rate=self.alpha, policy=self.policy, **self.init_kwargs)
        moment = param_init_net.ConstantFill([param], str(param) + '_squared_moment', value=0.0)
        moment_update = param_init_net.ConstantFill([param], str(param) + '_squared_moment_update', value=0.0)
        self._aux_params.local.append(moment)
        self._aux_params.local.append(moment_update)
        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            net.SparseAdadelta([param, moment, moment_update, grad.indices, grad.values, lr], [param, moment, moment_update], epsilon=self.epsilon, decay=self.decay, engine=self.engine)
        else:
            net.Adadelta([param, moment, moment_update, grad, lr], [param, moment, moment_update], epsilon=self.epsilon, decay=self.decay, engine=self.engine)
    
    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return



class FtrlOptimizer(Optimizer):
    
    def __init__(self, alpha=0.01, beta=0.0001, lambda1=0, lambda2=0, sparse_dedup_aggregator=None, engine=''):
        super(FtrlOptimizer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.engine = engine
    
    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad
        if self.alpha <= 0:
            return
        nz = param_init_net.ConstantFill([param], str(param) + '_ftrl_nz', extra_shape=[2], value=0.0)
        self._aux_params.local.append(nz)
        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            net.SparseFtrl([param, nz, grad.indices, grad.values], [param, nz], engine=self.engine, alpha=self.alpha, beta=self.beta, lambda1=self.lambda1, lambda2=self.lambda2)
        else:
            net.Ftrl([param, nz, grad], [param, nz], engine=self.engine, alpha=self.alpha, beta=self.beta, lambda1=self.lambda1, lambda2=self.lambda2)
    
    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return



class GFtrlOptimizer(Optimizer):
    """Group Lasso FTRL Optimizer."""
    
    def __init__(self, alpha=0.01, beta=0.0001, lambda1=0, lambda2=0, sparse_dedup_aggregator=None, engine=''):
        super(GFtrlOptimizer, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.engine = engine
    
    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad
        if self.alpha <= 0:
            return
        nz = param_init_net.ConstantFill([param], str(param) + '_gftrl_nz', extra_shape=[2], value=0.0)
        self._aux_params.local.append(nz)
        net.GFtrl([param, nz, grad], [param, nz], engine=self.engine, alpha=self.alpha, beta=self.beta, lambda1=self.lambda1, lambda2=self.lambda2)
    
    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return



class AdamOptimizer(Optimizer):
    
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, policy='fixed', use_lr_adaption=False, lr_alpha=0.01, normalized_lr_adaption=True, sparse_dedup_aggregator=None, rowWise=False, engine='', enableRAdam=False, **kwargs):
        super(AdamOptimizer, self).__init__()
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.policy = policy
        self.use_lr_adaption = use_lr_adaption
        self.lr_alpha = lr_alpha
        self.normalized_lr_adaption = normalized_lr_adaption
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.rowWise = rowWise
        self.engine = engine
        self.enableRAdam = enableRAdam
        self.init_kwargs = kwargs
    
    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad
        if self.alpha <= 0:
            return
        (lr, iteration) = self.build_lr(net, param_init_net, base_learning_rate=self.alpha, policy=self.policy, **self.init_kwargs)
        m1 = param_init_net.ConstantFill([param], param + '_first_moment', value=0.0)
        if self.rowWise:
            (shapes, types) = workspace.InferShapesAndTypes([param_init_net])
            m2 = param_init_net.ConstantFill([], param + '_avg_second_moment', shape=[shapes[param][0]], value=0.0)
        else:
            m2 = param_init_net.ConstantFill([param], param + '_second_moment', value=0.0)
        self._aux_params.shared.append(iteration)
        self._aux_params.local.append(m1)
        self._aux_params.local.append(m2)
        if self.rowWise:
            assert isinstance(grad, core.GradientSlice), 'If SparseAdam with rowWise=True, gradient must be a gradientslice. PLease ensure that rowWise is not enabled for the dense Adam optimizer, as it is not supported.'
        output_blobs = [param, m1, m2]
        if self.use_lr_adaption:
            effective_grad = str(param) + '_effective_grad'
            output_blobs.append(effective_grad)
        if isinstance(grad, core.GradientSlice):
            grad = self.dedup(net, self.sparse_dedup_aggregator, grad)
            if self.rowWise:
                op = 'RowWiseSparseAdam'
            else:
                op = 'SparseAdam'
            if op == 'SparseAdam':
                net.__getattr__(op)([param, m1, m2, grad.indices, grad.values, lr, iteration], output_blobs, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon, enableRAdam=self.enableRAdam)
            else:
                assert not self.enableRAdam, 'Currently, RowWiseSparseAdam is not supported by RAdam!'
                net.__getattr__(op)([param, m1, m2, grad.indices, grad.values, lr, iteration], output_blobs, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon)
            if self.use_lr_adaption:
                net.LearningRateAdaption([lr, grad.values, effective_grad], [lr], lr_alpha=self.lr_alpha, normalized_lr_adaption=self.normalized_lr_adaption)
        else:
            net.Adam([param, m1, m2, grad, lr, iteration], output_blobs, beta1=self.beta1, beta2=self.beta2, epsilon=self.epsilon)
            if self.use_lr_adaption:
                net.LearningRateAdaption([lr, grad, effective_grad], [lr], lr_alpha=self.lr_alpha, normalized_lr_adaption=self.normalized_lr_adaption)
    
    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return



class YellowFinOptimizer(Optimizer):
    """YellowFin: An automatic tuner for momentum SGD

    See https://arxiv.org/abs/1706.03471 for more details. This implementation
    has separate learning rate and momentum per each parameter."""
    
    def __init__(self, alpha=0.1, mu=0.0, beta=0.999, curv_win_width=20, zero_debias=True, epsilon=0.1**6, policy='fixed', sparse_dedup_aggregator=None, **kwargs):
        super(YellowFinOptimizer, self).__init__()
        self.alpha = alpha
        self.mu = mu
        self.beta = beta
        self.curv_win_width = curv_win_width
        self.zero_debias = zero_debias
        self.epsilon = epsilon
        self.policy = policy
        self.sparse_dedup_aggregator = sparse_dedup_aggregator
        self.init_kwargs = kwargs
    
    def _run(self, net, param_init_net, param_info):
        SCALARS_MEMORY_SIZE = 5
        param = param_info.blob
        grad = param_info.grad
        moment = param_init_net.ConstantFill([param], param + '_moment', value=0.0)
        curv_win = param_init_net.ConstantFill([], param + '_curv_win', shape=[self.curv_win_width], value=0.0)
        g_avg = param_init_net.ConstantFill([param], param + '_g_avg', value=0.0)
        g2_avg = param_init_net.ConstantFill([param], param + '_g2_avg', value=0.0)
        lr_avg = param_init_net.ConstantFill([], param + '_lr_avg', shape=[1], value=self.alpha)
        mu_avg = param_init_net.ConstantFill([], param + '_mu_avg', shape=[1], value=self.mu)
        scalars_memory = param_init_net.ConstantFill([], param + '_scalars_memory', shape=[SCALARS_MEMORY_SIZE], value=0.0)
        assert self.alpha > 0
        assert not isinstance(grad, core.GradientSlice), 'YellowFin does not support sparse gradients'
        iteration = utils.BuildUniqueMutexIter(param_init_net, net, iter_val=0)
        self._aux_params.shared.append(iteration)
        self._aux_params.local.append(moment)
        self._aux_params.local.append(lr_avg)
        self._aux_params.local.append(mu_avg)
        self._aux_params.local.append(curv_win)
        self._aux_params.local.append(g_avg)
        self._aux_params.local.append(g2_avg)
        self._aux_params.local.append(scalars_memory)
        yf_in_out_args = [param, moment, lr_avg, mu_avg, curv_win, g_avg, g2_avg, scalars_memory]
        net.YellowFin(yf_in_out_args + [grad, iteration], yf_in_out_args, beta=self.beta, epsilon=self.epsilon, curv_win_width=self.curv_win_width, zero_debias=self.zero_debias)
    
    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return



class RmsPropOptimizer(Optimizer):
    
    def __init__(self, alpha=0.01, decay=0.9, momentum=0.0, epsilon=1e-05, policy='fixed', engine='', **kwargs):
        super(RmsPropOptimizer, self).__init__()
        self.alpha = alpha
        self.decay = decay
        self.momentum = momentum
        self.epsilon = epsilon
        self.policy = policy
        self.engine = engine
        self.init_kwargs = kwargs
    
    def _run(self, net, param_init_net, param_info):
        param = param_info.blob
        grad = param_info.grad
        assert self.alpha > 0
        assert not isinstance(grad, core.GradientSlice), "RmsPropOptimizer doesn't support sparse gradients"
        dev = scope.CurrentDeviceScope()
        if dev is None:
            dev = core.DeviceOption(caffe2_pb2.CPU)
        ONE = param_init_net.ConstantFill([], 'ONE_{}_{}'.format(dev.device_type, dev.device_id), shape=[1], value=1.0)
        (lr, _) = self.build_lr(net, param_init_net, base_learning_rate=-self.alpha, policy=self.policy, **self.init_kwargs)
        grad_o = param_init_net.ConstantFill([param], str(param) + '_grad_o', values=0.0)
        ms = param_init_net.ConstantFill([param], str(param) + '_mean_squares', values=0.0)
        mom = param_init_net.ConstantFill([param], str(param) + '_momentum', values=0.0)
        self._aux_params.local.append(ms)
        self._aux_params.local.append(mom)
        net.RmsProp([grad, ms, mom, ONE], [grad_o, ms, mom], decay=self.decay, momentum=self.momentum, epsilon=self.epsilon, engine=self.engine)
        net.MomentumSGDUpdate([grad_o, mom, lr, param], [grad_o, mom, param])
    
    def scale_learning_rate(self, scale):
        self.alpha *= scale
        return


def _get_param_to_device(model):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.optimizer._get_param_to_device', '_get_param_to_device(model)', {'core': core, 'model': model}, 1)

def get_param_device(param_name, grad, param_to_device=None, default_device=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.optimizer.get_param_device', 'get_param_device(param_name, grad, param_to_device=None, default_device=None)', {'core': core, 'param_name': param_name, 'grad': grad, 'param_to_device': param_to_device, 'default_device': default_device}, 1)

def get_lr_injection():
    """
    Gets current value for lr_injection, a multiplier for all base
    learning rates.
    Must set allow_lr_injection=True when building optimizer, as it
    relies on synchronization over CPU.
    """
    return workspace.FetchBlob(_LEARNING_RATE_INJECTION)

def set_lr_injection(lr_injection_value):
    """
    Sets lr_injection, a multiplier for all base learning rates.
    Must set allow_lr_injection=True when building optimizer, as it
    relies on synchronization over CPU.
    """
    workspace.FeedBlob(_LEARNING_RATE_INJECTION, np.array([float(lr_injection_value)], dtype=np.float32))

def _calc_norm_ratio(model, params, name_scope, param_to_device, max_gradient_norm):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.optimizer._calc_norm_ratio', '_calc_norm_ratio(model, params, name_scope, param_to_device, max_gradient_norm)', {'core': core, 'get_param_device': get_param_device, 'caffe2_pb2': caffe2_pb2, 'model': model, 'params': params, 'name_scope': name_scope, 'param_to_device': param_to_device, 'max_gradient_norm': max_gradient_norm}, 1)

def _build(model, optimizer, weights_only=False, use_param_info_optim=True, max_gradient_norm=None, allow_lr_injection=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.optimizer._build', '_build(model, optimizer, weights_only=False, use_param_info_optim=True, max_gradient_norm=None, allow_lr_injection=False)', {'_get_param_to_device': _get_param_to_device, '_calc_norm_ratio': _calc_norm_ratio, '_LEARNING_RATE_INJECTION': _LEARNING_RATE_INJECTION, 'get_param_device': get_param_device, 'core': core, 'model': model, 'optimizer': optimizer, 'weights_only': weights_only, 'use_param_info_optim': use_param_info_optim, 'max_gradient_norm': max_gradient_norm, 'allow_lr_injection': allow_lr_injection}, 1)

def add_weight_decay(model, weight_decay):
    """Adds a decay to weights in the model.

    This is a form of L2 regularization.

    Args:
        weight_decay: strength of the regularization
    """
    _build(model, WeightDecayBuilder(weight_decay=weight_decay), weights_only=True, use_param_info_optim=False)

def build_sgd(model, base_learning_rate, max_gradient_norm=None, allow_lr_injection=False, **kwargs):
    sgd_optimizer = SgdOptimizer(base_learning_rate, **kwargs)
    return _build(model, sgd_optimizer, max_gradient_norm=max_gradient_norm, allow_lr_injection=allow_lr_injection)

def build_multi_precision_sgd(model, base_learning_rate, max_gradient_norm=None, allow_lr_injection=False, **kwargs):
    multi_prec_sgd_optimizer = MultiPrecisionSgdOptimizer(base_learning_rate, **kwargs)
    return _build(model, multi_prec_sgd_optimizer, max_gradient_norm=max_gradient_norm, allow_lr_injection=allow_lr_injection)

def build_fp16_sgd(model, base_learning_rate, **kwargs):
    fp16_sgd_optimizer = FP16SgdOptimizer(base_learning_rate, **kwargs)
    return _build(model, fp16_sgd_optimizer)

def build_ftrl(model, engine='SIMD', **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.optimizer.build_ftrl', "build_ftrl(model, engine='SIMD', **kwargs)", {'core': core, 'FtrlOptimizer': FtrlOptimizer, '_build': _build, 'model': model, 'engine': engine, 'kwargs': kwargs}, 1)

def build_gftrl(model, engine='', **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.optimizer.build_gftrl', "build_gftrl(model, engine='', **kwargs)", {'core': core, 'GFtrlOptimizer': GFtrlOptimizer, '_build': _build, 'model': model, 'engine': engine, 'kwargs': kwargs}, 1)

def build_adagrad(model, base_learning_rate, parameters=None, max_gradient_norm=None, allow_lr_injection=False, **kwargs):
    adagrad_optimizer = AdagradOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(model, adagrad_optimizer, max_gradient_norm=max_gradient_norm, allow_lr_injection=allow_lr_injection)

def build_wngrad(model, base_learning_rate, parameters=None, max_gradient_norm=None, allow_lr_injection=False, **kwargs):
    wngrad_optimizer = WngradOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(model, wngrad_optimizer, max_gradient_norm=max_gradient_norm, allow_lr_injection=allow_lr_injection)

def build_adadelta(model, base_learning_rate, parameters=None, max_gradient_norm=None, allow_lr_injection=False, **kwargs):
    adadelta_optimizer = AdadeltaOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(model, adadelta_optimizer, max_gradient_norm=max_gradient_norm, allow_lr_injection=allow_lr_injection)

def build_adam(model, base_learning_rate, max_gradient_norm=None, allow_lr_injection=False, **kwargs):
    adam_optimizer = AdamOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(model, adam_optimizer, max_gradient_norm=max_gradient_norm, allow_lr_injection=allow_lr_injection)

def build_yellowfin(model, base_learning_rate=0.1, **kwargs):
    yellowfin_optimizer = YellowFinOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(model, yellowfin_optimizer)

def build_rms_prop(model, base_learning_rate, max_gradient_norm=None, allow_lr_injection=False, **kwargs):
    rms_prop_optimizer = RmsPropOptimizer(alpha=base_learning_rate, **kwargs)
    return _build(model, rms_prop_optimizer, max_gradient_norm=max_gradient_norm, allow_lr_injection=allow_lr_injection)

