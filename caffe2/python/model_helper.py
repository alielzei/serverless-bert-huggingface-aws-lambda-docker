from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, scope, workspace
from caffe2.python.helpers.db_input import db_input
from caffe2.python.modeling import parameter_info
from caffe2.python.modeling.parameter_sharing import parameter_sharing_context
from caffe2.python.optimizer_context import OptimizerContext, DEFAULT_OPTIM
from caffe2.python.regularizer_context import RegularizerContext
from future.utils import viewitems, viewkeys
from itertools import chain
import logging
import six
_known_working_ops = ['Accuracy', 'Adam', 'Add', 'Adagrad', 'SparseAdagrad', 'Adadelta', 'SparseAdadelta', 'AveragedLoss', 'Cast', 'Checkpoint', 'ConstantFill', 'Copy', 'CopyGPUToCPU', 'CopyCPUToGPU', 'DequeueBlobs', 'EnsureCPUOutput', 'ExpandDims', 'Flatten', 'FlattenToVec', 'LabelCrossEntropy', 'LearningRate', 'MakeTwoClass', 'MatMul', 'NCCLAllreduce', 'NHWC2NCHW', 'PackSegments', 'Print', 'PRelu', 'ReduceFrontSum', 'Scale', 'ScatterWeightedSum', 'Sigmoid', 'SortedSegmentSum', 'Snapshot', 'Softmax', 'SoftmaxWithLoss', 'SquaredL2Distance', 'Squeeze', 'StopGradient', 'Summarize', 'Tanh', 'Transpose', 'UnpackSegments', 'WeightedSum', 'YellowFin']


class ModelHelper(object):
    """A helper model so we can manange models more easily. It contains net def
    and parameter storages. You can add an Operator yourself, e.g.

        model = model_helper.ModelHelper(name="train_net")
        # init your weight and bias as w and b
        w = model.param_init_net.XavierFill(...)
        b = model.param_init_net.ConstantFill(...)
        fc1 = model.FC([input, w, b], output, **kwargs)

    or you can use helper functions in brew module without manually
    defining parameter initializations and operators.

        model = model_helper.ModelHelper(name="train_net")
        fc1 = brew.fc(model, input, output, dim_in, dim_out, **kwargs)

    """
    
    def __init__(self, name=None, init_params=True, allow_not_known_ops=True, skip_sparse_optim=False, param_model=None, arg_scope=None):
        self.name = (name or 'model')
        self.net = core.Net(self.name)
        if param_model is not None:
            self.param_init_net = param_model.param_init_net
            self.param_to_grad = param_model.param_to_grad
            self.params = param_model.params
            self._parameters_info = param_model._parameters_info
            self._computed_params = param_model._computed_params
        else:
            self.param_init_net = core.Net(self.name + '_init')
            self.param_to_grad = {}
            self.params = []
            self._parameters_info = {}
            self._computed_params = []
        self._param_info_deprecated = []
        self._devices = []
        self.gradient_ops_added = False
        self.init_params = init_params
        self.allow_not_known_ops = allow_not_known_ops
        self.skip_sparse_optim = skip_sparse_optim
        self.weights = []
        self.biases = []
        self._arg_scope = {'order': 'NCHW', 'use_cudnn': True, 'cudnn_exhaustive_search': False}
        if arg_scope is not None:
            self._arg_scope.update(arg_scope)
    
    @property
    def arg_scope(self):
        return self._arg_scope
    
    def get_name(self):
        return self.name
    
    def _infer_param_shape(self, param):
        for op in self.param_init_net.Proto().op:
            if str(param) in op.output:
                for arg in op.arg:
                    if arg.name == 'shape':
                        return list(arg.ints)
        return None
    
    def _update_param_info_deprecated(self):
        assert len(self._param_info_deprecated) <= len(self.params)
        for param in self.params[len(self._param_info_deprecated):]:
            if not isinstance(param, core.BlobReference):
                raise ValueError('Param %s must be a BlobReference!' % str(param))
            self._param_info_deprecated.append(parameter_info.ParameterInfo(param_id=len(self._param_info_deprecated), param=param, shape=self._infer_param_shape(param)))
        for info in self._param_info_deprecated:
            info.grad = self.param_to_grad.get(info.name)
    
    def _normalize_tags(self, tags):
        tags = (tags or [])
        return (set(tags) if isinstance(tags, list) else set([tags]))
    
    def create_param(self, param_name, shape, initializer, tags=None):
        """
        Creates parameter with a given name and initializer.

        If param_name is instance of BlobRefernce - then this blob will be used
        to store parameter (no any logic will affect it's location).

        If param_name is instance of a string type, then the final blob will
        be created in the CurrentNameScope with the respect of all parameter
        sharing logic, i.e. 'resolved_name_scope/param_name'.

        Parameter sharing logic is going to override CurrentNameScope according
        to the rules that are specified through ParameterSharing contexts,
        all ParameterSharing contexts are applied recursively until there are no
        extra overrides present, where on each step the best match will be
        applied first.

        The following examples should clarify the way ParameterSharing logic
        works:

        As an example if this function is called with parameter 'w':
        a. Call from some scope 'global_scope' with no Parameter sharing:
          'global_scope/w'
        b. Call from scope 'scope_b', with override {'scope_b': 'scope_a'}:
          'scope_a/w'
        c. Call from scope 'scope_a', with override {'scope_a': ''}:
          'scope_a/w'
        d. Call from scope 'scope_b/shared', with overrides
          {'scope_b/shared': 'scope_b', 'scope_b': 'scope_a'}:
          'scope_a/w'
        d. Call from scope 'scope_b/unshared', with overrides
          {'scope_b/shared': 'scope_b', 'scope_b': 'scope_a'}:
          'scope_a/unshared/w'
        """
        if isinstance(param_name, core.BlobReference):
            param_name = str(param_name)
        elif isinstance(param_name, six.string_types):
            param_name = parameter_sharing_context.get_parameter_name(param_name)
        else:
            raise TypeError('Unsupported type for param_name')
        if param_name in self._parameters_info:
            assert self._parameters_info[param_name].shape == shape
            return self._parameters_info[param_name].blob
        param_info = initializer.create_param(param_name=core.BlobReference(param_name), init_net=self.param_init_net, shape=shape)
        optim_context = OptimizerContext.current()
        for tag in self._normalize_tags(tags):
            if optim_context.has_optimizer(tag):
                param_info.optimizer = optim_context.get_optimizer(tag)
        if (not param_info.optimizer and optim_context.has_optimizer(DEFAULT_OPTIM)):
            param_info.optimizer = optim_context.get_optimizer(DEFAULT_OPTIM)
        reg_context = RegularizerContext.current()
        param_info.regularizer = reg_context
        self._parameters_info[param_name] = param_info
        self.AddParameter(param_info.blob, tags)
        return param_info.blob
    
    def get_param_info(self, param):
        assert isinstance(param, core.BlobReference), 'Param {} is not a BlobReference'.format(param)
        return self._parameters_info.get(param, None)
    
    def add_param_DEPRECATED(self, param, key=None, shape=None, length=None):
        logging.warning('add_param method is DEPRECATED')
        self._update_param_info_deprecated()
        self.AddParameter(param)
        if (key is not None and self.net.input_record() is not None):
            idx = self.net.input_record().field_blobs().index(key)
            key = self.net.input_record().field_names()[idx]
        shape = (shape if shape is not None else self._infer_param_shape(param))
        if not isinstance(param, core.BlobReference):
            raise ValueError('Param %s must be a BlobReference!' % str(param))
        self._param_info_deprecated.append(parameter_info.ParameterInfo(param_id=len(self._param_info_deprecated), param=param, shape=shape, key=key, length=length))
        return self._param_info_deprecated[-1]
    
    def AddParameter(self, param, tags=None):
        assert isinstance(param, core.BlobReference)
        tags = self._normalize_tags(tags)
        if parameter_info.ParameterTags.COMPUTED_PARAM in tags:
            self._computed_params.append(param)
        else:
            self.params.append(param)
        if parameter_info.ParameterTags.WEIGHT in tags:
            self.weights.append(param)
        if parameter_info.ParameterTags.BIAS in tags:
            self.biases.append(param)
    
    @staticmethod
    def _NormalizeNamescope(namescope):
        if namescope is None:
            return scope.CurrentNameScope()
        elif (namescope == '' or namescope.endswith(scope._NAMESCOPE_SEPARATOR)):
            return namescope
        else:
            return namescope + scope._NAMESCOPE_SEPARATOR
    
    def GetParams(self, namescope=None, top_scope=False):
        """
        Returns the params in current namescope
        """
        namescope = ModelHelper._NormalizeNamescope(namescope)
        if namescope == '':
            return self.params[:]
        else:
            return [p for p in self.params if p.GetNameScope().startswith(namescope)]
    
    def Proto(self):
        return self.net.Proto()
    
    def InitProto(self):
        return self.param_init_net.Proto()
    
    def RunAllOnGPU(self, *args, **kwargs):
        self.param_init_net.RunAllOnGPU(*args, **kwargs)
        self.net.RunAllOnGPU(*args, **kwargs)
    
    def CreateDB(self, blob_out, db, db_type, **kwargs):
        dbreader = self.param_init_net.CreateDB([], blob_out, db=db, db_type=db_type, **kwargs)
        return dbreader
    
    def AddGradientOperators(self, *args, **kwargs):
        if self.gradient_ops_added:
            raise RuntimeError('You cannot run AddGradientOperators twice.')
        self.Validate()
        self.gradient_ops_added = True
        self.grad_map = self.net.AddGradientOperators(*args, **kwargs)
        self.param_to_grad = self.get_param_to_grad(self.params)
        for (param, grad) in self.param_to_grad.items():
            param_info = self.get_param_info(param)
            if param_info:
                param_info.grad = grad
            else:
                self._parameters_info[param] = parameter_info.ParameterInfo(param_id=None, param=param, grad=grad)
        return self.grad_map
    
    def get_param_to_grad(self, params):
        """
        Given a list of parameters returns a dict from a parameter
        to a corresponding gradient
        """
        param_to_grad = {}
        if not self.gradient_ops_added:
            raise RuntimeError('You need to run AddGradientOperators first.')
        for p in params:
            if str(p) in self.grad_map:
                param_to_grad[p] = self.grad_map[str(p)]
        return param_to_grad
    
    def GetOptimizationParamInfo(self, params=None):
        """
        Returns a map for param => grad.
        If params is not specified, all parameters will be considered.
        """
        if not self.gradient_ops_added:
            raise RuntimeError('Need to call AddGradientOperators first')
        param_to_grad = self.param_to_grad
        if params:
            param_to_grad = self.get_param_to_grad(params)
        return [self.get_param_info(param) for (param, grad) in viewitems(param_to_grad) if (not self.skip_sparse_optim or not isinstance(grad, core.GradientSlice))]
    
    def _Validate(self):
        """
        Check for duplicate params
        """
        params_list = [str(p) for p in self.params]
        params_set = set(params_list)
        dupes = []
        if len(params_set) != len(params_list):
            params_list = sorted(params_list)
            for (j, p) in enumerate(params_list):
                if (j > 0 and params_list[j - 1] == p):
                    if p not in dupes:
                        dupes.append(p)
        return dupes
    
    def Validate(self):
        dupes = self._Validate()
        assert dupes == [], 'Duplicate params: {}'.format(dupes)
    
    def GetComputedParams(self, namescope=None):
        """
        Returns the computed params in current namescope. 'Computed params'
        are such parameters that are not optimized via gradient descent but are
        directly computed from data, such as the running mean and variance
        of Spatial Batch Normalization.
        """
        namescope = ModelHelper._NormalizeNamescope(namescope)
        if namescope == '':
            return self._computed_params[:]
        else:
            return [p for p in self._computed_params if p.GetNameScope().startswith(namescope)]
    
    def GetAllParams(self, namescope=None):
        return self.GetParams(namescope) + self.GetComputedParams(namescope)
    
    def TensorProtosDBInput(self, unused_blob_in, blob_out, batch_size, db, db_type, **kwargs):
        """TensorProtosDBInput."""
        assert len(unused_blob_in) == 0, 'You cannot pass reader to model_helper.TensorProtosDBInput.\n               Use model.net.TensorProtosDBInput instead to create the op.'
        return db_input(self, blob_out, batch_size, db, db_type, **kwargs)
    
    def GetDevices(self):
        assert len(self._devices) > 0, 'Use data_parallel_model to run model on multiple GPUs.'
        return self._devices
    
    def __getattr__(self, op_type):
        """Catch-all for all other operators, mostly those without params."""
        if op_type.startswith('__'):
            raise AttributeError(op_type)
        if not core.IsOperator(op_type):
            raise AttributeError('Method ' + op_type + ' is not a registered operator.' + ' Did you mean: [' + ','.join(workspace.C.nearby_opnames(op_type)) + ']')
        if op_type not in _known_working_ops:
            if not self.allow_not_known_ops:
                raise AttributeError('Operator {} is not known to be safe'.format(op_type))
            logging.warning('You are creating an op that the ModelHelper does not recognize: {}.'.format(op_type))
        return self.net.__getattr__(op_type)
    
    def __dir__(self):
        return sorted(set(chain(dir(type(self)), viewkeys(self.__dict__), _known_working_ops)))
    
    def GetCompleteNet(self):
        """ Return param_init_net + net Net.
        Returns:
          'core.Net' containing param_init_net and net
        """
        new_net = self.param_init_net.Clone(self.name + '_complete_net', keep_schema=True)
        for op in new_net.Proto().op:
            op.debug_info = op.debug_info + '/param_init_net'
        new_net.AppendNet(self.net)
        if self.net.Proto().HasField('type'):
            new_net.Proto().type = self.net.Proto().type
        return new_net
    
    def ConstructInitTrainNetfromNet(self, net):
        """ construct init net and train net from complete_net
        Inputs:
          net: 'core.Net' containing param_init_net and train net
        """
        param_op_mask = []
        train_op_mask = []
        for (idx, op) in enumerate(net.Proto().op):
            if op.debug_info.endswith('/param_init_net'):
                param_op_mask.append(idx)
            else:
                train_op_mask.append(idx)
        self.param_init_net = net.Clone(net.Name() + '/generated_param_init_net', keep_schema=True, op_id_mask=param_op_mask, update_external_list=True)
        self.net = net.Clone(net.Name() + '/generated_net', keep_schema=True, op_id_mask=train_op_mask, update_external_list=True)


def ExtractPredictorNet(net_proto, input_blobs, output_blobs, device=None, renames=None, disabled_inputs=None):
    """
    Takes a model net for training and returns a net which can be
    used for prediction. For example, all gradient operators and
    input operators are removed.
    @param net_proto protobuf of the net you want to process (net.Proto())
    @param input_blobs list/set of blob names that are the inputs of predictor
    @param output_blobs list/set of blob names that are outputs of predictor
    @param device optional device option that is assigned
    @param renames dictionary of blob name to a new name (optional)
    @param disabled_inputs optional set of blobs that are 'switched off'. This
                will cause branches with those blobs as inputs to be removed
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.model_helper.ExtractPredictorNet', 'ExtractPredictorNet(net_proto, input_blobs, output_blobs, device=None, renames=None, disabled_inputs=None)', {'core': core, 'logging': logging, 'net_proto': net_proto, 'input_blobs': input_blobs, 'output_blobs': output_blobs, 'device': device, 'renames': renames, 'disabled_inputs': disabled_inputs}, 2)

