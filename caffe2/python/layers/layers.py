from __future__ import absolute_import, division, print_function, unicode_literals
import logging
from collections import namedtuple
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, schema, scope, utils, workspace
from caffe2.python.layers.tags import TagContext
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
IdList = schema.List(np.int64)
IdScoreList = schema.Map(np.int64, np.float32)
IdListWithEvicted = schema.ListWithEvicted(np.int64)
IdScoreListWithEvicted = schema.MapWithEvicted(np.int64, np.float32)

def almost_equal_schemas(record, original_schema, check_field_names=True, check_field_types=True, check_field_metas=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.layers.layers.almost_equal_schemas', 'almost_equal_schemas(record, original_schema, check_field_names=True, check_field_types=True, check_field_metas=False)', {'IdList': IdList, 'schema': schema, 'IdListWithEvicted': IdListWithEvicted, 'IdScoreList': IdScoreList, 'IdScoreListWithEvicted': IdScoreListWithEvicted, 'record': record, 'original_schema': original_schema, 'check_field_names': check_field_names, 'check_field_types': check_field_types, 'check_field_metas': check_field_metas}, 1)

def get_key(record):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.layers.layers.get_key', 'get_key(record)', {'almost_equal_schemas': almost_equal_schemas, 'IdList': IdList, 'IdScoreList': IdScoreList, 'record': record}, 1)

def get_categorical_limit(record):
    key = get_key(record)
    return key.metadata.categorical_limit

def get_avg_length(record):
    return record['lengths'].metadata.expected_value

def set_request_only(field):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.layers.layers.set_request_only', 'set_request_only(field)', {'schema': schema, 'np': np, 'field': field}, 0)


class InstantiationContext(object):
    """
    List of contexts where layer could be instantitated
    """
    ACCUMULATE_PRED = 'accumulate_pred'
    EVAL = 'eval'
    PREDICTION = 'prediction'
    TRAINING = 'training'

_LAYER_REGISTRY = {}

def register_layer(name, layer):
    assert name not in _LAYER_REGISTRY, '{0} already exists'.format(name)
    _LAYER_REGISTRY[name] = layer

def layer_exists(name):
    return name in _LAYER_REGISTRY

def get_layer_class(name):
    return _LAYER_REGISTRY[name]

def create_layer(layer_name, *args, **kwargs):
    return _LAYER_REGISTRY[layer_name](*args, **kwargs)
LayerPsParam = namedtuple('LayerPsParam', ['sparse_key', 'average_length'])


class LayerParameter(object):
    
    def __init__(self, parameter=None, optimizer=None, initializer=None, ps_param=None, regularizer=None):
        assert isinstance(parameter, core.BlobReference), 'expect {0} to be a blob reference'.format(str(parameter))
        self._shape = None
        self.parameter = parameter
        self.optimizer = optimizer
        self.initializer = initializer
        self.ps_param = ps_param
        self.regularizer = regularizer
    
    @property
    def initializer(self):
        return self._initializer
    
    @initializer.setter
    def initializer(self, op):
        assert (op is None or core.IsOperator(getattr(op, 'type', None))), 'initializer expects an operator, got type: {}'.format(type(op))
        self._initializer = op
        if op is not None:
            self.shape = self._infer_shape_from_initializer()
    
    @property
    def shape(self):
        return self._shape
    
    @shape.setter
    def shape(self, shape):
        assert (self.shape is None or self.shape == shape), 'inconsistent shape for layer parameter: {}, expect: {}, but got {}'.format(self, self.shape, shape)
        self._shape = shape
    
    def _infer_shape_from_initializer(self):
        for arg in self.initializer.arg:
            if arg.name == 'shape':
                return list(arg.ints)
        with workspace.WorkspaceGuard('model_init_by_loading_params'):
            try:
                net = core.Net('shape_checker')
                net._net.op.extend([self.initializer])
                shape_blob = net.NextScopedBlob(self.parameter + '_shape')
                net.Shape([self.parameter], shape_blob)
                workspace.RunNetOnce(net)
                shape = workspace.FetchBlob(shape_blob).tolist()
                workspace.ResetWorkspace()
                return shape
            except RuntimeError as exp:
                logger.warning('Cannot infer the shape of blob {} from operator {}: {}'.format(self.parameter, self.initializer.type, exp))
                workspace.ResetWorkspace()
                return None
    
    def __str__(self):
        return str(self.parameter)


def is_request_only_scalar(scalar):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.layers.layers.is_request_only_scalar', 'is_request_only_scalar(scalar)', {'scalar': scalar}, 1)
AccessedFeatures = namedtuple('AccessedFeatures', ['type', 'ids'])


class ModelLayer(object):
    
    def __init__(self, model, prefix, input_record, predict_input_record_fields=None, tags=None, **kwargs):
        """
        Base class for model layers. Layer is an abstraction that allows to
        provide model description in terms of meta-operators, where each of the
        meta-operators can have different implementations for training,
        evaluation and prediction, that are instantiated later. As an example
        SampledSoftmax can do something related to sampling depending on
        supervision during the training and just apply softmax if it's used for
        prediction/evaluation.

        All inputs/outputs from layers are represented as a record (instance of
        schema bounded to blobs) and are accessible through input_record and
        output_schema. If Layer needs to have only a subset of inputs/provides
        subset of outputs during the inference - it should provide
        predict_input_record and predict_output_schema correspondingly (those
        records are expected to be a subset of input_record/output_schema).

        Each layer has a list of Tags associated with it, that depends on
        current context and arguments. It's possible to use those tags during
        the instantiation time.

        """
        self.name = model.next_layer_name(prefix)
        self.model = model
        self.kwargs = kwargs
        self._input_record = input_record
        if predict_input_record_fields:
            if not isinstance(predict_input_record_fields, list):
                predict_input_record_fields = [predict_input_record_fields]
            self._predict_input_record = self._input_record[predict_input_record_fields]
        else:
            self._predict_input_record = None
        self.request_only = True
        if len(input_record.all_scalars()) == 0:
            self.request_only = False
        for scalar in input_record.all_scalars():
            if not is_request_only_scalar(scalar):
                self.request_only = False
                break
        self.precomputation_request_only = False
        self.precomputation_object_only = False
        self._output_schema = None
        self._predict_output_schema = None
        self.eval_output_schema = None
        self.tags = set((tags or []))
        self.tags.update(TagContext.current().tags)
        self.params = []
        self._export_output_for_metrics = False
        self._export_params_for_metrics = False
    
    def get_type(self):
        return self.__class__.__name__
    
    def _check_output_schema(self):
        assert self._output_schema is not None, 'Schema is not initialized'
        assert (self._predict_output_schema is None or schema.is_schema_subset(self._predict_output_schema, self._output_schema)), 'predict_output_schema is not a subset of the output_schema'
    
    @property
    def predict_input_record(self):
        return (self._predict_input_record or self._input_record)
    
    @property
    def input_record(self):
        return self._input_record
    
    @property
    def predict_output_schema(self):
        self._check_output_schema()
        return (self._predict_output_schema or self._output_schema)
    
    @predict_output_schema.setter
    def predict_output_schema(self, output_schema):
        assert self._predict_output_schema is None
        self._predict_output_schema = output_schema
    
    @property
    def output_schema(self):
        if self.request_only:
            set_request_only(self._output_schema)
        self._check_output_schema()
        return self._output_schema
    
    @output_schema.setter
    def output_schema(self, output_schema):
        assert self._output_schema is None
        self._output_schema = output_schema
    
    def get_parameters(self):
        return self.params
    
    def get_fp16_compatible_parameters(self):
        """Return a subset of parameters which can be converted to fp16"""
        return []
    
    def get_memory_usage(self):
        return 0
    
    def get_accessed_features(self):
        """
        Return a map from field to list of AccessedFeatures, the map should
        contain all features accessed in the model layer
        """
        return {}
    
    def add_init_params(self, init_net):
        """
        Adds layer initialization operators to passed net.
        """
        for param in self.params:
            init_op = param.initializer
            current_device_scope = scope.CurrentDeviceScope()
            if not init_op:
                continue
            if (not init_op.HasField('device_option') and current_device_scope):
                init_op = caffe2_pb2.OperatorDef()
                init_op.CopyFrom(param.initializer)
                init_op.device_option.CopyFrom(current_device_scope)
            if any((utils.OpAlmostEqual(op, init_op, 'debug_info') for op in init_net._net.op)):
                continue
            init_net._net.op.extend([init_op])
    
    def create_param(self, param_name, shape, initializer, optimizer, ps_param=None, regularizer=None):
        with scope.NameScope(self.name, reset=True):
            param = self.model.create_param(param_name=param_name, shape=shape, initializer=initializer, optimizer=optimizer, ps_param=ps_param, regularizer=regularizer)
            assert all((param.parameter != p.parameter for p in self.params))
            self.params.append(param)
            return param.parameter
    
    def get_next_blob_reference(self, name):
        with scope.NameScope(self.name, reset=True):
            return self.model.net.NextScopedBlob(name)
    
    def add_operators(self, net, init_net=None, context=InstantiationContext.TRAINING):
        """
        Adds layer trainig or initialization operators to the passed in net.
        init_net can be None and can be called independently from add_init_params
        """
        with scope.NameScope(self.name):
            if context not in {InstantiationContext.PREDICTION, InstantiationContext.EVAL, InstantiationContext.ACCUMULATE_PRED}:
                assert init_net, "Only prediction and eval context don't need init_net"
            if init_net:
                self.add_init_params(init_net)
            if context == InstantiationContext.TRAINING:
                self.add_train_ops(net)
            elif context == InstantiationContext.EVAL:
                self.add_eval_ops(net)
            elif context == InstantiationContext.ACCUMULATE_PRED:
                self.add_ops_to_accumulate_pred(net)
            else:
                self.add_ops(net)
            if (context in {InstantiationContext.TRAINING, InstantiationContext.EVAL} and self._export_params_for_metrics):
                self.add_param_copy_operators(net)
    
    def add_ops(self, net):
        raise NotImplementedError
    
    def add_eval_ops(self, net):
        self.add_ops(net)
    
    def add_train_ops(self, net):
        self.add_eval_ops(net)
    
    def add_ops_to_accumulate_pred(self, net):
        self.add_eval_ops(net)
    
    def add_param_copy_operators(self, net):
        for param in self.params:
            param_copy_ref = self.model.metrics_schema[str(param.parameter)]
            net.Copy([param.parameter], param_copy_ref.field_blobs())
    
    def export_output_for_metrics(self):
        self._export_output_for_metrics = True
        export_name = self.name + '/output'
        self.model.add_metric_field(export_name, self.output_schema)
    
    def export_params_for_metrics(self):
        self._export_params_for_metrics = True
        for param in self.params:
            param_copy_ref = self.get_next_blob_reference(str(param).split('/')[-1] + '_copy')
            self.model.add_metric_field(str(param.parameter), param_copy_ref)


