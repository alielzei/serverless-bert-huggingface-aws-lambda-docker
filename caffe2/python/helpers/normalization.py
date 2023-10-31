from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import scope
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.proto import caffe2_pb2
from caffe2.python.modeling import initializers

def lrn(model, blob_in, blob_out, order='NCHW', use_cudnn=False, **kwargs):
    """LRN"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.normalization.lrn', "lrn(model, blob_in, blob_out, order='NCHW', use_cudnn=False, **kwargs)", {'scope': scope, 'caffe2_pb2': caffe2_pb2, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'order': order, 'use_cudnn': use_cudnn, 'kwargs': kwargs}, 1)

def softmax(model, blob_in, blob_out=None, use_cudnn=False, **kwargs):
    """Softmax."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.normalization.softmax', 'softmax(model, blob_in, blob_out=None, use_cudnn=False, **kwargs)', {'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'use_cudnn': use_cudnn, 'kwargs': kwargs}, 1)

def instance_norm(model, blob_in, blob_out, dim_in, order='NCHW', **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.normalization.instance_norm', "instance_norm(model, blob_in, blob_out, dim_in, order='NCHW', **kwargs)", {'ParameterTags': ParameterTags, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'dim_in': dim_in, 'order': order, 'kwargs': kwargs}, 1)

def spatial_bn(model, blob_in, blob_out, dim_in, init_scale=1.0, init_bias=0.0, ScaleInitializer=None, BiasInitializer=None, RunningMeanInitializer=None, RunningVarianceInitializer=None, order='NCHW', **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.normalization.spatial_bn', "spatial_bn(model, blob_in, blob_out, dim_in, init_scale=1.0, init_bias=0.0, ScaleInitializer=None, BiasInitializer=None, RunningMeanInitializer=None, RunningVarianceInitializer=None, order='NCHW', **kwargs)", {'initializers': initializers, 'ParameterTags': ParameterTags, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'dim_in': dim_in, 'init_scale': init_scale, 'init_bias': init_bias, 'ScaleInitializer': ScaleInitializer, 'BiasInitializer': BiasInitializer, 'RunningMeanInitializer': RunningMeanInitializer, 'RunningVarianceInitializer': RunningVarianceInitializer, 'order': order, 'kwargs': kwargs}, 1)

def spatial_gn(model, blob_in, blob_out, dim_in, init_scale=1.0, init_bias=0.0, ScaleInitializer=None, BiasInitializer=None, RunningMeanInitializer=None, RunningVarianceInitializer=None, order='NCHW', **kwargs):
    """
    Group normalizes the input, cf. https://arxiv.org/abs/1803.08494.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.normalization.spatial_gn', "spatial_gn(model, blob_in, blob_out, dim_in, init_scale=1.0, init_bias=0.0, ScaleInitializer=None, BiasInitializer=None, RunningMeanInitializer=None, RunningVarianceInitializer=None, order='NCHW', **kwargs)", {'initializers': initializers, 'ParameterTags': ParameterTags, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'dim_in': dim_in, 'init_scale': init_scale, 'init_bias': init_bias, 'ScaleInitializer': ScaleInitializer, 'BiasInitializer': BiasInitializer, 'RunningMeanInitializer': RunningMeanInitializer, 'RunningVarianceInitializer': RunningVarianceInitializer, 'order': order, 'kwargs': kwargs}, 1)

def layer_norm(model, blob_in, blob_out, dim_in, axis=1, epsilon=0.0001, initial_scale=1.0, initial_bias=0.0):
    """
    Layer normalizes the input, cf. https://arxiv.org/pdf/1607.06450.pdf.

    Args:
        blob_in: The input blob to layer normalize.
        blob_out: The layer normalized output blob.
        dim_in: The dimension of the scale and bias. For example, if blob_in is
            a 2D design matrix and axis is 1, this would be the number of
            columns.
        axis: (optional) The axis to normalize. Typically the feature axis.
            Defaults to 1.
        epsilon: (optional) A small value used for numerical stability in
            calculation. Defaults to 1e-4.
        initial_scale: (optional) The initial value for the learned scale
            parameter. Defaults to 1.0
        initial_bias: (optional) The initial value for the learned bias
            parameter of the layerwise standard deviation. Defaults to 0.0.

    Returns:
        A 3-tuple consisting of:
            - The layer normalized input blob.
            - The mean of the input blob across the given axis.
            - The standard deviation of the input blob acress the given axis.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.normalization.layer_norm', 'layer_norm(model, blob_in, blob_out, dim_in, axis=1, epsilon=0.0001, initial_scale=1.0, initial_bias=0.0)', {'initializers': initializers, 'ParameterTags': ParameterTags, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'dim_in': dim_in, 'axis': axis, 'epsilon': epsilon, 'initial_scale': initial_scale, 'initial_bias': initial_bias}, 3)

def moments_with_running_stats(model, blob_in, blob_out, dim_in, RunningMeanInitializer=None, RunningVarianceInitializer=None, order='NCHW', **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.normalization.moments_with_running_stats', "moments_with_running_stats(model, blob_in, blob_out, dim_in, RunningMeanInitializer=None, RunningVarianceInitializer=None, order='NCHW', **kwargs)", {'initializers': initializers, 'ParameterTags': ParameterTags, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'dim_in': dim_in, 'RunningMeanInitializer': RunningMeanInitializer, 'RunningVarianceInitializer': RunningVarianceInitializer, 'order': order, 'kwargs': kwargs}, 1)

