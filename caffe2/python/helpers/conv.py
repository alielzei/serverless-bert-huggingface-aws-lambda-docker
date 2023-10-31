from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

def _ConvBase(model, is_nd, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None, bias_init=None, WeightInitializer=None, BiasInitializer=None, group=1, transform_inputs=None, use_cudnn=False, order='NCHW', cudnn_exhaustive_search=False, ws_nbytes_limit=None, float16_compute=False, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.conv._ConvBase', "_ConvBase(model, is_nd, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None, bias_init=None, WeightInitializer=None, BiasInitializer=None, group=1, transform_inputs=None, use_cudnn=False, order='NCHW', cudnn_exhaustive_search=False, ws_nbytes_limit=None, float16_compute=False, **kwargs)", {'initializers': initializers, 'ParameterTags': ParameterTags, 'model': model, 'is_nd': is_nd, 'blob_in': blob_in, 'blob_out': blob_out, 'dim_in': dim_in, 'dim_out': dim_out, 'kernel': kernel, 'weight_init': weight_init, 'bias_init': bias_init, 'WeightInitializer': WeightInitializer, 'BiasInitializer': BiasInitializer, 'group': group, 'transform_inputs': transform_inputs, 'use_cudnn': use_cudnn, 'order': order, 'cudnn_exhaustive_search': cudnn_exhaustive_search, 'ws_nbytes_limit': ws_nbytes_limit, 'float16_compute': float16_compute, 'kwargs': kwargs}, 1)

def conv_nd(model, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None, bias_init=None, WeightInitializer=None, BiasInitializer=None, group=1, transform_inputs=None, order='NCHW', **kwargs):
    """N-dimensional convolution for inputs with NCHW storage order.
    """
    assert order == 'NCHW', 'ConvNd only supported for NCHW storage.'
    return _ConvBase(model, True, blob_in, blob_out, dim_in, dim_out, kernel, weight_init, bias_init, WeightInitializer, BiasInitializer, group, transform_inputs, order=order, **kwargs)

def conv(model, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None, bias_init=None, WeightInitializer=None, BiasInitializer=None, group=1, transform_inputs=None, **kwargs):
    """2-dimensional convolution.
    """
    return _ConvBase(model, False, blob_in, blob_out, dim_in, dim_out, kernel, weight_init, bias_init, WeightInitializer, BiasInitializer, group, transform_inputs, **kwargs)

def conv_transpose(model, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None, bias_init=None, use_cudnn=False, order='NCHW', cudnn_exhaustive_search=False, ws_nbytes_limit=None, **kwargs):
    """ConvTranspose.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.conv.conv_transpose', "conv_transpose(model, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None, bias_init=None, use_cudnn=False, order='NCHW', cudnn_exhaustive_search=False, ws_nbytes_limit=None, **kwargs)", {'core': core, 'ParameterTags': ParameterTags, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'dim_in': dim_in, 'dim_out': dim_out, 'kernel': kernel, 'weight_init': weight_init, 'bias_init': bias_init, 'use_cudnn': use_cudnn, 'order': order, 'cudnn_exhaustive_search': cudnn_exhaustive_search, 'ws_nbytes_limit': ws_nbytes_limit, 'kwargs': kwargs}, 1)

def group_conv(model, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None, bias_init=None, group=1, **kwargs):
    """Group Convolution.

    This is essentially the same as Conv with a group argument passed in.
    We specialize this for backward interface compatibility.
    """
    return conv(model, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=weight_init, bias_init=bias_init, group=group, **kwargs)

def group_conv_deprecated(model, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None, bias_init=None, group=1, use_cudnn=False, order='NCHW', cudnn_exhaustive_search=False, ws_nbytes_limit=None, **kwargs):
    """GroupConvolution's deprecated interface.

    This is used to simulate a group convolution via split and concat. You
    should always use the new group convolution in your new code.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.conv.group_conv_deprecated', "group_conv_deprecated(model, blob_in, blob_out, dim_in, dim_out, kernel, weight_init=None, bias_init=None, group=1, use_cudnn=False, order='NCHW', cudnn_exhaustive_search=False, ws_nbytes_limit=None, **kwargs)", {'core': core, 'ParameterTags': ParameterTags, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'dim_in': dim_in, 'dim_out': dim_out, 'kernel': kernel, 'weight_init': weight_init, 'bias_init': bias_init, 'group': group, 'use_cudnn': use_cudnn, 'order': order, 'cudnn_exhaustive_search': cudnn_exhaustive_search, 'ws_nbytes_limit': ws_nbytes_limit, 'kwargs': kwargs}, 1)

