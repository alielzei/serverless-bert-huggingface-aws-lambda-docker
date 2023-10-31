from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core

def prelu(model, blob_in, blob_out, num_channels=1, slope_init=None, **kwargs):
    """PRelu"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.nonlinearity.prelu', 'prelu(model, blob_in, blob_out, num_channels=1, slope_init=None, **kwargs)', {'core': core, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'num_channels': num_channels, 'slope_init': slope_init, 'kwargs': kwargs}, 1)

def relu(model, blob_in, blob_out, use_cudnn=False, order='NCHW', **kwargs):
    """Relu."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.nonlinearity.relu', "relu(model, blob_in, blob_out, use_cudnn=False, order='NCHW', **kwargs)", {'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'use_cudnn': use_cudnn, 'order': order, 'kwargs': kwargs}, 1)

def tanh(model, blob_in, blob_out, use_cudnn=False, order='NCHW', **kwargs):
    """Tanh."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.nonlinearity.tanh', "tanh(model, blob_in, blob_out, use_cudnn=False, order='NCHW', **kwargs)", {'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'use_cudnn': use_cudnn, 'order': order, 'kwargs': kwargs}, 1)

