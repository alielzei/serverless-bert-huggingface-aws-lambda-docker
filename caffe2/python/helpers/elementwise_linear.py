from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from caffe2.python.modeling.parameter_info import ParameterTags

def _elementwise_linear(model, op_call, blob_in, blob_out, dim, weight_init=None, bias_init=None, **kwargs):
    """Elementwise_Linear"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.elementwise_linear._elementwise_linear', '_elementwise_linear(model, op_call, blob_in, blob_out, dim, weight_init=None, bias_init=None, **kwargs)', {'core': core, 'ParameterTags': ParameterTags, 'model': model, 'op_call': op_call, 'blob_in': blob_in, 'blob_out': blob_out, 'dim': dim, 'weight_init': weight_init, 'bias_init': bias_init, 'kwargs': kwargs}, 1)

def elementwise_linear(model, *args, **kwargs):
    return _elementwise_linear(model, model.net.ElementwiseLinear, *args, **kwargs)

