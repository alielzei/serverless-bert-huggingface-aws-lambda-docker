from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, scope
from caffe2.proto import caffe2_pb2

def _get_weights(model, namescope=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.train._get_weights', '_get_weights(model, namescope=None)', {'scope': scope, 'model': model, 'namescope': namescope}, 1)

def iter(model, blob_out, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.train.iter', 'iter(model, blob_out, **kwargs)', {'core': core, 'caffe2_pb2': caffe2_pb2, 'model': model, 'blob_out': blob_out, 'kwargs': kwargs}, 1)

def accuracy(model, blob_in, blob_out, **kwargs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.helpers.train.accuracy', 'accuracy(model, blob_in, blob_out, **kwargs)', {'scope': scope, 'caffe2_pb2': caffe2_pb2, 'core': core, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'kwargs': kwargs}, 0)

def add_weight_decay(model, weight_decay):
    """Adds a decay to weights in the model.

    This is a form of L2 regularization.

    Args:
        weight_decay: strength of the regularization
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.train.add_weight_decay', 'add_weight_decay(model, weight_decay)', {'_get_weights': _get_weights, 'model': model, 'weight_decay': weight_decay}, 1)

