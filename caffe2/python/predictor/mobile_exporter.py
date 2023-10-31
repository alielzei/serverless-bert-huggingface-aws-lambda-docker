from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, utils
from caffe2.proto import caffe2_pb2
import numpy as np

def add_tensor(net, name, blob):
    """ Create an operator to store the tensor 'blob',
        run the operator to put the blob to workspace.
        uint8 is stored as an array of string with one element.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.predictor.mobile_exporter.add_tensor', 'add_tensor(net, name, blob)', {'np': np, 'core': core, 'utils': utils, 'net': net, 'name': name, 'blob': blob}, 0)

def Export(workspace, net, params):
    """Returns init_net and predict_net suitable for writing to disk
       and loading into a Predictor"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.mobile_exporter.Export', 'Export(workspace, net, params)', {'caffe2_pb2': caffe2_pb2, 'core': core, 'add_tensor': add_tensor, 'utils': utils, 'workspace': workspace, 'net': net, 'params': params}, 2)

