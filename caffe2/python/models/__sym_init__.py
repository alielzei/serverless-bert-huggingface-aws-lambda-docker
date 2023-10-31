from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from caffe2.proto import caffe2_pb2

def _parseFile(filename):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.__sym_init__._parseFile', '_parseFile(filename)', {'caffe2_pb2': caffe2_pb2, 'os': os, '__file__': __file__, 'filename': filename}, 1)
init_net = _parseFile('init_net.pb')
predict_net = _parseFile('predict_net.pb')

