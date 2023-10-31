"""
ONNXIFI a Caffe2 net
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import caffe2.python._import_c_extension as C
import numpy as np

def onnxifi_caffe2_net(pred_net, input_shapes, max_batch_size=1, max_seq_size=1, debug=False, use_onnx=True, merge_fp32_inputs_into_fp16=False, adjust_batch=True, black_list=None, weight_names=None):
    """
    Transform the caffe2_net by collapsing ONNXIFI-runnable nodes into Onnxifi c2 ops
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.onnx.onnxifi.onnxifi_caffe2_net', 'onnxifi_caffe2_net(pred_net, input_shapes, max_batch_size=1, max_seq_size=1, debug=False, use_onnx=True, merge_fp32_inputs_into_fp16=False, adjust_batch=True, black_list=None, weight_names=None)', {'C': C, 'caffe2_pb2': caffe2_pb2, 'pred_net': pred_net, 'input_shapes': input_shapes, 'max_batch_size': max_batch_size, 'max_seq_size': max_seq_size, 'debug': debug, 'use_onnx': use_onnx, 'merge_fp32_inputs_into_fp16': merge_fp32_inputs_into_fp16, 'adjust_batch': adjust_batch, 'black_list': black_list, 'weight_names': weight_names}, 1)

