"""
TensorRT related transformation
Note that ONNX-TRT enforce an NCHW input!
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python.onnx.helper import c2_native_run_net, c2_native_run_op
from caffe2.python import core, workspace
import caffe2.python.onnx.frontend as c2_front
import caffe2.python._import_c_extension as C
import numpy as np

def _dim_values_to_list(dim_values):
    return [x.dim_value for x in dim_values]

def _get_output_shapes(output_value_infos):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.trt.transform._get_output_shapes', '_get_output_shapes(output_value_infos)', {'_dim_values_to_list': _dim_values_to_list, 'output_value_infos': output_value_infos}, 1)

def check_gpu_():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.trt.transform.check_gpu_', 'check_gpu_()', {'C': C}, 0)

def convert_onnx_model_to_trt_op(onnx_model, max_batch_size=64, max_workspace_size=2 * 1024 * 1024, verbosity=1, debug_builder=False):
    """
    Convert the whole ONNX model to a TensorRT C2 op
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.trt.transform.convert_onnx_model_to_trt_op', 'convert_onnx_model_to_trt_op(onnx_model, max_batch_size=64, max_workspace_size=2 * 1024 * 1024, verbosity=1, debug_builder=False)', {'check_gpu_': check_gpu_, 'C': C, '_get_output_shapes': _get_output_shapes, 'caffe2_pb2': caffe2_pb2, 'onnx_model': onnx_model, 'max_batch_size': max_batch_size, 'max_workspace_size': max_workspace_size, 'verbosity': verbosity, 'debug_builder': debug_builder}, 1)

def _infer_shapes(pred_net, inputs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.trt.transform._infer_shapes', '_infer_shapes(pred_net, inputs)', {'workspace': workspace, 'pred_net': pred_net, 'inputs': inputs}, 1)

def transform_caffe2_net(pred_net, input_shapes, populate_shapes=False, max_batch_size=64, max_workspace_size=2 * 1024 * 1024, verbosity=1, debug_builder=False, build_serializable_op=True):
    """
    Transform the caffe2_net by collapsing TRT-runnable nodes into trt c2 ops
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.trt.transform.transform_caffe2_net', 'transform_caffe2_net(pred_net, input_shapes, populate_shapes=False, max_batch_size=64, max_workspace_size=2 * 1024 * 1024, verbosity=1, debug_builder=False, build_serializable_op=True)', {'check_gpu_': check_gpu_, 'np': np, '_infer_shapes': _infer_shapes, 'C': C, 'caffe2_pb2': caffe2_pb2, 'pred_net': pred_net, 'input_shapes': input_shapes, 'populate_shapes': populate_shapes, 'max_batch_size': max_batch_size, 'max_workspace_size': max_workspace_size, 'verbosity': verbosity, 'debug_builder': debug_builder, 'build_serializable_op': build_serializable_op}, 1)

