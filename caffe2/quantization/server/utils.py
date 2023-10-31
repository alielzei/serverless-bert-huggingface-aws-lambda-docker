from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import logging
from collections import defaultdict
import numpy as np
from caffe2.python import core, utils
from caffe2.python.fb import hardcode_scale_zp
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.pairwise', 'pairwise(iterable)', {'iterable': iterable}, 1)

def blob_uses(net, blob):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.blob_uses', 'blob_uses(net, blob)', {'net': net, 'blob': blob}, 1)

def fuse_first_bn(net, params, removed_tensors, begin_op_index):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.fuse_first_bn', 'fuse_first_bn(net, params, removed_tensors, begin_op_index)', {'copy': copy, 'blob_uses': blob_uses, 'logger': logger, 'np': np, 'net': net, 'params': params, 'removed_tensors': removed_tensors, 'begin_op_index': begin_op_index}, 4)

def fuse_bn(net, params, ignore_failure):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.fuse_bn', 'fuse_bn(net, params, ignore_failure)', {'fuse_first_bn': fuse_first_bn, 'net': net, 'params': params, 'ignore_failure': ignore_failure}, 3)

def fuse_first_scale(net, params, removed_tensors):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.fuse_first_scale', 'fuse_first_scale(net, params, removed_tensors)', {'copy': copy, 'pairwise': pairwise, 'net': net, 'params': params, 'removed_tensors': removed_tensors}, 3)

def fuse_scale(net, params, ignore_failure):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.fuse_scale', 'fuse_scale(net, params, ignore_failure)', {'fuse_first_scale': fuse_first_scale, 'net': net, 'params': params, 'ignore_failure': ignore_failure}, 3)

def fuse_first_relu(net, begin_op_index, ignore_op_with_output=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.fuse_first_relu', 'fuse_first_relu(net, begin_op_index, ignore_op_with_output=None)', {'copy': copy, 'blob_uses': blob_uses, 'logger': logger, 'net': net, 'begin_op_index': begin_op_index, 'ignore_op_with_output': ignore_op_with_output}, 2)

def fuse_relu(net, ignore_failure, ignore_op_with_output=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.fuse_relu', 'fuse_relu(net, ignore_failure, ignore_op_with_output=None)', {'fuse_first_relu': fuse_first_relu, 'net': net, 'ignore_failure': ignore_failure, 'ignore_op_with_output': ignore_op_with_output}, 1)

def last_producer(ops, blob):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.last_producer', 'last_producer(ops, blob)', {'ops': ops, 'blob': blob}, 1)

def swap_first_concat_relu(net, ignore_op_with_output=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.swap_first_concat_relu', 'swap_first_concat_relu(net, ignore_op_with_output=None)', {'copy': copy, 'pairwise': pairwise, 'last_producer': last_producer, 'net': net, 'ignore_op_with_output': ignore_op_with_output}, 1)

def swap_concat_relu(net, ignore_op_with_output=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.swap_concat_relu', 'swap_concat_relu(net, ignore_op_with_output=None)', {'swap_first_concat_relu': swap_first_concat_relu, 'net': net, 'ignore_op_with_output': ignore_op_with_output}, 1)

def add_version_to_conv_bias(net, init_net):
    """
    In architectures such as FPN (https://arxiv.org/abs/1612.03144), few Conv
    ops share the same weight and bias and are run at different scales of
    the input. Since 'bias_scale = input_scale * weight_scale', sharing the
    same bias blob among multiple Conv ops means that we need different bias
    scale for each of the ops. To achieve this, we just duplicate those bias
    blobs that are used by multiple Conv ops before performing int8 rewrite.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.add_version_to_conv_bias', 'add_version_to_conv_bias(net, init_net)', {'defaultdict': defaultdict, 'copy': copy, 'net': net, 'init_net': init_net}, 0)

def add_quantization_param_args_(op, q_param):
    op.arg.extend([utils.MakeArgument('Y_scale', q_param.scale), utils.MakeArgument('Y_zero_point', q_param.zero_point)])

def choose_quantization_params(tensor_min, tensor_max, preserve_sparsity=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.choose_quantization_params', 'choose_quantization_params(tensor_min, tensor_max, preserve_sparsity=False)', {'hardcode_scale_zp': hardcode_scale_zp, 'tensor_min': tensor_min, 'tensor_max': tensor_max, 'preserve_sparsity': preserve_sparsity}, 1)

def add_quantization_param_args(op, tensor, preserve_sparsity=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.add_quantization_param_args', 'add_quantization_param_args(op, tensor, preserve_sparsity=False)', {'choose_quantization_params': choose_quantization_params, 'add_quantization_param_args_': add_quantization_param_args_, 'op': op, 'tensor': tensor, 'preserve_sparsity': preserve_sparsity}, 1)

def create_int8_given_tensor_fill(tensor, out_blob_name, preserve_sparsity=False):
    """
    Create Int8GivenTensorFill op that quantizes the given tensor and outputs
    an Int8Tensor with out_blob_name.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.create_int8_given_tensor_fill', 'create_int8_given_tensor_fill(tensor, out_blob_name, preserve_sparsity=False)', {'core': core, 'add_quantization_param_args': add_quantization_param_args, 'np': np, 'utils': utils, 'tensor': tensor, 'out_blob_name': out_blob_name, 'preserve_sparsity': preserve_sparsity}, 2)

def create_int8_bias_tensor_fill(tensor, out_blob_name, x_q_param, w_q_param):
    """
    Similar to create_int8_given_tensor_fill, but for bias blobs to be stored
    as int32.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.utils.create_int8_bias_tensor_fill', 'create_int8_bias_tensor_fill(tensor, out_blob_name, x_q_param, w_q_param)', {'np': np, 'core': core, 'utils': utils, 'hardcode_scale_zp': hardcode_scale_zp, 'add_quantization_param_args_': add_quantization_param_args_, 'tensor': tensor, 'out_blob_name': out_blob_name, 'x_q_param': x_q_param, 'w_q_param': w_q_param}, 1)

