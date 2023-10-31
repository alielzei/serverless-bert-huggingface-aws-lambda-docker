from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import numpy as np
from caffe2.python import utils, workspace
from hypothesis import assume

def check_quantized_results_close(outputs, ref=None, symmetric=False, atol_scale=0.53):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.dnnlowp_test_utils.check_quantized_results_close', 'check_quantized_results_close(outputs, ref=None, symmetric=False, atol_scale=0.53)', {'np': np, 'outputs': outputs, 'ref': ref, 'symmetric': symmetric, 'atol_scale': atol_scale}, 1)

def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.dnnlowp_test_utils.pairwise', 'pairwise(iterable)', {'iterable': iterable}, 1)

def avoid_vpmaddubsw_overflow_fc(batch_size, input_channels, output_channels, X, X_min, X_max, W, W_min, W_max):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.quantization.server.dnnlowp_test_utils.avoid_vpmaddubsw_overflow_fc', 'avoid_vpmaddubsw_overflow_fc(batch_size, input_channels, output_channels, X, X_min, X_max, W, W_min, W_max)', {'np': np, 'batch_size': batch_size, 'input_channels': input_channels, 'output_channels': output_channels, 'X': X, 'X_min': X_min, 'X_max': X_max, 'W': W, 'W_min': W_min, 'W_max': W_max}, 0)

def avoid_vpmaddubsw_overflow(strides, pads, kernels, dilations, sizes, input_channels, output_channels, batch_size, X, X_min, X_max, W, W_min, W_max):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.quantization.server.dnnlowp_test_utils.avoid_vpmaddubsw_overflow', 'avoid_vpmaddubsw_overflow(strides, pads, kernels, dilations, sizes, input_channels, output_channels, batch_size, X, X_min, X_max, W, W_min, W_max)', {'np': np, 'pairwise': pairwise, 'strides': strides, 'pads': pads, 'kernels': kernels, 'dilations': dilations, 'sizes': sizes, 'input_channels': input_channels, 'output_channels': output_channels, 'batch_size': batch_size, 'X': X, 'X_min': X_min, 'X_max': X_max, 'W': W, 'W_min': W_min, 'W_max': W_max}, 0)

def generate_convnd_inputs(strides, pads, kernels, dilations, sizes, group, input_channels_per_group, output_channels_per_group, batch_size, order, groupwise_quantization=False, preserve_activation_sparsity=False, preserve_weight_sparsity=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.quantization.server.dnnlowp_test_utils.generate_convnd_inputs', 'generate_convnd_inputs(strides, pads, kernels, dilations, sizes, group, input_channels_per_group, output_channels_per_group, batch_size, order, groupwise_quantization=False, preserve_activation_sparsity=False, preserve_weight_sparsity=False)', {'assume': assume, 'np': np, 'avoid_vpmaddubsw_overflow': avoid_vpmaddubsw_overflow, 'utils': utils, 'strides': strides, 'pads': pads, 'kernels': kernels, 'dilations': dilations, 'sizes': sizes, 'group': group, 'input_channels_per_group': input_channels_per_group, 'output_channels_per_group': output_channels_per_group, 'batch_size': batch_size, 'order': order, 'groupwise_quantization': groupwise_quantization, 'preserve_activation_sparsity': preserve_activation_sparsity, 'preserve_weight_sparsity': preserve_weight_sparsity}, 3)

def generate_conv_inputs(stride, pad, kernel, dilation, size, group, input_channels_per_group, output_channels_per_group, batch_size, order, groupwise_quantization=False, preserve_activation_sparsity=False, preserve_weight_sparsity=False):
    return generate_convnd_inputs((stride, ) * 2, (pad, ) * 2, (kernel, ) * 2, (dilation, ) * 2, (size, ) * 2, group, input_channels_per_group, output_channels_per_group, batch_size, order, groupwise_quantization, preserve_activation_sparsity, preserve_weight_sparsity)

def run_conv_or_fc(test_case, init_net, net, X, W, b, op_type, engine, order, gc, outputs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.quantization.server.dnnlowp_test_utils.run_conv_or_fc', 'run_conv_or_fc(test_case, init_net, net, X, W, b, op_type, engine, order, gc, outputs)', {'collections': collections, 'workspace': workspace, 'test_case': test_case, 'init_net': init_net, 'net': net, 'X': X, 'W': W, 'b': b, 'op_type': op_type, 'engine': engine, 'order': order, 'gc': gc, 'outputs': outputs}, 0)

