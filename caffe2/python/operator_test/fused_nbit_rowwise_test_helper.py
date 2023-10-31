from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

def param_search_greedy(x, bit_rate, n_bins=200, ratio=0.16):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.fused_nbit_rowwise_test_helper.param_search_greedy', 'param_search_greedy(x, bit_rate, n_bins=200, ratio=0.16)', {'np': np, '_compress_uniform_simplified': _compress_uniform_simplified, 'x': x, 'bit_rate': bit_rate, 'n_bins': n_bins, 'ratio': ratio}, 2)

def _compress_uniform_simplified(X, bit_rate, xmin, xmax, fp16_scale_bias=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.fused_nbit_rowwise_test_helper._compress_uniform_simplified', '_compress_uniform_simplified(X, bit_rate, xmin, xmax, fp16_scale_bias=True)', {'np': np, 'X': X, 'bit_rate': bit_rate, 'xmin': xmin, 'xmax': xmax, 'fp16_scale_bias': fp16_scale_bias}, 2)

