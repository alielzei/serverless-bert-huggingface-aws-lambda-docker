"""Importing this file includes common utility methods for checking quantized
tensors and modules.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import torch
from contextlib import contextmanager
'Computes the output shape given convolution parameters.'

def _conv_output_shape(input_size, kernel_size, padding, stride, dilation, output_padding=0):
    return np.floor((input_size + 2 * padding - kernel_size - (kernel_size - 1) * (dilation - 1)) / stride) + 2 * output_padding + 1

def _quantize(x, scale, zero_point, qmin=None, qmax=None, dtype=np.uint8):
    """Quantizes a numpy array."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_quantized._quantize', '_quantize(x, scale, zero_point, qmin=None, qmax=None, dtype=np.uint8)', {'np': np, 'x': x, 'scale': scale, 'zero_point': zero_point, 'qmin': qmin, 'qmax': qmax, 'dtype': dtype}, 1)

def _dequantize(qx, scale, zero_point):
    """Dequantizes a numpy array."""
    x = (qx.astype(np.float) - zero_point) * scale
    return x

def _requantize(x, multiplier, zero_point, qmin=0, qmax=255, qtype=np.uint8):
    """Requantizes a numpy array, i.e., intermediate int32 or int16 values are
    converted back to given type"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_quantized._requantize', '_requantize(x, multiplier, zero_point, qmin=0, qmax=255, qtype=np.uint8)', {'np': np, 'x': x, 'multiplier': multiplier, 'zero_point': zero_point, 'qmin': qmin, 'qmax': qmax, 'qtype': qtype}, 1)

def _calculate_dynamic_qparams(X, dtype, reduce_range=False):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_quantized._calculate_dynamic_qparams', '_calculate_dynamic_qparams(X, dtype, reduce_range=False)', {'torch': torch, 'np': np, 'X': X, 'dtype': dtype, 'reduce_range': reduce_range}, 1)

def _calculate_dynamic_per_channel_qparams(X, dtype):
    """Calculate the dynamic quantization parameters (scale, zero_point)
    according to the min and max element of the tensor"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_quantized._calculate_dynamic_per_channel_qparams', '_calculate_dynamic_per_channel_qparams(X, dtype)', {'torch': torch, 'np': np, 'X': X, 'dtype': dtype}, 2)

@contextmanager
def override_quantized_engine(qengine):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.common_quantized.override_quantized_engine', 'override_quantized_engine(qengine)', {'torch': torch, 'contextmanager': contextmanager, 'qengine': qengine}, 0)

