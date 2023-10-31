"""
This module converts objects into numpy array.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import six

def make_np(x):
    """
    Args:
      x: An instance of torch tensor or caffe blob name

    Returns:
        numpy.array: Numpy array
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._convert_np.make_np', 'make_np(x)', {'np': np, 'six': six, '_prepare_caffe2': _prepare_caffe2, 'torch': torch, '_prepare_pytorch': _prepare_pytorch, 'x': x}, 1)

def _prepare_pytorch(x):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._convert_np._prepare_pytorch', '_prepare_pytorch(x)', {'torch': torch, 'x': x}, 1)

def _prepare_caffe2(x):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._convert_np._prepare_caffe2', '_prepare_caffe2(x)', {'x': x}, 1)

