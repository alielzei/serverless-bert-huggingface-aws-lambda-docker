from collections import defaultdict
import numpy as np
import torch
import hypothesis
from hypothesis import assume
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp
from hypothesis.strategies import SearchStrategy
from torch.testing._internal.common_quantized import _calculate_dynamic_qparams, _calculate_dynamic_per_channel_qparams
_ALL_QINT_TYPES = (torch.quint8, torch.qint8, torch.qint32)
_ENFORCED_ZERO_POINT = defaultdict(lambda: None, {torch.quint8: None, torch.qint8: None, torch.qint32: 0})

def _get_valid_min_max(qparams):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.hypothesis_utils._get_valid_min_max', '_get_valid_min_max(qparams)', {'torch': torch, 'np': np, 'qparams': qparams}, 2)

def _floats_wrapper(*args, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.hypothesis_utils._floats_wrapper', '_floats_wrapper(*args, **kwargs)', {'hypothesis': hypothesis, 'st': st, 'args': args, 'kwargs': kwargs}, 1)

def floats(*args, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.hypothesis_utils.floats', 'floats(*args, **kwargs)', {'_floats_wrapper': _floats_wrapper, 'args': args, 'kwargs': kwargs}, 1)
'Hypothesis filter to avoid overflows with quantized tensors.\n\nArgs:\n    tensor: Tensor of floats to filter\n    qparams: Quantization parameters as returned by the `qparams`.\n\nReturns:\n    True\n\nRaises:\n    hypothesis.UnsatisfiedAssumption\n\nNote: This filter is slow. Use it only when filtering of the test cases is\n      absolutely necessary!\n'

def assume_not_overflowing(tensor, qparams):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.hypothesis_utils.assume_not_overflowing', 'assume_not_overflowing(tensor, qparams)', {'_get_valid_min_max': _get_valid_min_max, 'assume': assume, 'tensor': tensor, 'qparams': qparams}, 1)
'Strategy for generating the quantization parameters.\n\nArgs:\n    dtypes: quantized data types to sample from.\n    scale_min / scale_max: Min and max scales. If None, set to 1e-3 / 1e3.\n    zero_point_min / zero_point_max: Min and max for the zero point. If None,\n        set to the minimum and maximum of the quantized data type.\n        Note: The min and max are only valid if the zero_point is not enforced\n              by the data type itself.\n\nGenerates:\n    scale: Sampled scale.\n    zero_point: Sampled zero point.\n    quantized_type: Sampled quantized type.\n'

@st.composite
def qparams(dtypes=None, scale_min=None, scale_max=None, zero_point_min=None, zero_point_max=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.hypothesis_utils.qparams', 'qparams(dtypes=None, scale_min=None, scale_max=None, zero_point_min=None, zero_point_max=None)', {'_ALL_QINT_TYPES': _ALL_QINT_TYPES, 'draw': draw, 'torch': torch, '_ENFORCED_ZERO_POINT': _ENFORCED_ZERO_POINT, 'floats': floats, 'st': st, 'dtypes': dtypes, 'scale_min': scale_min, 'scale_max': scale_max, 'zero_point_min': zero_point_min, 'zero_point_max': zero_point_max}, 3)
'Strategy to create different shapes.\nArgs:\n    min_dims / max_dims: minimum and maximum rank.\n    min_side / max_side: minimum and maximum dimensions per rank.\n\nGenerates:\n    Possible shapes for a tensor, constrained to the rank and dimensionality.\n\nExample:\n    # Generates 3D and 4D tensors.\n    @given(Q = qtensor(shapes=array_shapes(min_dims=3, max_dims=4))\n    some_test(self, Q):...\n'

@st.composite
def array_shapes(min_dims=1, max_dims=None, min_side=1, max_side=None):
    """Return a strategy for array shapes (tuples of int >= 1)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.hypothesis_utils.array_shapes', 'array_shapes(min_dims=1, max_dims=None, min_side=1, max_side=None)', {'draw': draw, 'st': st, 'min_dims': min_dims, 'max_dims': max_dims, 'min_side': min_side, 'max_side': max_side}, 1)
'Strategy for generating test cases for tensors.\nThe resulting tensor is in float32 format.\n\nArgs:\n    shapes: Shapes under test for the tensor. Could be either a hypothesis\n            strategy, or an iterable of different shapes to sample from.\n    elements: Elements to generate from for the returned data type.\n              If None, the strategy resolves to float within range [-1e6, 1e6].\n    qparams: Instance of the qparams strategy. This is used to filter the tensor\n             such that the overflow would not happen.\n\nGenerates:\n    X: Tensor of type float32. Note that NaN and +/-inf is not included.\n    qparams: (If `qparams` arg is set) Quantization parameters for X.\n        The returned parameters are `(scale, zero_point, quantization_type)`.\n        (If `qparams` arg is None), returns None.\n'

@st.composite
def tensor(shapes=None, elements=None, qparams=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.hypothesis_utils.tensor', 'tensor(shapes=None, elements=None, qparams=None)', {'SearchStrategy': SearchStrategy, 'draw': draw, 'floats': floats, 'stnp': stnp, 'np': np, 'assume': assume, '_get_valid_min_max': _get_valid_min_max, '_calculate_dynamic_qparams': _calculate_dynamic_qparams, '_ENFORCED_ZERO_POINT': _ENFORCED_ZERO_POINT, 'st': st, 'shapes': shapes, 'elements': elements, 'qparams': qparams}, 2)

@st.composite
def per_channel_tensor(shapes=None, elements=None, qparams=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.hypothesis_utils.per_channel_tensor', 'per_channel_tensor(shapes=None, elements=None, qparams=None)', {'SearchStrategy': SearchStrategy, 'draw': draw, 'floats': floats, 'stnp': stnp, 'np': np, 'assume': assume, '_get_valid_min_max': _get_valid_min_max, '_calculate_dynamic_per_channel_qparams': _calculate_dynamic_per_channel_qparams, '_ENFORCED_ZERO_POINT': _ENFORCED_ZERO_POINT, 'st': st, 'shapes': shapes, 'elements': elements, 'qparams': qparams}, 2)
'Strategy for generating test cases for tensors used in Conv.\nThe resulting tensors is in float32 format.\n\nArgs:\n    spatial_dim: Spatial Dim for feature maps.\n    batch_size_range: Range to generate `batch_size`.\n                      Must be tuple of `(min, max)`.\n    input_channels_per_group_range:\n        Range to generate `input_channels_per_group`.\n        Must be tuple of `(min, max)`.\n    output_channels_per_group_range:\n        Range to generate `output_channels_per_group`.\n        Must be tuple of `(min, max)`.\n    feature_map_range: Range to generate feature map size for each spatial_dim.\n                       Must be tuple of `(min, max)`.\n    kernel_range: Range to generate kernel size for each spatial_dim. Must be\n                  tuple of `(min, max)`.\n    max_groups: Maximum number of groups to generate.\n    elements: Elements to generate from for the returned data type.\n              If None, the strategy resolves to float within range [-1e6, 1e6].\n    qparams: Strategy for quantization parameters. for X, w, and b.\n             Could be either a single strategy (used for all) or a list of\n             three strategies for X, w, b.\nGenerates:\n    (X, W, b, g): Tensors of type `float32` of the following drawen shapes:\n        X: (`batch_size, input_channels, H, W`)\n        W: (`output_channels, input_channels_per_group) + kernel_shape\n        b: `(output_channels,)`\n        groups: Number of groups the input is divided into\nNote: X, W, b are tuples of (Tensor, qparams), where qparams could be either\n      None or (scale, zero_point, quantized_type)\n\n\nExample:\n    @given(tensor_conv(\n        spatial_dim=2,\n        batch_size_range=(1, 3),\n        input_channels_per_group_range=(1, 7),\n        output_channels_per_group_range=(1, 7),\n        feature_map_range=(6, 12),\n        kernel_range=(3, 5),\n        max_groups=4,\n        elements=st.floats(-1.0, 1.0),\n        qparams=qparams()\n    ))\n'

@st.composite
def tensor_conv(spatial_dim=2, batch_size_range=(1, 4), input_channels_per_group_range=(3, 7), output_channels_per_group_range=(3, 7), feature_map_range=(6, 12), kernel_range=(3, 7), max_groups=1, elements=None, qparams=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.hypothesis_utils.tensor_conv', 'tensor_conv(spatial_dim=2, batch_size_range=(1, 4), input_channels_per_group_range=(3, 7), output_channels_per_group_range=(3, 7), feature_map_range=(6, 12), kernel_range=(3, 7), max_groups=1, elements=None, qparams=None)', {'draw': draw, 'tensor': tensor, 'st': st, 'spatial_dim': spatial_dim, 'batch_size_range': batch_size_range, 'input_channels_per_group_range': input_channels_per_group_range, 'output_channels_per_group_range': output_channels_per_group_range, 'feature_map_range': feature_map_range, 'kernel_range': kernel_range, 'max_groups': max_groups, 'elements': elements, 'qparams': qparams}, 4)
hypothesis_version = hypothesis.version.__version_info__
current_settings = settings._profiles[settings._current_profile].__dict__
current_settings['deadline'] = None
if (hypothesis_version >= (3, 16, 0) and hypothesis_version < (5, 0, 0)):
    current_settings['timeout'] = hypothesis.unlimited

def assert_deadline_disabled():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.hypothesis_utils.assert_deadline_disabled', 'assert_deadline_disabled()', {'hypothesis_version': hypothesis_version, 'hypothesis': hypothesis, 'settings': settings}, 0)

