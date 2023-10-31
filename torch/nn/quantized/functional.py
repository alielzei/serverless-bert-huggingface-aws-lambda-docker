""" Functional interface (quantized)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
from torch._jit_internal import List as _List
from torch.nn.modules.utils import _pair, _triple

def avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    """
    Applies 2D average-pooling operation in :math:`kH 	imes kW` regions by step size
    :math:`sH 	imes sW` steps. The number of output features is equal to the number of
    input planes.

    .. note:: The input quantization parameters propagate to the output.

    See :class:`~torch.nn.quantized.AvgPool2d` for details and output shape.

    Args:
        input: quantized input tensor :math:`(	ext{minibatch} , 	ext{in\_channels} , iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a
          tuple `(kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
          tuple `(sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula
            to compute the output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation. Default: ``True``
        divisor_override: if specified, it will be used as divisor, otherwise
             size of the pooling region will be used. Default: None
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.avg_pool2d', 'avg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)', {'torch': torch, 'input': input, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'ceil_mode': ceil_mode, 'count_include_pad': count_include_pad, 'divisor_override': divisor_override}, 1)

def avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None):
    """
    Applies 3D average-pooling operation in :math:`kD \ times kH 	imes kW` regions by step size
    :math:`sD 	imes sH 	imes sW` steps. The number of output features is equal to the number of
    input planes.

    .. note:: The input quantization parameters propagate to the output.

    Args:
        input: quantized input tensor :math:`(	ext{minibatch} , 	ext{in\_channels} , iH , iW)`
        kernel_size: size of the pooling region. Can be a single number or a
          tuple `(kD, kH, kW)`
        stride: stride of the pooling operation. Can be a single number or a
          tuple `(sD, sH, sW)`. Default: :attr:`kernel_size`
        padding: implicit zero paddings on both sides of the input. Can be a
          single number or a tuple `(padD, padH, padW)`. Default: 0
        ceil_mode: when True, will use `ceil` instead of `floor` in the formula
            to compute the output shape. Default: ``False``
        count_include_pad: when True, will include the zero-padding in the
            averaging calculation. Default: ``True``
        divisor_override: if specified, it will be used as divisor, otherwise
             size of the pooling region will be used. Default: None
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.avg_pool3d', 'avg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)', {'torch': torch, 'input': input, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'ceil_mode': ceil_mode, 'count_include_pad': count_include_pad, 'divisor_override': divisor_override}, 1)

def adaptive_avg_pool2d(input, output_size):
    """
    Applies a 2D adaptive average pooling over a quantized input signal composed
    of several quantized input planes.

    .. note:: The input quantization paramteres propagate to the output.

    See :class:`~torch.nn.quantized.AdaptiveAvgPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
                     double-integer tuple)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.adaptive_avg_pool2d', 'adaptive_avg_pool2d(input, output_size)', {'torch': torch, 'input': input, 'output_size': output_size}, 1)

def conv2d(input, weight, bias, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', scale=1.0, zero_point=0, dtype=torch.quint8):
    """
    Applies a 2D convolution over a quantized 2D input composed of several input
    planes.

    See :class:`~torch.nn.quantized.Conv2d` for details and output shape.

    Args:
        input: quantized input tensor of shape :math:`(	ext{minibatch} , 	ext{in\_channels} , iH , iW)`
        weight: quantized filters of shape :math:`(	ext{out\_channels} , rac{	ext{in\_channels}}{	ext{groups}} , kH , kW)`
        bias: **non-quantized** bias tensor of shape :math:`(	ext{out\_channels})`. The tensor type must be `torch.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dH, dW)`. Default: 1
        groups: split input into groups, :math:`	ext{in\_channels}` should be divisible by the
          number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``

    Examples::

        >>> from torch.nn.quantized import functional as qF
        >>> filters = torch.randn(8, 4, 3, 3, dtype=torch.float)
        >>> inputs = torch.randn(1, 4, 5, 5, dtype=torch.float)
        >>> bias = torch.randn(8, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype_inputs = torch.quint8
        >>> dtype_filters = torch.qint8
        >>>
        >>> q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype_filters)
        >>> q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype_inputs)
        >>> qF.conv2d(q_inputs, q_filters, bias, padding=1, scale=scale, zero_point=zero_point)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.conv2d', "conv2d(input, weight, bias, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', scale=1.0, zero_point=0, dtype=torch.quint8)", {'torch': torch, '_pair': _pair, 'input': input, 'weight': weight, 'bias': bias, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups, 'padding_mode': padding_mode, 'scale': scale, 'zero_point': zero_point, 'dtype': dtype}, 1)

def conv3d(input, weight, bias, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', scale=1.0, zero_point=0, dtype=torch.quint8):
    """
    Applies a 3D convolution over a quantized 3D input composed of several input
    planes.

    See :class:`~torch.nn.quantized.Conv3d` for details and output shape.

    Args:
        input: quantized input tensor of shape
          :math:`(	ext{minibatch} , 	ext{in\_channels} , iD , iH , iW)`
        weight: quantized filters of shape
          :math:`(	ext{out\_channels} , rac{	ext{in\_channels}}{	ext{groups}} , kD , kH , kW)`
        bias: **non-quantized** bias tensor of shape
          :math:`(	ext{out\_channels})`. The tensor type must be `torch.float`.
        stride: the stride of the convolving kernel. Can be a single number or a
          tuple `(sD, sH, sW)`. Default: 1
        padding: implicit paddings on both sides of the input. Can be a
          single number or a tuple `(padD, padH, padW)`. Default: 0
        dilation: the spacing between kernel elements. Can be a single number or
          a tuple `(dD, dH, dW)`. Default: 1
        groups: split input into groups, :math:`	ext{in\_channels}` should be
          divisible by the number of groups. Default: 1
        padding_mode: the padding mode to use. Only "zeros" is supported for
          quantized convolution at the moment. Default: "zeros"
        scale: quantization scale for the output. Default: 1.0
        zero_point: quantization zero_point for the output. Default: 0
        dtype: quantization data type to use. Default: ``torch.quint8``

    Examples::

        >>> from torch.nn.quantized import functional as qF
        >>> filters = torch.randn(8, 4, 3, 3, 3, dtype=torch.float)
        >>> inputs = torch.randn(1, 4, 5, 5, 5, dtype=torch.float)
        >>> bias = torch.randn(8, dtype=torch.float)
        >>>
        >>> scale, zero_point = 1.0, 0
        >>> dtype_inputs = torch.quint8
        >>> dtype_filters = torch.qint8
        >>>
        >>> q_filters = torch.quantize_per_tensor(filters, scale, zero_point, dtype_filters)
        >>> q_inputs = torch.quantize_per_tensor(inputs, scale, zero_point, dtype_inputs)
        >>> qF.conv3d(q_inputs, q_filters, bias, padding=1, scale=scale, zero_point=zero_point)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.conv3d', "conv3d(input, weight, bias, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', scale=1.0, zero_point=0, dtype=torch.quint8)", {'torch': torch, '_triple': _triple, 'input': input, 'weight': weight, 'bias': bias, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups, 'padding_mode': padding_mode, 'scale': scale, 'zero_point': zero_point, 'dtype': dtype}, 1)

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    See :func:`torch.nn.functional.interpolate` for implementation details.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D/3D input is supported for quantized inputs

    .. note:: Only the following modes are supported for the quantized inputs:

        - `bilinear`
        - `nearest`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'bilinear'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'bilinear'``.
            Default: ``False``
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.interpolate', "interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)", {'torch': torch, 'input': input, 'size': size, 'scale_factor': scale_factor, 'mode': mode, 'align_corners': align_corners}, 1)

def linear(input, weight, bias=None, scale=None, zero_point=None):
    """
    Applies a linear transformation to the incoming quantized data:
    :math:`y = xA^T + b`.
    See :class:`~torch.nn.quantized.Linear`

    .. note::

      Current implementation packs weights on every call, which has penalty on performance.
      If you want to avoid the overhead, use :class:`~torch.nn.quantized.Linear`.

    Args:
      input (Tensor): Quantized input of type `torch.quint8`
      weight (Tensor): Quantized weight of type `torch.qint8`
      bias (Tensor): None or fp32 bias of type `torch.float`
      scale (double): output scale. If None, derived from the input scale
      zero_point (long): output zero point. If None, derived from the input zero_point

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.linear', 'linear(input, weight, bias=None, scale=None, zero_point=None)', {'torch': torch, 'input': input, 'weight': weight, 'bias': bias, 'scale': scale, 'zero_point': zero_point}, 1)

def max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """Applies a 2D max pooling over a quantized input signal composed of
    several quantized input planes.

    .. note:: The input quantization parameters are propagated to the output.

    See :class:`~torch.nn.quantized.MaxPool2d` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.max_pool2d', 'max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False)', {'torch': torch, '_List': _List, 'input': input, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'dilation': dilation, 'ceil_mode': ceil_mode, 'return_indices': return_indices}, 1)

def relu(input, inplace=False):
    """relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise.
    See :class:`~torch.nn.quantized.ReLU` for more details.

    Args:
        input: quantized input
        inplace: perform the computation inplace
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.relu', 'relu(input, inplace=False)', {'torch': torch, 'input': input, 'inplace': inplace}, 1)

def leaky_relu(input, negative_slope=0.01, inplace=False, scale=None, zero_point=None):
    """
    Quantized version of the.
    leaky_relu(input, negative_slope=0.01, inplace=False, scale, zero_point) -> Tensor

    Applies element-wise,
    :math:`	ext{LeakyReLU}(x) = \max(0, x) + 	ext{negative\_slope} * \min(0, x)`

    Args:
        input: Quaintized input
        negative_slope: The slope of the negative input
        inplace: Inplace modification of the input tensor
        scale, zero_point: Scale and zero point of thhe output tensor.

    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.leaky_relu', 'leaky_relu(input, negative_slope=0.01, inplace=False, scale=None, zero_point=None)', {'torch': torch, 'input': input, 'negative_slope': negative_slope, 'inplace': inplace, 'scale': scale, 'zero_point': zero_point}, 1)

def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    """
    hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor

    Applies the quantized HardTanh function element-wise, with scale and
    zero-point carried over from the input tensor. See :class:`~torch.nn.Hardtanh`
    for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.hardtanh', 'hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False)', {'torch': torch, 'input': input, 'min_val': min_val, 'max_val': max_val, 'inplace': inplace}, 1)

def elu(input, alpha=1.0, inplace=False, scale=None, zero_point=None):
    """
    Applies the quantized ELU function element-wise:

    .. math::
        	ext{ELU}(x) = \max(0,x) + \min(0, lpha * (\exp(x) - 1))

    Args:
        input: quantized input
        alpha: the :math:`lpha` value for the ELU formulation. Default: 1.0
        inplace: Inplace modification of the input tensor
        scale, zero_point: Scale and zero point of the output tensor.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.elu', 'elu(input, alpha=1.0, inplace=False, scale=None, zero_point=None)', {'torch': torch, 'input': input, 'alpha': alpha, 'inplace': inplace, 'scale': scale, 'zero_point': zero_point}, 1)

def clamp(input, min_, max_):
    """float(input, min_, max_) -> Tensor

    Applies the clamp function element-wise.
    See :class:`~torch.nn.quantized.clamp` for more details.

    Args:
        input: quantized input
        min_: minimum value for clamping
        max_: maximum value for clamping
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.functional.clamp', 'clamp(input, min_, max_)', {'torch': torch, 'input': input, 'min_': min_, 'max_': max_}, 1)

def upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """Upsamples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    .. warning::
        This function is deprecated in favor of
        :func:`torch.nn.quantized.functional.interpolate`.
        This is equivalent with ``nn.quantized.functional.interpolate(...)``.

    See :func:`torch.nn.functional.interpolate` for implementation details.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D input is supported for quantized inputs

    .. note:: Only the following modes are supported for the quantized inputs:

        - `bilinear`
        - `nearest`

    Args:
        input (Tensor): quantized input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to be an integer.
        mode (string): algorithm used for upsampling:
            ``'nearest'`` | ``'bilinear'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'bilinear'``.
            Default: ``False``

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`bilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.
    """
    warnings.warn('nn.quantized.functional.upsample is deprecated. Use nn.quantized.functional.interpolate instead.')
    return interpolate(input, size, scale_factor, mode, align_corners)

def upsample_bilinear(input, size=None, scale_factor=None):
    """Upsamples the input, using bilinear upsampling.

    .. warning::
        This function is deprecated in favor of
        :func:`torch.nn.quantized.functional.interpolate`.
        This is equivalent with
        ``nn.quantized.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D inputs are supported

    Args:
        input (Tensor): quantized input
        size (int or Tuple[int, int]): output spatial size.
        scale_factor (int or Tuple[int, int]): multiplier for spatial size
    """
    warnings.warn('nn.quantized.functional.upsample_bilinear is deprecated. Use nn.quantized.functional.interpolate instead.')
    return interpolate(input, size, scale_factor, mode='bilinear', align_corners=True)

def upsample_nearest(input, size=None, scale_factor=None):
    """Upsamples the input, using nearest neighbours' pixel values.

    .. warning::
        This function is deprecated in favor of
        :func:`torch.nn.quantized.functional.interpolate`.
        This is equivalent with ``nn.quantized.functional.interpolate(..., mode='nearest')``.

    .. note:: The input quantization parameters propagate to the output.

    .. note:: Only 2D inputs are supported

    Args:
        input (Tensor): quantized input
        size (int or Tuple[int, int] or Tuple[int, int, int]): output spatial
            size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.
    """
    warnings.warn('nn.quantized.functional.upsample_nearest is deprecated. Use nn.quantized.functional.interpolate instead.')
    return interpolate(input, size, scale_factor, mode='nearest')

