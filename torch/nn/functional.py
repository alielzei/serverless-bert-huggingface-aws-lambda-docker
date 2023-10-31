"""Functional interface"""

from __future__ import division
import warnings
import math
import torch
from torch._C import _infer_size, _add_docstr
from . import _reduction as _Reduction
from .modules import utils
from .modules.utils import _single, _pair, _triple, _list_with_default
from . import grad
from torch import _VF
from .._jit_internal import boolean_dispatch, List, Optional, _overload
from .._overrides import has_torch_function, handle_torch_function
Tensor = torch.Tensor
conv1d = _add_docstr(torch.conv1d, '\nconv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor\n\nApplies a 1D convolution over an input signal composed of several input\nplanes.\n\nSee :class:`~torch.nn.Conv1d` for details and output shape.\n\n.. include:: cudnn_deterministic.rst\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iW)`\n    weight: filters of shape :math:`(\\text{out\\_channels} , \\frac{\\text{in\\_channels}}{\\text{groups}} , kW)`\n    bias: optional bias of shape :math:`(\\text{out\\_channels})`. Default: ``None``\n    stride: the stride of the convolving kernel. Can be a single number or\n      a one-element tuple `(sW,)`. Default: 1\n    padding: implicit paddings on both sides of the input. Can be a\n      single number or a one-element tuple `(padW,)`. Default: 0\n    dilation: the spacing between kernel elements. Can be a single number or\n      a one-element tuple `(dW,)`. Default: 1\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by\n      the number of groups. Default: 1\n\nExamples::\n\n    >>> filters = torch.randn(33, 16, 3)\n    >>> inputs = torch.randn(20, 16, 50)\n    >>> F.conv1d(inputs, filters)\n')
conv2d = _add_docstr(torch.conv2d, '\nconv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor\n\nApplies a 2D convolution over an input image composed of several input\nplanes.\n\nSee :class:`~torch.nn.Conv2d` for details and output shape.\n\n.. include:: cudnn_deterministic.rst\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iH , iW)`\n    weight: filters of shape :math:`(\\text{out\\_channels} , \\frac{\\text{in\\_channels}}{\\text{groups}} , kH , kW)`\n    bias: optional bias tensor of shape :math:`(\\text{out\\_channels})`. Default: ``None``\n    stride: the stride of the convolving kernel. Can be a single number or a\n      tuple `(sH, sW)`. Default: 1\n    padding: implicit paddings on both sides of the input. Can be a\n      single number or a tuple `(padH, padW)`. Default: 0\n    dilation: the spacing between kernel elements. Can be a single number or\n      a tuple `(dH, dW)`. Default: 1\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by the\n      number of groups. Default: 1\n\nExamples::\n\n    >>> # With square kernels and equal stride\n    >>> filters = torch.randn(8,4,3,3)\n    >>> inputs = torch.randn(1,4,5,5)\n    >>> F.conv2d(inputs, filters, padding=1)\n')
conv3d = _add_docstr(torch.conv3d, '\nconv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1) -> Tensor\n\nApplies a 3D convolution over an input image composed of several input\nplanes.\n\nSee :class:`~torch.nn.Conv3d` for details and output shape.\n\n.. include:: cudnn_deterministic.rst\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iT , iH , iW)`\n    weight: filters of shape :math:`(\\text{out\\_channels} , \\frac{\\text{in\\_channels}}{\\text{groups}} , kT , kH , kW)`\n    bias: optional bias tensor of shape :math:`(\\text{out\\_channels})`. Default: None\n    stride: the stride of the convolving kernel. Can be a single number or a\n      tuple `(sT, sH, sW)`. Default: 1\n    padding: implicit paddings on both sides of the input. Can be a\n      single number or a tuple `(padT, padH, padW)`. Default: 0\n    dilation: the spacing between kernel elements. Can be a single number or\n      a tuple `(dT, dH, dW)`. Default: 1\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by\n      the number of groups. Default: 1\n\nExamples::\n\n    >>> filters = torch.randn(33, 16, 3, 3, 3)\n    >>> inputs = torch.randn(20, 16, 50, 10, 20)\n    >>> F.conv3d(inputs, filters)\n')
conv_transpose1d = _add_docstr(torch.conv_transpose1d, '\nconv_transpose1d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor\n\nApplies a 1D transposed convolution operator over an input signal\ncomposed of several input planes, sometimes also called "deconvolution".\n\nSee :class:`~torch.nn.ConvTranspose1d` for details and output shape.\n\n.. include:: cudnn_deterministic.rst\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iW)`\n    weight: filters of shape :math:`(\\text{in\\_channels} , \\frac{\\text{out\\_channels}}{\\text{groups}} , kW)`\n    bias: optional bias of shape :math:`(\\text{out\\_channels})`. Default: None\n    stride: the stride of the convolving kernel. Can be a single number or a\n      tuple ``(sW,)``. Default: 1\n    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both\n      sides of each dimension in the input. Can be a single number or a tuple\n      ``(padW,)``. Default: 0\n    output_padding: additional size added to one side of each dimension in the\n      output shape. Can be a single number or a tuple ``(out_padW)``. Default: 0\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by the\n      number of groups. Default: 1\n    dilation: the spacing between kernel elements. Can be a single number or\n      a tuple ``(dW,)``. Default: 1\n\nExamples::\n\n    >>> inputs = torch.randn(20, 16, 50)\n    >>> weights = torch.randn(16, 33, 5)\n    >>> F.conv_transpose1d(inputs, weights)\n')
conv_transpose2d = _add_docstr(torch.conv_transpose2d, '\nconv_transpose2d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor\n\nApplies a 2D transposed convolution operator over an input image\ncomposed of several input planes, sometimes also called "deconvolution".\n\nSee :class:`~torch.nn.ConvTranspose2d` for details and output shape.\n\n.. include:: cudnn_deterministic.rst\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iH , iW)`\n    weight: filters of shape :math:`(\\text{in\\_channels} , \\frac{\\text{out\\_channels}}{\\text{groups}} , kH , kW)`\n    bias: optional bias of shape :math:`(\\text{out\\_channels})`. Default: None\n    stride: the stride of the convolving kernel. Can be a single number or a\n      tuple ``(sH, sW)``. Default: 1\n    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both\n      sides of each dimension in the input. Can be a single number or a tuple\n      ``(padH, padW)``. Default: 0\n    output_padding: additional size added to one side of each dimension in the\n      output shape. Can be a single number or a tuple ``(out_padH, out_padW)``.\n      Default: 0\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by the\n      number of groups. Default: 1\n    dilation: the spacing between kernel elements. Can be a single number or\n      a tuple ``(dH, dW)``. Default: 1\n\nExamples::\n\n    >>> # With square kernels and equal stride\n    >>> inputs = torch.randn(1, 4, 5, 5)\n    >>> weights = torch.randn(4, 8, 3, 3)\n    >>> F.conv_transpose2d(inputs, weights, padding=1)\n')
conv_transpose3d = _add_docstr(torch.conv_transpose3d, '\nconv_transpose3d(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1) -> Tensor\n\nApplies a 3D transposed convolution operator over an input image\ncomposed of several input planes, sometimes also called "deconvolution"\n\nSee :class:`~torch.nn.ConvTranspose3d` for details and output shape.\n\n.. include:: cudnn_deterministic.rst\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iT , iH , iW)`\n    weight: filters of shape :math:`(\\text{in\\_channels} , \\frac{\\text{out\\_channels}}{\\text{groups}} , kT , kH , kW)`\n    bias: optional bias of shape :math:`(\\text{out\\_channels})`. Default: None\n    stride: the stride of the convolving kernel. Can be a single number or a\n      tuple ``(sT, sH, sW)``. Default: 1\n    padding: ``dilation * (kernel_size - 1) - padding`` zero-padding will be added to both\n      sides of each dimension in the input. Can be a single number or a tuple\n      ``(padT, padH, padW)``. Default: 0\n    output_padding: additional size added to one side of each dimension in the\n      output shape. Can be a single number or a tuple\n      ``(out_padT, out_padH, out_padW)``. Default: 0\n    groups: split input into groups, :math:`\\text{in\\_channels}` should be divisible by the\n      number of groups. Default: 1\n    dilation: the spacing between kernel elements. Can be a single number or\n      a tuple `(dT, dH, dW)`. Default: 1\n\nExamples::\n\n    >>> inputs = torch.randn(20, 16, 50, 10, 20)\n    >>> weights = torch.randn(16, 33, 3, 3, 3)\n    >>> F.conv_transpose3d(inputs, weights)\n')
conv_tbc = _add_docstr(torch.conv_tbc, '\nApplies a 1-dimensional sequence convolution over an input sequence.\nInput and output dimensions are (Time, Batch, Channels) - hence TBC.\n\nArgs:\n    input: input tensor of shape :math:`(\\text{sequence length} \\times batch \\times \\text{in\\_channels})`\n    weight: filter of shape (:math:`\\text{kernel width} \\times \\text{in\\_channels} \\times \\text{out\\_channels}`)\n    bias: bias of shape (:math:`\\text{out\\_channels}`)\n    pad: number of timesteps to pad. Default: 0\n')
avg_pool1d = _add_docstr(torch.avg_pool1d, '\navg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True) -> Tensor\n\nApplies a 1D average pooling over an input signal composed of several\ninput planes.\n\nSee :class:`~torch.nn.AvgPool1d` for details and output shape.\n\nArgs:\n    input: input tensor of shape :math:`(\\text{minibatch} , \\text{in\\_channels} , iW)`\n    kernel_size: the size of the window. Can be a single number or a\n      tuple `(kW,)`\n    stride: the stride of the window. Can be a single number or a tuple\n      `(sW,)`. Default: :attr:`kernel_size`\n    padding: implicit zero paddings on both sides of the input. Can be a\n      single number or a tuple `(padW,)`. Default: 0\n    ceil_mode: when True, will use `ceil` instead of `floor` to compute the\n        output shape. Default: ``False``\n    count_include_pad: when True, will include the zero-padding in the\n        averaging calculation. Default: ``True``\n\nExamples::\n\n    >>> # pool of square window of size=3, stride=2\n    >>> input = torch.tensor([[[1, 2, 3, 4, 5, 6, 7]]], dtype=torch.float32)\n    >>> F.avg_pool1d(input, kernel_size=3, stride=2)\n    tensor([[[ 2.,  4.,  6.]]])\n\n')
avg_pool2d = _add_docstr(torch._C._nn.avg_pool2d, '\navg_pool2d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor\n\nApplies 2D average-pooling operation in :math:`kH \\times kW` regions by step size\n:math:`sH \\times sW` steps. The number of output features is equal to the number of\ninput planes.\n\nSee :class:`~torch.nn.AvgPool2d` for details and output shape.\n\nArgs:\n    input: input tensor :math:`(\\text{minibatch} , \\text{in\\_channels} , iH , iW)`\n    kernel_size: size of the pooling region. Can be a single number or a\n      tuple `(kH, kW)`\n    stride: stride of the pooling operation. Can be a single number or a\n      tuple `(sH, sW)`. Default: :attr:`kernel_size`\n    padding: implicit zero paddings on both sides of the input. Can be a\n      single number or a tuple `(padH, padW)`. Default: 0\n    ceil_mode: when True, will use `ceil` instead of `floor` in the formula\n        to compute the output shape. Default: ``False``\n    count_include_pad: when True, will include the zero-padding in the\n        averaging calculation. Default: ``True``\n    divisor_override: if specified, it will be used as divisor, otherwise\n         size of the pooling region will be used. Default: None\n')
avg_pool3d = _add_docstr(torch._C._nn.avg_pool3d, '\navg_pool3d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None) -> Tensor\n\nApplies 3D average-pooling operation in :math:`kT \\times kH \\times kW` regions by step\nsize :math:`sT \\times sH \\times sW` steps. The number of output features is equal to\n:math:`\\lfloor\\frac{\\text{input planes}}{sT}\\rfloor`.\n\nSee :class:`~torch.nn.AvgPool3d` for details and output shape.\n\nArgs:\n    input: input tensor :math:`(\\text{minibatch} , \\text{in\\_channels} , iT \\times iH , iW)`\n    kernel_size: size of the pooling region. Can be a single number or a\n      tuple `(kT, kH, kW)`\n    stride: stride of the pooling operation. Can be a single number or a\n      tuple `(sT, sH, sW)`. Default: :attr:`kernel_size`\n    padding: implicit zero paddings on both sides of the input. Can be a\n      single number or a tuple `(padT, padH, padW)`, Default: 0\n    ceil_mode: when True, will use `ceil` instead of `floor` in the formula\n        to compute the output shape\n    count_include_pad: when True, will include the zero-padding in the\n        averaging calculation\n    divisor_override: if specified, it will be used as divisor, otherwise\n        size of the pooling region will be used. Default: None\n')

def fractional_max_pool2d_with_indices(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None):
    """Applies 2D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kH 	imes kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k 	imes k`)
                     or a tuple `(kH, kW)`
        output_size: the target output size of the image of the form :math:`oH 	imes oW`.
                     Can be a tuple `(oH, oW)` or a single number :math:`oH` for a square image :math:`oH 	imes oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool2d`.

    Examples::
        >>> input = torch.randn(20, 16, 50, 32)
        >>> # pool of square window of size=3, and target output size 13x12
        >>> F.fractional_max_pool2d(input, 3, output_size=(13, 12))
        >>> # pool of square window and target output size being half of input image size
        >>> F.fractional_max_pool2d(input, 3, output_ratio=(0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(fractional_max_pool2d_with_indices, (input, ), input, kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices, _random_samples=_random_samples)
    if (output_size is None and output_ratio is None):
        raise ValueError('fractional_max_pool2d requires specifying either an output_size or an output_ratio')
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _pair(output_ratio)
        output_size = [int(input.size(2) * _output_ratio[0]), int(input.size(3) * _output_ratio[1])]
    if _random_samples is None:
        _random_samples = torch.rand(input.size(0), input.size(1), 2, dtype=input.dtype, device=input.device)
    return torch._C._nn.fractional_max_pool2d(input, kernel_size, output_size, _random_samples)

def _fractional_max_pool2d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None):
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(fractional_max_pool2d, (input, ), input, kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices, _random_samples=_random_samples)
    return fractional_max_pool2d_with_indices(input, kernel_size, output_size, output_ratio, return_indices, _random_samples)[0]
fractional_max_pool2d = boolean_dispatch(arg_name='return_indices', arg_index=4, default=False, if_true=fractional_max_pool2d_with_indices, if_false=_fractional_max_pool2d, module_name=__name__, func_name='fractional_max_pool2d')

def fractional_max_pool3d_with_indices(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None):
    """Applies 3D fractional max pooling over an input signal composed of several input planes.

    Fractional MaxPooling is described in detail in the paper `Fractional MaxPooling`_ by Ben Graham

    The max-pooling operation is applied in :math:`kT 	imes kH 	imes kW` regions by a stochastic
    step size determined by the target output size.
    The number of output features is equal to the number of input planes.

    Args:
        kernel_size: the size of the window to take a max over.
                     Can be a single number :math:`k` (for a square kernel of :math:`k 	imes k 	imes k`)
                     or a tuple `(kT, kH, kW)`
        output_size: the target output size of the form :math:`oT 	imes oH 	imes oW`.
                     Can be a tuple `(oT, oH, oW)` or a single number :math:`oH` for a cubic output
                      :math:`oH 	imes oH 	imes oH`
        output_ratio: If one wants to have an output size as a ratio of the input size, this option can be given.
                      This has to be a number or tuple in the range (0, 1)
        return_indices: if ``True``, will return the indices along with the outputs.
                        Useful to pass to :func:`~torch.nn.functional.max_unpool3d`.

    Examples::
        >>> input = torch.randn(20, 16, 50, 32, 16)
        >>> # pool of cubic window of size=3, and target output size 13x12x11
        >>> F.fractional_max_pool3d(input, 3, output_size=(13, 12, 11))
        >>> # pool of cubic window and target output size being half of input size
        >>> F.fractional_max_pool3d(input, 3, output_ratio=(0.5, 0.5, 0.5))

    .. _Fractional MaxPooling:
        http://arxiv.org/abs/1412.6071
    """
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(fractional_max_pool3d_with_indices, (input, ), input, kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices, _random_samples=_random_samples)
    if (output_size is None and output_ratio is None):
        raise ValueError('fractional_max_pool3d requires specifying either an output_size or an output_ratio')
    if output_size is None:
        assert output_ratio is not None
        _output_ratio = _triple(output_ratio)
        output_size = [int(input.size(2) * _output_ratio[0]), int(input.size(3) * _output_ratio[1]), int(input.size(4) * _output_ratio[2])]
    if _random_samples is None:
        _random_samples = torch.rand(input.size(0), input.size(1), 3, dtype=input.dtype, device=input.device)
    return torch._C._nn.fractional_max_pool3d(input, kernel_size, output_size, _random_samples)

def _fractional_max_pool3d(input, kernel_size, output_size=None, output_ratio=None, return_indices=False, _random_samples=None):
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(fractional_max_pool3d, (input, ), input, kernel_size, output_size=output_size, output_ratio=output_ratio, return_indices=return_indices, _random_samples=_random_samples)
    return fractional_max_pool3d_with_indices(input, kernel_size, output_size, output_ratio, return_indices, _random_samples)[0]
fractional_max_pool3d = boolean_dispatch(arg_name='return_indices', arg_index=4, default=False, if_true=fractional_max_pool3d_with_indices, if_false=_fractional_max_pool3d, module_name=__name__, func_name='fractional_max_pool3d')

def max_pool1d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """Applies a 1D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool1d` for details.
    """
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(max_pool1d_with_indices, (input, ), input, kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool1d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)

def _max_pool1d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(max_pool1d, (input, ), input, kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool1d(input, kernel_size, stride, padding, dilation, ceil_mode)
max_pool1d = boolean_dispatch(arg_name='return_indices', arg_index=6, default=False, if_true=max_pool1d_with_indices, if_false=_max_pool1d, module_name=__name__, func_name='max_pool1d')

def max_pool2d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """Applies a 2D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool2d` for details.
    """
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(max_pool2d_with_indices, (input, ), input, kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch._C._nn.max_pool2d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)

def _max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(max_pool2d, (input, ), input, kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
max_pool2d = boolean_dispatch(arg_name='return_indices', arg_index=6, default=False, if_true=max_pool2d_with_indices, if_false=_max_pool2d, module_name=__name__, func_name='max_pool2d')

def max_pool3d_with_indices(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    """Applies a 3D max pooling over an input signal composed of several input
    planes.

    See :class:`~torch.nn.MaxPool3d` for details.
    """
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(max_pool3d_with_indices, (input, ), input, kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch._C._nn.max_pool3d_with_indices(input, kernel_size, stride, padding, dilation, ceil_mode)

def _max_pool3d(input, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False, return_indices=False):
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(max_pool3d, (input, ), input, kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=return_indices)
    if stride is None:
        stride = torch.jit.annotate(List[int], [])
    return torch.max_pool3d(input, kernel_size, stride, padding, dilation, ceil_mode)
max_pool3d = boolean_dispatch(arg_name='return_indices', arg_index=6, default=False, if_true=max_pool3d_with_indices, if_false=_max_pool3d, module_name=__name__, func_name='max_pool3d')

def _unpool_output_size(input, kernel_size, stride, padding, output_size):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional._unpool_output_size', '_unpool_output_size(input, kernel_size, stride, padding, output_size)', {'torch': torch, 'List': List, 'input': input, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}, 1)

def max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    """Computes a partial inverse of :class:`MaxPool1d`.

    See :class:`~torch.nn.MaxUnpool1d` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.max_unpool1d', 'max_unpool1d(input, indices, kernel_size, stride=None, padding=0, output_size=None)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'max_unpool1d': max_unpool1d, '_single': _single, '_unpool_output_size': _unpool_output_size, 'input': input, 'indices': indices, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}, 1)

def max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    """Computes a partial inverse of :class:`MaxPool2d`.

    See :class:`~torch.nn.MaxUnpool2d` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.max_unpool2d', 'max_unpool2d(input, indices, kernel_size, stride=None, padding=0, output_size=None)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'max_unpool2d': max_unpool2d, '_pair': _pair, '_unpool_output_size': _unpool_output_size, 'input': input, 'indices': indices, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}, 1)

def max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None):
    """Computes a partial inverse of :class:`MaxPool3d`.

    See :class:`~torch.nn.MaxUnpool3d` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.max_unpool3d', 'max_unpool3d(input, indices, kernel_size, stride=None, padding=0, output_size=None)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'max_unpool3d': max_unpool3d, '_triple': _triple, '_unpool_output_size': _unpool_output_size, 'input': input, 'indices': indices, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}, 1)

def lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    """Applies a 2D power-average pooling over an input signal composed of
    several input planes. If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool2d` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.lp_pool2d', 'lp_pool2d(input, norm_type, kernel_size, stride=None, ceil_mode=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'lp_pool2d': lp_pool2d, 'utils': utils, 'avg_pool2d': avg_pool2d, 'relu': relu, 'input': input, 'norm_type': norm_type, 'kernel_size': kernel_size, 'stride': stride, 'ceil_mode': ceil_mode}, 1)

def lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False):
    """Applies a 1D power-average pooling over an input signal composed of
    several input planes. If the sum of all inputs to the power of `p` is
    zero, the gradient is set to zero as well.

    See :class:`~torch.nn.LPPool1d` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.lp_pool1d', 'lp_pool1d(input, norm_type, kernel_size, stride=None, ceil_mode=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'lp_pool1d': lp_pool1d, 'avg_pool1d': avg_pool1d, 'relu': relu, 'input': input, 'norm_type': norm_type, 'kernel_size': kernel_size, 'stride': stride, 'ceil_mode': ceil_mode}, 1)

def adaptive_max_pool1d_with_indices(input, output_size, return_indices=False):
    """Applies a 1D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool1d` for details and output shape.

    Args:
        output_size: the target output size (single integer)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(adaptive_max_pool1d_with_indices, (input, ), input, output_size, return_indices=return_indices)
    return torch.adaptive_max_pool1d(input, output_size)

def _adaptive_max_pool1d(input, output_size, return_indices=False):
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(adaptive_max_pool1d, (input, ), input, output_size, return_indices=return_indices)
    return adaptive_max_pool1d_with_indices(input, output_size)[0]
adaptive_max_pool1d = boolean_dispatch(arg_name='return_indices', arg_index=2, default=False, if_true=adaptive_max_pool1d_with_indices, if_false=_adaptive_max_pool1d, module_name=__name__, func_name='adaptive_max_pool1d')

def adaptive_max_pool2d_with_indices(input, output_size, return_indices=False):
    """Applies a 2D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(adaptive_max_pool2d_with_indices, (input, ), input, output_size, return_indices=return_indices)
    output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_max_pool2d(input, output_size)

def _adaptive_max_pool2d(input, output_size, return_indices=False):
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(adaptive_max_pool2d, (input, ), input, output_size, return_indices=return_indices)
    return adaptive_max_pool2d_with_indices(input, output_size)[0]
adaptive_max_pool2d = boolean_dispatch(arg_name='return_indices', arg_index=2, default=False, if_true=adaptive_max_pool2d_with_indices, if_false=_adaptive_max_pool2d, module_name=__name__, func_name='adaptive_max_pool2d')

def adaptive_max_pool3d_with_indices(input, output_size, return_indices=False):
    """Applies a 3D adaptive max pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveMaxPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
        return_indices: whether to return pooling indices. Default: ``False``
    """
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(adaptive_max_pool3d_with_indices, (input, ), input, output_size, return_indices=return_indices)
    output_size = _list_with_default(output_size, input.size())
    return torch._C._nn.adaptive_max_pool3d(input, output_size)

def _adaptive_max_pool3d(input, output_size, return_indices=False):
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(adaptive_max_pool3d, (input, ), input, output_size, return_indices=return_indices)
    return adaptive_max_pool3d_with_indices(input, output_size)[0]
adaptive_max_pool3d = boolean_dispatch(arg_name='return_indices', arg_index=2, default=False, if_true=adaptive_max_pool3d_with_indices, if_false=_adaptive_max_pool3d, module_name=__name__, func_name='adaptive_max_pool3d')
adaptive_avg_pool1d = _add_docstr(torch.adaptive_avg_pool1d, '\nadaptive_avg_pool1d(input, output_size) -> Tensor\n\nApplies a 1D adaptive average pooling over an input signal composed of\nseveral input planes.\n\nSee :class:`~torch.nn.AdaptiveAvgPool1d` for details and output shape.\n\nArgs:\n    output_size: the target output size (single integer)\n')

def adaptive_avg_pool2d(input, output_size):
    """
    Applies a 2D adaptive average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool2d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            double-integer tuple)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.adaptive_avg_pool2d', 'adaptive_avg_pool2d(input, output_size)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'adaptive_avg_pool2d': adaptive_avg_pool2d, '_list_with_default': _list_with_default, 'input': input, 'output_size': output_size}, 1)

def adaptive_avg_pool3d(input, output_size):
    """
    Applies a 3D adaptive average pooling over an input signal composed of
    several input planes.

    See :class:`~torch.nn.AdaptiveAvgPool3d` for details and output shape.

    Args:
        output_size: the target output size (single integer or
            triple-integer tuple)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.adaptive_avg_pool3d', 'adaptive_avg_pool3d(input, output_size)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'adaptive_avg_pool3d': adaptive_avg_pool3d, '_list_with_default': _list_with_default, 'input': input, 'output_size': output_size}, 1)

def dropout(input, p=0.5, training=True, inplace=False):
    """
    During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution.

    See :class:`~torch.nn.Dropout` for details.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.dropout', 'dropout(input, p=0.5, training=True, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'dropout': dropout, '_VF': _VF, 'input': input, 'p': p, 'training': training, 'inplace': inplace}, 1)

def alpha_dropout(input, p=0.5, training=False, inplace=False):
    """Applies alpha dropout to the input.

    See :class:`~torch.nn.AlphaDropout` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.alpha_dropout', 'alpha_dropout(input, p=0.5, training=False, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'alpha_dropout': alpha_dropout, '_VF': _VF, 'input': input, 'p': p, 'training': training, 'inplace': inplace}, 1)

def dropout2d(input, p=0.5, training=True, inplace=False):
    """
    Randomly zero out entire channels (a channel is a 2D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 2D tensor :math:`	ext{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout2d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.dropout2d', 'dropout2d(input, p=0.5, training=True, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'dropout2d': dropout2d, '_VF': _VF, 'input': input, 'p': p, 'training': training, 'inplace': inplace}, 1)

def dropout3d(input, p=0.5, training=True, inplace=False):
    """
    Randomly zero out entire channels (a channel is a 3D feature map,
    e.g., the :math:`j`-th channel of the :math:`i`-th sample in the
    batched input is a 3D tensor :math:`	ext{input}[i, j]`) of the input tensor).
    Each channel will be zeroed out independently on every forward call with
    probability :attr:`p` using samples from a Bernoulli distribution.

    See :class:`~torch.nn.Dropout3d` for details.

    Args:
        p: probability of a channel to be zeroed. Default: 0.5
        training: apply dropout if is ``True``. Default: ``True``
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.dropout3d', 'dropout3d(input, p=0.5, training=True, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'dropout3d': dropout3d, '_VF': _VF, 'input': input, 'p': p, 'training': training, 'inplace': inplace}, 1)

def feature_alpha_dropout(input, p=0.5, training=False, inplace=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.feature_alpha_dropout', 'feature_alpha_dropout(input, p=0.5, training=False, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'feature_alpha_dropout': feature_alpha_dropout, '_VF': _VF, 'input': input, 'p': p, 'training': training, 'inplace': inplace}, 1)

def threshold(input, threshold, value, inplace=False):
    """Thresholds each element of the input Tensor.

    See :class:`~torch.nn.Threshold` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.threshold', 'threshold(input, threshold, value, inplace=False)', {'_VF': _VF, 'input': input, 'threshold': threshold, 'value': value, 'inplace': inplace}, 1)
threshold_ = _add_docstr(_VF.threshold_, '\nthreshold_(input, threshold, value) -> Tensor\n\nIn-place version of :func:`~threshold`.\n')

def relu(input, inplace=False):
    """relu(input, inplace=False) -> Tensor

    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.relu', 'relu(input, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'relu': relu, 'input': input, 'inplace': inplace}, 1)
relu_ = _add_docstr(torch.relu_, '\nrelu_(input) -> Tensor\n\nIn-place version of :func:`~relu`.\n')

def glu(input, dim=-1):
    """
    glu(input, dim=-1) -> Tensor

    The gated linear unit. Computes:

    .. math ::
        	ext{GLU}(a, b) = a \otimes \sigma(b)

    where `input` is split in half along `dim` to form `a` and `b`, :math:`\sigma`
    is the sigmoid function and :math:`\otimes` is the element-wise product between matrices.

    See `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_.

    Args:
        input (Tensor): input tensor
        dim (int): dimension on which to split the input. Default: -1
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.glu', 'glu(input, dim=-1)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'glu': glu, 'input': input, 'dim': dim}, 1)

def hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False):
    """
    hardtanh(input, min_val=-1., max_val=1., inplace=False) -> Tensor

    Applies the HardTanh function element-wise. See :class:`~torch.nn.Hardtanh` for more
    details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.hardtanh', 'hardtanh(input, min_val=-1.0, max_val=1.0, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'hardtanh': hardtanh, 'input': input, 'min_val': min_val, 'max_val': max_val, 'inplace': inplace}, 1)
hardtanh_ = _add_docstr(torch._C._nn.hardtanh_, '\nhardtanh_(input, min_val=-1., max_val=1.) -> Tensor\n\nIn-place version of :func:`~hardtanh`.\n')

def relu6(input, inplace=False):
    """relu6(input, inplace=False) -> Tensor

    Applies the element-wise function :math:`	ext{ReLU6}(x) = \min(\max(0,x), 6)`.

    See :class:`~torch.nn.ReLU6` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.relu6', 'relu6(input, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'relu6': relu6, 'hardtanh': hardtanh, 'input': input, 'inplace': inplace}, 1)

def elu(input, alpha=1.0, inplace=False):
    """Applies element-wise,
    :math:`	ext{ELU}(x) = \max(0,x) + \min(0, lpha * (\exp(x) - 1))`.

    See :class:`~torch.nn.ELU` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.elu', 'elu(input, alpha=1.0, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'elu': elu, 'input': input, 'alpha': alpha, 'inplace': inplace}, 1)
elu_ = _add_docstr(torch._C._nn.elu_, '\nelu_(input, alpha=1.) -> Tensor\n\nIn-place version of :func:`~elu`.\n')

def selu(input, inplace=False):
    """selu(input, inplace=False) -> Tensor

    Applies element-wise,
    :math:`	ext{SELU}(x) = scale * (\max(0,x) + \min(0, lpha * (\exp(x) - 1)))`,
    with :math:`lpha=1.6732632423543772848170429916717` and
    :math:`scale=1.0507009873554804934193349852946`.

    See :class:`~torch.nn.SELU` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.selu', 'selu(input, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'selu': selu, 'input': input, 'inplace': inplace}, 1)
selu_ = _add_docstr(torch.selu_, '\nselu_(input) -> Tensor\n\nIn-place version of :func:`~selu`.\n')

def celu(input, alpha=1.0, inplace=False):
    """celu(input, alpha=1., inplace=False) -> Tensor

    Applies element-wise,
    :math:`	ext{CELU}(x) = \max(0,x) + \min(0, lpha * (\exp(x/lpha) - 1))`.

    See :class:`~torch.nn.CELU` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.celu', 'celu(input, alpha=1.0, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'celu': celu, 'input': input, 'alpha': alpha, 'inplace': inplace}, 1)
celu_ = _add_docstr(torch.celu_, '\ncelu_(input, alpha=1.) -> Tensor\n\nIn-place version of :func:`~celu`.\n')

def leaky_relu(input, negative_slope=0.01, inplace=False):
    """
    leaky_relu(input, negative_slope=0.01, inplace=False) -> Tensor

    Applies element-wise,
    :math:`	ext{LeakyReLU}(x) = \max(0, x) + 	ext{negative\_slope} * \min(0, x)`

    See :class:`~torch.nn.LeakyReLU` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.leaky_relu', 'leaky_relu(input, negative_slope=0.01, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'leaky_relu': leaky_relu, 'input': input, 'negative_slope': negative_slope, 'inplace': inplace}, 1)
leaky_relu_ = _add_docstr(torch._C._nn.leaky_relu_, '\nleaky_relu_(input, negative_slope=0.01) -> Tensor\n\nIn-place version of :func:`~leaky_relu`.\n')

def prelu(input, weight):
    """prelu(input, weight) -> Tensor

    Applies element-wise the function
    :math:`	ext{PReLU}(x) = \max(0,x) + 	ext{weight} * \min(0,x)` where weight is a
    learnable parameter.

    See :class:`~torch.nn.PReLU` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.prelu', 'prelu(input, weight)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'prelu': prelu, 'input': input, 'weight': weight}, 1)

def rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False):
    """rrelu(input, lower=1./8, upper=1./3, training=False, inplace=False) -> Tensor

    Randomized leaky ReLU.

    See :class:`~torch.nn.RReLU` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.rrelu', 'rrelu(input, lower=1.0 / 8, upper=1.0 / 3, training=False, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'rrelu': rrelu, 'input': input, 'lower': lower, 'upper': upper, 'training': training, 'inplace': inplace}, 1)
rrelu_ = _add_docstr(torch.rrelu_, '\nrrelu_(input, lower=1./8, upper=1./3, training=False) -> Tensor\n\nIn-place version of :func:`~rrelu`.\n')
logsigmoid = _add_docstr(torch._C._nn.log_sigmoid, '\nlogsigmoid(input) -> Tensor\n\nApplies element-wise :math:`\\text{LogSigmoid}(x_i) = \\log \\left(\\frac{1}{1 + \\exp(-x_i)}\\right)`\n\nSee :class:`~torch.nn.LogSigmoid` for more details.\n')

def gelu(input):
    """gelu(input) -> Tensor

    Applies element-wise the function
    :math:`	ext{GELU}(x) = x * \Phi(x)`

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    See `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.gelu', 'gelu(input)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'gelu': gelu, 'input': input}, 1)

def hardshrink(input, lambd=0.5):
    """
    hardshrink(input, lambd=0.5) -> Tensor

    Applies the hard shrinkage function element-wise

    See :class:`~torch.nn.Hardshrink` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.hardshrink', 'hardshrink(input, lambd=0.5)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'hardshrink': hardshrink, 'input': input, 'lambd': lambd}, 1)

def tanhshrink(input):
    """tanhshrink(input) -> Tensor

    Applies element-wise, :math:`	ext{Tanhshrink}(x) = x - 	ext{Tanh}(x)`

    See :class:`~torch.nn.Tanhshrink` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.tanhshrink', 'tanhshrink(input)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'tanhshrink': tanhshrink, 'input': input}, 1)

def softsign(input):
    """softsign(input) -> Tensor

    Applies element-wise, the function :math:`	ext{SoftSign}(x) = rac{x}{1 + |x|}`

    See :class:`~torch.nn.Softsign` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.softsign', 'softsign(input)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'softsign': softsign, 'input': input}, 1)
softplus = _add_docstr(torch._C._nn.softplus, '\nsoftplus(input, beta=1, threshold=20) -> Tensor\n\nApplies element-wise, the function :math:`\\text{Softplus}(x) = \\frac{1}{\\beta} * \\log(1 + \\exp(\\beta * x))`.\n\nFor numerical stability the implementation reverts to the linear function\nwhen :math:`input \\times \\beta > threshold`.\n\nSee :class:`~torch.nn.Softplus` for more details.\n')

def _get_softmax_dim(name, ndim, stacklevel):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional._get_softmax_dim', '_get_softmax_dim(name, ndim, stacklevel)', {'warnings': warnings, 'name': name, 'ndim': ndim, 'stacklevel': stacklevel}, 1)

def softmin(input, dim=None, _stacklevel=3, dtype=None):
    """Applies a softmin function.

    Note that :math:`	ext{Softmin}(x) = 	ext{Softmax}(-x)`. See softmax definition for mathematical formula.

    See :class:`~torch.nn.Softmin` for more details.

    Arguments:
        input (Tensor): input
        dim (int): A dimension along which softmin will be computed (so every slice
            along dim will sum to 1).
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.softmin', 'softmin(input, dim=None, _stacklevel=3, dtype=None)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'softmin': softmin, '_get_softmax_dim': _get_softmax_dim, 'input': input, 'dim': dim, '_stacklevel': _stacklevel, 'dtype': dtype}, 1)

def softmax(input, dim=None, _stacklevel=3, dtype=None):
    """Applies a softmax function.

    Softmax is defined as:

    :math:`	ext{Softmax}(x_{i}) = rac{exp(x_i)}{\sum_j exp(x_j)}`

    It is applied to all slices along dim, and will re-scale them so that the elements
    lie in the range `[0, 1]` and sum to 1.

    See :class:`~torch.nn.Softmax` for more details.

    Arguments:
        input (Tensor): input
        dim (int): A dimension along which softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.

    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.softmax', 'softmax(input, dim=None, _stacklevel=3, dtype=None)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'softmax': softmax, '_get_softmax_dim': _get_softmax_dim, 'input': input, 'dim': dim, '_stacklevel': _stacklevel, 'dtype': dtype}, 1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    """
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.

    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.

    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.

    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)

    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)

    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.gumbel_softmax', 'gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'gumbel_softmax': gumbel_softmax, 'warnings': warnings, 'logits': logits, 'tau': tau, 'hard': hard, 'eps': eps, 'dim': dim}, 1)

def log_softmax(input, dim=None, _stacklevel=3, dtype=None):
    """Applies a softmax followed by a logarithm.

    While mathematically equivalent to log(softmax(x)), doing these two
    operations separately is slower, and numerically unstable. This function
    uses an alternative formulation to compute the output and gradient correctly.

    See :class:`~torch.nn.LogSoftmax` for more details.

    Arguments:
        input (Tensor): input
        dim (int): A dimension along which log_softmax will be computed.
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
          If specified, the input tensor is casted to :attr:`dtype` before the operation
          is performed. This is useful for preventing data type overflows. Default: None.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.log_softmax', 'log_softmax(input, dim=None, _stacklevel=3, dtype=None)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'log_softmax': log_softmax, '_get_softmax_dim': _get_softmax_dim, 'input': input, 'dim': dim, '_stacklevel': _stacklevel, 'dtype': dtype}, 1)
softshrink = _add_docstr(torch._C._nn.softshrink, '\nsoftshrink(input, lambd=0.5) -> Tensor\n\nApplies the soft shrinkage function elementwise\n\nSee :class:`~torch.nn.Softshrink` for more details.\n')

def tanh(input):
    """tanh(input) -> Tensor

    Applies element-wise,
    :math:`	ext{Tanh}(x) = 	anh(x) = rac{\exp(x) - \exp(-x)}{\exp(x) + \exp(-x)}`

    See :class:`~torch.nn.Tanh` for more details.
    """
    warnings.warn('nn.functional.tanh is deprecated. Use torch.tanh instead.')
    return input.tanh()

def sigmoid(input):
    """sigmoid(input) -> Tensor

    Applies the element-wise function :math:`	ext{Sigmoid}(x) = rac{1}{1 + \exp(-x)}`

    See :class:`~torch.nn.Sigmoid` for more details.
    """
    warnings.warn('nn.functional.sigmoid is deprecated. Use torch.sigmoid instead.')
    return input.sigmoid()

def hardsigmoid(input, inplace=False):
    """hardsigmoid(input) -> Tensor

    Applies the element-wise function :math:`	ext{Hardsigmoid}(x) = rac{ReLU6(x + 3)}{6}`

    Args:
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    See :class:`~torch.nn.Hardsigmoid` for more details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.hardsigmoid', 'hardsigmoid(input, inplace=False)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'hardsigmoid': hardsigmoid, 'input': input, 'inplace': inplace}, 1)

def linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

    Shape:

        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.linear', 'linear(input, weight, bias=None)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'linear': linear, 'input': input, 'weight': weight, 'bias': bias}, 1)

def bilinear(input1, input2, weight, bias=None):
    """
    Applies a bilinear transformation to the incoming data:
    :math:`y = x_1 A x_2 + b`

    Shape:

        - input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=	ext{in1\_features}`
          and :math:`*` means any number of additional dimensions.
          All but the last dimension of the inputs should be the same.
        - input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=	ext{in2\_features}`
        - weight: :math:`(	ext{out\_features}, 	ext{in1\_features},
          	ext{in2\_features})`
        - bias: :math:`(	ext{out\_features})`
        - output: :math:`(N, *, H_{out})` where :math:`H_{out}=	ext{out\_features}`
          and all but the last dimension are the same shape as the input.
    """
    return torch.bilinear(input1, input2, weight, bias)

def _no_grad_embedding_renorm_(weight, input, max_norm, norm_type):
    with torch.no_grad():
        torch.embedding_renorm_(weight, input, max_norm, norm_type)

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False):
    """A simple lookup table that looks up embeddings in a fixed dictionary and size.

    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.

    See :class:`torch.nn.Embedding` for more details.

    Args:
        input (LongTensor): Tensor containing indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        padding_idx (int, optional): If given, pads the output with the embedding vector at :attr:`padding_idx`
                                         (initialized to zeros) whenever it encounters the index.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The p of the p-norm to compute for the :attr:`max_norm` option. Default ``2``.
        scale_grad_by_freq (boolean, optional): If given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
        sparse (bool, optional): If ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.

    Shape:
        - Input: LongTensor of arbitrary shape containing the indices to extract
        - Weight: Embedding matrix of floating point type with shape `(V, embedding_dim)`,
                            where V = maximum index + 1 and embedding_dim = the embedding size
        - Output: `(*, embedding_dim)`, where `*` is the input shape

    Examples::

        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([[1,2,4,5],[4,3,2,9]])
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> F.embedding(input, embedding_matrix)
        tensor([[[ 0.8490,  0.9625,  0.6753],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.6246,  0.9751,  0.3618],
                 [ 0.4161,  0.2419,  0.7383]],

                [[ 0.6246,  0.9751,  0.3618],
                 [ 0.0237,  0.7794,  0.0528],
                 [ 0.9666,  0.7761,  0.6108],
                 [ 0.3385,  0.8612,  0.1867]]])

        >>> # example with padding_idx
        >>> weights = torch.rand(10, 3)
        >>> weights[0, :].zero_()
        >>> embedding_matrix = weights
        >>> input = torch.tensor([[0,2,0,5]])
        >>> F.embedding(input, embedding_matrix, padding_idx=0)
        tensor([[[ 0.0000,  0.0000,  0.0000],
                 [ 0.5609,  0.5384,  0.8720],
                 [ 0.0000,  0.0000,  0.0000],
                 [ 0.6262,  0.2438,  0.7471]]])
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.embedding', 'embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False)', {'_no_grad_embedding_renorm_': _no_grad_embedding_renorm_, 'torch': torch, 'input': input, 'weight': weight, 'padding_idx': padding_idx, 'max_norm': max_norm, 'norm_type': norm_type, 'scale_grad_by_freq': scale_grad_by_freq, 'sparse': sparse}, 1)

def embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, mode='mean', sparse=False, per_sample_weights=None, include_last_offset=False):
    """Computes sums, means or maxes of `bags` of embeddings, without instantiating the
    intermediate embeddings.

    See :class:`torch.nn.EmbeddingBag` for more details.

    .. include:: cuda_deterministic_backward.rst

    Args:
        input (LongTensor): Tensor containing bags of indices into the embedding matrix
        weight (Tensor): The embedding matrix with number of rows equal to the maximum possible index + 1,
            and number of columns equal to the embedding size
        offsets (LongTensor, optional): Only used when :attr:`input` is 1D. :attr:`offsets` determines
                             the starting index position of each bag (sequence) in :attr:`input`.
        max_norm (float, optional): If given, each embedding vector with norm larger than :attr:`max_norm`
                                    is renormalized to have norm :attr:`max_norm`.
                                    Note: this will modify :attr:`weight` in-place.
        norm_type (float, optional): The ``p`` in the ``p``-norm to compute for the :attr:`max_norm` option.
                                     Default ``2``.
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the inverse of frequency of
                                                the words in the mini-batch. Default ``False``.
                                                Note: this option is not supported when ``mode="max"``.
        mode (string, optional): ``"sum"``, ``"mean"`` or ``"max"``. Specifies the way to reduce the bag.
                                 Default: ``"mean"``
        sparse (bool, optional): if ``True``, gradient w.r.t. :attr:`weight` will be a sparse tensor. See Notes under
                                 :class:`torch.nn.Embedding` for more details regarding sparse gradients.
                                 Note: this option is not supported when ``mode="max"``.
        per_sample_weights (Tensor, optional): a tensor of float / double weights, or None
            to indicate all weights should be taken to be 1. If specified, :attr:`per_sample_weights`
            must have exactly the same shape as input and is treated as having the same
            :attr:`offsets`, if those are not None.

        include_last_offset (bool, optional): if ``True``, the size of offsets is equal to the number of bags + 1.
        The last element is the size of the input, or the ending index position of the last bag (sequence).


    Shape:

        - :attr:`input` (LongTensor) and :attr:`offsets` (LongTensor, optional)

          - If :attr:`input` is 2D of shape `(B, N)`,

            it will be treated as ``B`` bags (sequences) each of fixed length ``N``, and
            this will return ``B`` values aggregated in a way depending on the :attr:`mode`.
            :attr:`offsets` is ignored and required to be ``None`` in this case.

          - If :attr:`input` is 1D of shape `(N)`,

            it will be treated as a concatenation of multiple bags (sequences).
            :attr:`offsets` is required to be a 1D tensor containing the
            starting index positions of each bag in :attr:`input`. Therefore,
            for :attr:`offsets` of shape `(B)`, :attr:`input` will be viewed as
            having ``B`` bags. Empty bags (i.e., having 0-length) will have
            returned vectors filled by zeros.

        - :attr:`weight` (Tensor): the learnable weights of the module of
          shape `(num_embeddings, embedding_dim)`

        - :attr:`per_sample_weights` (Tensor, optional). Has the same shape as
          :attr:`input`.

        - :attr:`output`: aggregated embedding values of shape `(B, embedding_dim)`

    Examples::

        >>> # an Embedding module containing 10 tensors of size 3
        >>> embedding_matrix = torch.rand(10, 3)
        >>> # a batch of 2 samples of 4 indices each
        >>> input = torch.tensor([1,2,4,5,4,3,2,9])
        >>> offsets = torch.tensor([0,4])
        >>> F.embedding_bag(embedding_matrix, input, offsets)
        tensor([[ 0.3397,  0.3552,  0.5545],
                [ 0.5893,  0.4386,  0.5882]])
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.embedding_bag', "embedding_bag(input, weight, offsets=None, max_norm=None, norm_type=2, scale_grad_by_freq=False, mode='mean', sparse=False, per_sample_weights=None, include_last_offset=False)", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'embedding_bag': embedding_bag, 'warnings': warnings, '_no_grad_embedding_renorm_': _no_grad_embedding_renorm_, 'input': input, 'weight': weight, 'offsets': offsets, 'max_norm': max_norm, 'norm_type': norm_type, 'scale_grad_by_freq': scale_grad_by_freq, 'mode': mode, 'sparse': sparse, 'per_sample_weights': per_sample_weights, 'include_last_offset': include_last_offset}, 1)

def _verify_batch_size(size):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.nn.functional._verify_batch_size', '_verify_batch_size(size)', {'size': size}, 0)

def batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05):
    """Applies Batch Normalization for each channel across a batch of data.

    See :class:`~torch.nn.BatchNorm1d`, :class:`~torch.nn.BatchNorm2d`,
    :class:`~torch.nn.BatchNorm3d` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.batch_norm', 'batch_norm(input, running_mean, running_var, weight=None, bias=None, training=False, momentum=0.1, eps=1e-05)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'batch_norm': batch_norm, '_verify_batch_size': _verify_batch_size, 'input': input, 'running_mean': running_mean, 'running_var': running_var, 'weight': weight, 'bias': bias, 'training': training, 'momentum': momentum, 'eps': eps}, 1)

def instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05):
    """Applies Instance Normalization for each channel in each data sample in a
    batch.

    See :class:`~torch.nn.InstanceNorm1d`, :class:`~torch.nn.InstanceNorm2d`,
    :class:`~torch.nn.InstanceNorm3d` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.instance_norm', 'instance_norm(input, running_mean=None, running_var=None, weight=None, bias=None, use_input_stats=True, momentum=0.1, eps=1e-05)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'instance_norm': instance_norm, '_verify_batch_size': _verify_batch_size, 'input': input, 'running_mean': running_mean, 'running_var': running_var, 'weight': weight, 'bias': bias, 'use_input_stats': use_input_stats, 'momentum': momentum, 'eps': eps}, 1)

def layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-05):
    """Applies Layer Normalization for last certain number of dimensions.

    See :class:`~torch.nn.LayerNorm` for details.
    """
    return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)

def group_norm(input, num_groups, weight=None, bias=None, eps=1e-05):
    """Applies Group Normalization for last certain number of dimensions.

    See :class:`~torch.nn.GroupNorm` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.group_norm', 'group_norm(input, num_groups, weight=None, bias=None, eps=1e-05)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'group_norm': group_norm, '_verify_batch_size': _verify_batch_size, 'input': input, 'num_groups': num_groups, 'weight': weight, 'bias': bias, 'eps': eps}, 1)

def local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0):
    """Applies local response normalization over an input signal composed of
    several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    See :class:`~torch.nn.LocalResponseNorm` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.local_response_norm', 'local_response_norm(input, size, alpha=0.0001, beta=0.75, k=1.0)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'local_response_norm': local_response_norm, 'pad': pad, 'avg_pool2d': avg_pool2d, 'avg_pool3d': avg_pool3d, 'input': input, 'size': size, 'alpha': alpha, 'beta': beta, 'k': k}, 1)

def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean', zero_infinity=False):
    """The Connectionist Temporal Classification loss.

    See :class:`~torch.nn.CTCLoss` for details.

    .. include:: cudnn_deterministic.rst
    .. include:: cuda_deterministic_backward.rst

    Args:
        log_probs: :math:`(T, N, C)` where `C = number of characters in alphabet including blank`,
            `T = input length`, and `N = batch size`.
            The logarithmized probabilities of the outputs
            (e.g. obtained with :func:`torch.nn.functional.log_softmax`).
        targets: :math:`(N, S)` or `(sum(target_lengths))`.
            Targets cannot be blank. In the second form, the targets are assumed to be concatenated.
        input_lengths: :math:`(N)`.
            Lengths of the inputs (must each be :math:`\leq T`)
        target_lengths: :math:`(N)`.
            Lengths of the targets
        blank (int, optional):
            Blank label. Default :math:`0`.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output losses will be divided by the target lengths and
            then the mean over the batch is taken, ``'sum'``: the output will be
            summed. Default: ``'mean'``
        zero_infinity (bool, optional):
            Whether to zero infinite losses and the associated gradients.
            Default: ``False``
            Infinite losses mainly occur when the inputs are too short
            to be aligned to the targets.

    Example::

        >>> log_probs = torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_()
        >>> targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        >>> input_lengths = torch.full((16,), 50, dtype=torch.long)
        >>> target_lengths = torch.randint(10,30,(16,), dtype=torch.long)
        >>> loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        >>> loss.backward()
    """
    return torch.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, _Reduction.get_enum(reduction), zero_infinity)

def nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    """The negative log likelihood loss.

    See :class:`~torch.nn.NLLLoss` for details.

    Args:
        input: :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss.
        target: :math:`(N)` where each value is :math:`0 \leq 	ext{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Example::

        >>> # input is of size N x C = 3 x 5
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> # each element in target has to have 0 <= value < C
        >>> target = torch.tensor([1, 0, 4])
        >>> output = F.nll_loss(F.log_softmax(input), target)
        >>> output.backward()
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.nll_loss', "nll_loss(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'nll_loss': nll_loss, '_Reduction': _Reduction, 'input': input, 'target': target, 'weight': weight, 'size_average': size_average, 'ignore_index': ignore_index, 'reduce': reduce, 'reduction': reduction}, 1)

def poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean'):
    """Poisson negative log likelihood loss.

    See :class:`~torch.nn.PoissonNLLLoss` for details.

    Args:
        input: expectation of underlying Poisson distribution.
        target: random sample :math:`target \sim 	ext{Poisson}(input)`.
        log_input: if ``True`` the loss is computed as
            :math:`\exp(	ext{input}) - 	ext{target} * 	ext{input}`, if ``False`` then loss is
            :math:`	ext{input} - 	ext{target} * \log(	ext{input}+	ext{eps})`. Default: ``True``
        full: whether to compute full loss, i. e. to add the Stirling
            approximation term. Default: ``False``
            :math:`	ext{target} * \log(	ext{target}) - 	ext{target} + 0.5 * \log(2 * \pi * 	ext{target})`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        eps (float, optional): Small value to avoid evaluation of :math:`\log(0)` when
            :attr:`log_input`=``False``. Default: 1e-8
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.poisson_nll_loss', "poisson_nll_loss(input, target, log_input=True, full=False, size_average=None, eps=1e-08, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'poisson_nll_loss': poisson_nll_loss, '_Reduction': _Reduction, 'input': input, 'target': target, 'log_input': log_input, 'full': full, 'size_average': size_average, 'eps': eps, 'reduce': reduce, 'reduction': reduction}, 1)

def kl_div(input, target, size_average=None, reduce=None, reduction='mean'):
    """The `Kullback-Leibler divergence`_ Loss.

    See :class:`~torch.nn.KLDivLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'batchmean'`` | ``'sum'`` | ``'mean'``.
            ``'none'``: no reduction will be applied
            ``'batchmean'``: the sum of the output will be divided by the batchsize
            ``'sum'``: the output will be summed
            ``'mean'``: the output will be divided by the number of elements in the output
            Default: ``'mean'``

    .. note::
        :attr:`size_average` and :attr:`reduce` are in the process of being deprecated,
        and in the meantime, specifying either of those two args will override :attr:`reduction`.

    .. note::
        :attr:``reduction`` = ``'mean'`` doesn't return the true kl divergence value, please use
        :attr:``reduction`` = ``'batchmean'`` which aligns with KL math definition.
        In the next major release, ``'mean'`` will be changed to be the same as 'batchmean'.

    .. _Kullback-Leibler divergence:
        https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.kl_div', "kl_div(input, target, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'kl_div': kl_div, '_Reduction': _Reduction, 'warnings': warnings, 'input': input, 'target': target, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    """This criterion combines `log_softmax` and `nll_loss` in a single
    function.

    See :class:`~torch.nn.CrossEntropyLoss` for details.

    Args:
        input (Tensor) : :math:`(N, C)` where `C = number of classes` or :math:`(N, C, H, W)`
            in case of 2D Loss, or :math:`(N, C, d_1, d_2, ..., d_K)` where :math:`K \geq 1`
            in the case of K-dimensional loss.
        target (Tensor) : :math:`(N)` where each value is :math:`0 \leq 	ext{targets}[i] \leq C-1`,
            or :math:`(N, d_1, d_2, ..., d_K)` where :math:`K \geq 1` for
            K-dimensional loss.
        weight (Tensor, optional): a manual rescaling weight given to each
            class. If given, has to be a Tensor of size `C`
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When :attr:`size_average` is
            ``True``, the loss is averaged over non-ignored targets. Default: -100
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Examples::

        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randint(5, (3,), dtype=torch.int64)
        >>> loss = F.cross_entropy(input, target)
        >>> loss.backward()
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.cross_entropy', "cross_entropy(input, target, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'cross_entropy': cross_entropy, '_Reduction': _Reduction, 'nll_loss': nll_loss, 'log_softmax': log_softmax, 'input': input, 'target': target, 'weight': weight, 'size_average': size_average, 'ignore_index': ignore_index, 'reduce': reduce, 'reduction': reduction}, 1)

def binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean'):
    """Function that measures the Binary Cross Entropy
    between the target and the output.

    See :class:`~torch.nn.BCELoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
                if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``

    Examples::

        >>> input = torch.randn((3, 2), requires_grad=True)
        >>> target = torch.rand((3, 2), requires_grad=False)
        >>> loss = F.binary_cross_entropy(F.sigmoid(input), target)
        >>> loss.backward()
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.binary_cross_entropy', "binary_cross_entropy(input, target, weight=None, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'binary_cross_entropy': binary_cross_entropy, '_Reduction': _Reduction, 'warnings': warnings, '_infer_size': _infer_size, 'input': input, 'target': target, 'weight': weight, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None):
    """Function that measures Binary Cross Entropy between target and output
    logits.

    See :class:`~torch.nn.BCEWithLogitsLoss` for details.

    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        weight (Tensor, optional): a manual rescaling weight
            if provided it's repeated to match input tensor shape
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when reduce is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.

    Examples::

         >>> input = torch.randn(3, requires_grad=True)
         >>> target = torch.empty(3).random_(2)
         >>> loss = F.binary_cross_entropy_with_logits(input, target)
         >>> loss.backward()
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.binary_cross_entropy_with_logits', "binary_cross_entropy_with_logits(input, target, weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'binary_cross_entropy_with_logits': binary_cross_entropy_with_logits, '_Reduction': _Reduction, 'input': input, 'target': target, 'weight': weight, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction, 'pos_weight': pos_weight}, 1)

def _pointwise_loss(lambd, lambd_optimized, input, target, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional._pointwise_loss', "_pointwise_loss(lambd, lambd_optimized, input, target, reduction='mean')", {'torch': torch, '_Reduction': _Reduction, 'lambd': lambd, 'lambd_optimized': lambd_optimized, 'input': input, 'target': target, 'reduction': reduction}, 1)

def _smooth_l1_loss(input, target):
    t = torch.abs(input - target)
    return torch.where(t < 1, 0.5 * t**2, t - 0.5)

def smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """Function that uses a squared term if the absolute
    element-wise error falls below 1 and an L1 term otherwise.

    See :class:`~torch.nn.SmoothL1Loss` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.smooth_l1_loss', "smooth_l1_loss(input, target, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'smooth_l1_loss': smooth_l1_loss, 'warnings': warnings, '_Reduction': _Reduction, '_smooth_l1_loss': _smooth_l1_loss, 'input': input, 'target': target, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def l1_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """l1_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Function that takes the mean element-wise absolute value difference.

    See :class:`~torch.nn.L1Loss` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.l1_loss', "l1_loss(input, target, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'l1_loss': l1_loss, 'warnings': warnings, '_Reduction': _Reduction, 'input': input, 'target': target, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def mse_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error.

    See :class:`~torch.nn.MSELoss` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.mse_loss', "mse_loss(input, target, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'mse_loss': mse_loss, 'warnings': warnings, '_Reduction': _Reduction, 'input': input, 'target': target, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean'):
    """margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MarginRankingLoss` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.margin_ranking_loss', "margin_ranking_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'margin_ranking_loss': margin_ranking_loss, '_Reduction': _Reduction, 'input1': input1, 'input2': input2, 'target': target, 'margin': margin, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean'):
    """hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.HingeEmbeddingLoss` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.hinge_embedding_loss', "hinge_embedding_loss(input, target, margin=1.0, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'hinge_embedding_loss': hinge_embedding_loss, '_Reduction': _Reduction, 'input': input, 'target': target, 'margin': margin, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiLabelMarginLoss` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.multilabel_margin_loss', "multilabel_margin_loss(input, target, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'multilabel_margin_loss': multilabel_margin_loss, '_Reduction': _Reduction, 'input': input, 'target': target, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean'):
    """soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.SoftMarginLoss` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.soft_margin_loss', "soft_margin_loss(input, target, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'soft_margin_loss': soft_margin_loss, '_Reduction': _Reduction, 'input': input, 'target': target, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def multilabel_soft_margin_loss(input, target, weight=None, size_average=None, reduce=None, reduction='mean'):
    """multilabel_soft_margin_loss(input, target, weight=None, size_average=None) -> Tensor

    See :class:`~torch.nn.MultiLabelSoftMarginLoss` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.multilabel_soft_margin_loss', "multilabel_soft_margin_loss(input, target, weight=None, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'multilabel_soft_margin_loss': multilabel_soft_margin_loss, '_Reduction': _Reduction, 'logsigmoid': logsigmoid, 'input': input, 'target': target, 'weight': weight, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean'):
    """cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.CosineEmbeddingLoss` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.cosine_embedding_loss', "cosine_embedding_loss(input1, input2, target, margin=0, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'cosine_embedding_loss': cosine_embedding_loss, '_Reduction': _Reduction, 'input1': input1, 'input2': input2, 'target': target, 'margin': margin, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def multi_margin_loss(input, target, p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean'):
    """multi_margin_loss(input, target, p=1, margin=1, weight=None, size_average=None,
                          reduce=None, reduction='mean') -> Tensor

    See :class:`~torch.nn.MultiMarginLoss` for details.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.multi_margin_loss', "multi_margin_loss(input, target, p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'multi_margin_loss': multi_margin_loss, '_Reduction': _Reduction, 'input': input, 'target': target, 'p': p, 'margin': margin, 'weight': weight, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)
pixel_shuffle = _add_docstr(torch.pixel_shuffle, '\nRearranges elements in a tensor of shape :math:`(*, C \\times r^2, H, W)` to a\ntensor of shape :math:`(*, C, H \\times r, W \\times r)`.\n\nSee :class:`~torch.nn.PixelShuffle` for details.\n\nArgs:\n    input (Tensor): the input tensor\n    upscale_factor (int): factor to increase spatial resolution by\n\nExamples::\n\n    >>> input = torch.randn(1, 9, 4, 4)\n    >>> output = torch.nn.functional.pixel_shuffle(input, 3)\n    >>> print(output.size())\n    torch.Size([1, 1, 12, 12])\n')

@_overload
def upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    pass

@_overload
def upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    pass

def upsample(input, size=None, scale_factor=None, mode='nearest', align_corners=None):
    """Upsamples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with ``nn.functional.interpolate(...)``.

    .. include:: cuda_deterministic_backward.rst

    The algorithm used for upsampling is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric upsampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for upsampling are: `nearest`, `linear` (3D-only),
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only)

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to be an integer.
        mode (string): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'``. Default: ``'nearest'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
            Default: ``False``

    .. note::
        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
        negative values or values greater than 255 for images.
        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
        when displaying the image.

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.

    """
    warnings.warn('nn.functional.upsample is deprecated. Use nn.functional.interpolate instead.')
    return interpolate(input, size, scale_factor, mode, align_corners)

@_overload
def _interp_output_size(dim, closed_over_args):
    pass

@_overload
def _interp_output_size(dim, closed_over_args):
    pass

@_overload
def _interp_output_size(dim, closed_over_args):
    pass

@_overload
def _interp_output_size(dim, closed_over_args):
    pass

def _interp_output_size(dim, closed_over_args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional._interp_output_size', '_interp_output_size(dim, closed_over_args)', {'math': math, 'warnings': warnings, 'torch': torch, 'dim': dim, 'closed_over_args': closed_over_args}, 1)

@_overload
def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    pass

@_overload
def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    pass

@_overload
def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    pass

@_overload
def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    pass

def interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None):
    """Down/up samples the input to either the given :attr:`size` or the given
    :attr:`scale_factor`

    The algorithm used for interpolation is determined by :attr:`mode`.

    Currently temporal, spatial and volumetric sampling are supported, i.e.
    expected inputs are 3-D, 4-D or 5-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x [optional depth] x [optional height] x width`.

    The modes available for resizing are: `nearest`, `linear` (3D-only),
    `bilinear`, `bicubic` (4D-only), `trilinear` (5D-only), `area`

    Args:
        input (Tensor): the input tensor
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]):
            output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size. Has to match input size if it is a tuple.
        mode (str): algorithm used for upsampling:
            ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
            ``'trilinear'`` | ``'area'``. Default: ``'nearest'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input and output as squares rather than points.
            If set to ``True``, the input and output tensors are aligned by the
            center points of their corner pixels, preserving the values at the corner pixels.
            If set to ``False``, the input and output tensors are aligned by the corner
            points of their corner pixels, and the interpolation uses edge value padding
            for out-of-boundary values, making this operation *independent* of input size
            when :attr:`scale_factor` is kept the same. This only has an effect when :attr:`mode`
            is ``'linear'``, ``'bilinear'``, ``'bicubic'`` or ``'trilinear'``.
            Default: ``False``
        recompute_scale_factor (bool, optional): recompute the scale_factor for use in the
            interpolation calculation.  When `scale_factor` is passed as a parameter, it is used
            to compute the `output_size`.  If `recompute_scale_factor` is ```True`` or not specified,
            a new `scale_factor` will be computed based on the output and input sizes for use in the
            interpolation computation (i.e. the computation will be identical to if the computed
            `output_size` were passed-in explicitly).  Otherwise, the passed-in `scale_factor` will
            be used in the interpolation computation.  Note that when `scale_factor` is floating-point,
            the recomputed scale_factor may differ from the one passed in due to rounding and precision
            issues.

    .. note::
        With ``mode='bicubic'``, it's possible to cause overshoot, in other words it can produce
        negative values or values greater than 255 for images.
        Explicitly call ``result.clamp(min=0, max=255)`` if you want to reduce the overshoot
        when displaying the image.

    .. warning::
        With ``align_corners = True``, the linearly interpolating modes
        (`linear`, `bilinear`, and `trilinear`) don't proportionally align the
        output and input pixels, and thus the output values can depend on the
        input size. This was the default behavior for these modes up to version
        0.3.1. Since then, the default behavior is ``align_corners = False``.
        See :class:`~torch.nn.Upsample` for concrete examples on how this
        affects the outputs.

    .. warning::
        When scale_factor is specified, if recompute_scale_factor=True,
        scale_factor is used to compute the output_size which will then
        be used to infer new scales for the interpolation. This is the current
        default behavior when recompute_scale_factor is not specified.
        The default behavior for recompute_scale_factor will change to False
        in 1.6.0, and scale_factor will be used in the interpolation
        calculation.

    .. include:: cuda_deterministic_backward.rst
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.interpolate', "interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None)", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'interpolate': interpolate, 'warnings': warnings, 'List': List, 'Optional': Optional, '_interp_output_size': _interp_output_size, 'adaptive_avg_pool1d': adaptive_avg_pool1d, 'adaptive_avg_pool2d': adaptive_avg_pool2d, 'adaptive_avg_pool3d': adaptive_avg_pool3d, 'input': input, 'size': size, 'scale_factor': scale_factor, 'mode': mode, 'align_corners': align_corners, 'recompute_scale_factor': recompute_scale_factor}, 1)

@_overload
def upsample_nearest(input, size=None, scale_factor=None):
    pass

@_overload
def upsample_nearest(input, size=None, scale_factor=None):
    pass

def upsample_nearest(input, size=None, scale_factor=None):
    """Upsamples the input, using nearest neighbours' pixel values.

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with ``nn.functional.interpolate(..., mode='nearest')``.

    Currently spatial and volumetric upsampling are supported (i.e. expected
    inputs are 4 or 5 dimensional).

    Args:
        input (Tensor): input
        size (int or Tuple[int, int] or Tuple[int, int, int]): output spatia
            size.
        scale_factor (int): multiplier for spatial size. Has to be an integer.

    .. include:: cuda_deterministic_backward.rst
    """
    warnings.warn('nn.functional.upsample_nearest is deprecated. Use nn.functional.interpolate instead.')
    return interpolate(input, size, scale_factor, mode='nearest')

@_overload
def upsample_bilinear(input, size=None, scale_factor=None):
    pass

@_overload
def upsample_bilinear(input, size=None, scale_factor=None):
    pass

@_overload
def upsample_bilinear(input, size=None, scale_factor=None):
    pass

@_overload
def upsample_bilinear(input, size=None, scale_factor=None):
    pass

def upsample_bilinear(input, size=None, scale_factor=None):
    """Upsamples the input, using bilinear upsampling.

    .. warning::
        This function is deprecated in favor of :func:`torch.nn.functional.interpolate`.
        This is equivalent with
        ``nn.functional.interpolate(..., mode='bilinear', align_corners=True)``.

    Expected inputs are spatial (4 dimensional). Use `upsample_trilinear` fo
    volumetric (5 dimensional) inputs.

    Args:
        input (Tensor): input
        size (int or Tuple[int, int]): output spatial size.
        scale_factor (int or Tuple[int, int]): multiplier for spatial size

    .. include:: cuda_deterministic_backward.rst
    """
    warnings.warn('nn.functional.upsample_bilinear is deprecated. Use nn.functional.interpolate instead.')
    return interpolate(input, size, scale_factor, mode='bilinear', align_corners=True)
GRID_SAMPLE_INTERPOLATION_MODES = {'bilinear': 0, 'nearest': 1}
GRID_SAMPLE_PADDING_MODES = {'zeros': 0, 'border': 1, 'reflection': 2}

def grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None):
    """Given an :attr:`input` and a flow-field :attr:`grid`, computes the
    ``output`` using :attr:`input` values and pixel locations from :attr:`grid`.

    Currently, only spatial (4-D) and volumetric (5-D) :attr:`input` are
    supported.

    In the spatial (4-D) case, for :attr:`input` with shape
    :math:`(N, C, H_	ext{in}, W_	ext{in})` and :attr:`grid` with shape
    :math:`(N, H_	ext{out}, W_	ext{out}, 2)`, the output will have shape
    :math:`(N, C, H_	ext{out}, W_	ext{out})`.

    For each output location ``output[n, :, h, w]``, the size-2 vector
    ``grid[n, h, w]`` specifies :attr:`input` pixel locations ``x`` and ``y``,
    which are used to interpolate the output value ``output[n, :, h, w]``.
    In the case of 5D inputs, ``grid[n, d, h, w]`` specifies the
    ``x``, ``y``, ``z`` pixel locations for interpolating
    ``output[n, :, d, h, w]``. :attr:`mode` argument specifies ``nearest`` or
    ``bilinear`` interpolation method to sample the input pixels.

    :attr:`grid` specifies the sampling pixel locations normalized by the
    :attr:`input` spatial dimensions. Therefore, it should have most values in
    the range of ``[-1, 1]``. For example, values ``x = -1, y = -1`` is the
    left-top pixel of :attr:`input`, and values  ``x = 1, y = 1`` is the
    right-bottom pixel of :attr:`input`.

    If :attr:`grid` has values outside the range of ``[-1, 1]``, the corresponding
    outputs are handled as defined by :attr:`padding_mode`. Options are

        * ``padding_mode="zeros"``: use ``0`` for out-of-bound grid locations,
        * ``padding_mode="border"``: use border values for out-of-bound grid locations,
        * ``padding_mode="reflection"``: use values at locations reflected by
          the border for out-of-bound grid locations. For location far away
          from the border, it will keep being reflected until becoming in bound,
          e.g., (normalized) pixel location ``x = -3.5`` reflects by border ``-1``
          and becomes ``x' = 1.5``, then reflects by border ``1`` and becomes
          ``x'' = -0.5``.

    .. note::
        This function is often used in conjunction with :func:`affine_grid`
        to build `Spatial Transformer Networks`_ .
    .. include:: cuda_deterministic_backward.rst

    Args:
        input (Tensor): input of shape :math:`(N, C, H_	ext{in}, W_	ext{in})` (4-D case)
                        or :math:`(N, C, D_	ext{in}, H_	ext{in}, W_	ext{in})` (5-D case)
        grid (Tensor): flow-field of shape :math:`(N, H_	ext{out}, W_	ext{out}, 2)` (4-D case)
                       or :math:`(N, D_	ext{out}, H_	ext{out}, W_	ext{out}, 3)` (5-D case)
        mode (str): interpolation mode to calculate output values
            ``'bilinear'`` | ``'nearest'``. Default: ``'bilinear'``
        padding_mode (str): padding mode for outside grid values
            ``'zeros'`` | ``'border'`` | ``'reflection'``. Default: ``'zeros'``
        align_corners (bool, optional): Geometrically, we consider the pixels of the
            input  as squares rather than points.
            If set to ``True``, the extrema (``-1`` and ``1``) are considered as referring
            to the center points of the input's corner pixels. If set to ``False``, they
            are instead considered as referring to the corner points of the input's corner
            pixels, making the sampling more resolution agnostic.
            This option parallels the ``align_corners`` option in
            :func:`interpolate`, and so whichever option is used here
            should also be used there to resize the input image before grid sampling.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.grid_sample', "grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=None)", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'grid_sample': grid_sample, 'warnings': warnings, 'input': input, 'grid': grid, 'mode': mode, 'padding_mode': padding_mode, 'align_corners': align_corners}, 1)

def affine_grid(theta, size, align_corners=None):
    """Generates a 2D or 3D flow field (sampling grid), given a batch of
    affine matrices :attr:`theta`.

    .. note::
        This function is often used in conjunction with :func:`grid_sample`
        to build `Spatial Transformer Networks`_ .

    Args:
        theta (Tensor): input batch of affine matrices with shape
            (:math:`N 	imes 2 	imes 3`) for 2D or
            (:math:`N 	imes 3 	imes 4`) for 3D
        size (torch.Size): the target output image size.
            (:math:`N 	imes C 	imes H 	imes W` for 2D or
            :math:`N 	imes C 	imes D 	imes H 	imes W` for 3D)
            Example: torch.Size((32, 3, 24, 24))
        align_corners (bool, optional): if ``True``, consider ``-1`` and ``1``
            to refer to the centers of the corner pixels rather than the image corners.
            Refer to :func:`grid_sample` for a more complete description.
            A grid generated by :func:`affine_grid` should be passed to :func:`grid_sample`
            with the same setting for this option.
            Default: ``False``

    Returns:
        output (Tensor): output Tensor of size (:math:`N 	imes H 	imes W 	imes 2`)

    .. _`Spatial Transformer Networks`:
        https://arxiv.org/abs/1506.02025

    .. warning::
        When ``align_corners = True``, the grid positions depend on the pixel
        size relative to the input image size, and so the locations sampled by
        :func:`grid_sample` will differ for the same input given at different
        resolutions (that is, after being upsampled or downsampled).
        The default behavior up to version 1.2.0 was ``align_corners = True``.
        Since then, the default behavior has been changed to ``align_corners = False``,
        in order to bring it in line with the default for :func:`interpolate`.
    .. warning::
        When ``align_corners = True``, 2D affine transforms on 1D data and
        3D affine transforms on 2D data (that is, when one of the spatial
        dimensions has unit size) are ill-defined, and not an intended use case.
        This is not a problem when ``align_corners = False``.
        Up to version 1.2.0, all grid points along a unit dimension were
        considered arbitrarily to be at ``-1``.
        From version 1.3.0, under ``align_corners = True`` all grid points
        along a unit dimension are condsidered to be at ```0``
        (the center of the input image).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.affine_grid', 'affine_grid(theta, size, align_corners=None)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'affine_grid': affine_grid, 'warnings': warnings, 'theta': theta, 'size': size, 'align_corners': align_corners}, 1)

def _pad(input, pad, mode='constant', value=0):
    """Pads tensor.

    Padding size:
        The padding size by which to pad some dimensions of :attr:`input`
        are described starting from the last dimension and moving forward.
        :math:`\left\lfloorrac{	ext{len(pad)}}{2}ightfloor` dimensions
        of ``input`` will be padded.
        For example, to pad only the last dimension of the input tensor, then
        :attr:`pad` has the form
        :math:`(	ext{padding\_left}, 	ext{padding\_right})`;
        to pad the last 2 dimensions of the input tensor, then use
        :math:`(	ext{padding\_left}, 	ext{padding\_right},`
        :math:`	ext{padding\_top}, 	ext{padding\_bottom})`;
        to pad the last 3 dimensions, use
        :math:`(	ext{padding\_left}, 	ext{padding\_right},`
        :math:`	ext{padding\_top}, 	ext{padding\_bottom}`
        :math:`	ext{padding\_front}, 	ext{padding\_back})`.

    Padding mode:
        See :class:`torch.nn.ConstantPad2d`, :class:`torch.nn.ReflectionPad2d`, and
        :class:`torch.nn.ReplicationPad2d` for concrete examples on how each of the
        padding modes works. Constant padding is implemented for arbitrary dimensions.
        Replicate padding is implemented for padding the last 3 dimensions of 5D input
        tensor, or the last 2 dimensions of 4D input tensor, or the last dimension of
        3D input tensor. Reflect padding is only implemented for padding the last 2
        dimensions of 4D input tensor, or the last dimension of 3D input tensor.

    .. include:: cuda_deterministic_backward.rst

    Args:
        input (Tensor): N-dimensional tensor
        pad (tuple): m-elements tuple, where
            :math:`rac{m}{2} \leq` input dimensions and :math:`m` is even.
        mode: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
            Default: ``'constant'``
        value: fill value for ``'constant'`` padding. Default: ``0``

    Examples::

        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p1d = (1, 1) # pad last dim by 1 on each side
        >>> out = F.pad(t4d, p1d, "constant", 0)  # effectively zero padding
        >>> print(out.size())
        torch.Size([3, 3, 4, 4])
        >>> p2d = (1, 1, 2, 2) # pad last dim by (1, 1) and 2nd to last by (2, 2)
        >>> out = F.pad(t4d, p2d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 3, 8, 4])
        >>> t4d = torch.empty(3, 3, 4, 2)
        >>> p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        >>> out = F.pad(t4d, p3d, "constant", 0)
        >>> print(out.size())
        torch.Size([3, 9, 7, 3])

    """
    if not torch.jit.is_scripting():
        if (type(input) is not Tensor and has_torch_function((input, ))):
            return handle_torch_function(_pad, (input, ), input, pad, mode=mode, value=value)
    assert len(pad) % 2 == 0, 'Padding length must be divisible by 2'
    assert len(pad) // 2 <= input.dim(), 'Padding length too large'
    if mode == 'constant':
        return _VF.constant_pad_nd(input, pad, value)
    else:
        assert value == 0, 'Padding mode "{}"" doesn\'t take in value argument'.format(mode)
        if input.dim() == 3:
            assert len(pad) == 2, '3D tensors expect 2 values for padding'
            if mode == 'reflect':
                return torch._C._nn.reflection_pad1d(input, pad)
            elif mode == 'replicate':
                return torch._C._nn.replication_pad1d(input, pad)
            elif mode == 'circular':
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError
        elif input.dim() == 4:
            assert len(pad) == 4, '4D tensors expect 4 values for padding'
            if mode == 'reflect':
                return torch._C._nn.reflection_pad2d(input, pad)
            elif mode == 'replicate':
                return torch._C._nn.replication_pad2d(input, pad)
            elif mode == 'circular':
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError
        elif input.dim() == 5:
            assert len(pad) == 6, '5D tensors expect 6 values for padding'
            if mode == 'reflect':
                raise NotImplementedError
            elif mode == 'replicate':
                return torch._C._nn.replication_pad3d(input, pad)
            elif mode == 'circular':
                return _pad_circular(input, pad)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError('Only 3D, 4D, 5D padding with non-constant padding are supported for now')
pad = _pad

def pairwise_distance(x1, x2, p=2.0, eps=1e-06, keepdim=False):
    """
    See :class:`torch.nn.PairwiseDistance` for details
    """
    return torch.pairwise_distance(x1, x2, p, eps, keepdim)
pdist = _add_docstr(torch.pdist, "\npdist(input, p=2) -> Tensor\n\nComputes the p-norm distance between every pair of row vectors in the input.\nThis is identical to the upper triangular portion, excluding the diagonal, of\n`torch.norm(input[:, None] - input, dim=2, p=p)`. This function will be faster\nif the rows are contiguous.\n\nIf input has shape :math:`N \\times M` then the output will have shape\n:math:`\\frac{1}{2} N (N - 1)`.\n\nThis function is equivalent to `scipy.spatial.distance.pdist(input,\n'minkowski', p=p)` if :math:`p \\in (0, \\infty)`. When :math:`p = 0` it is\nequivalent to `scipy.spatial.distance.pdist(input, 'hamming') * M`.\nWhen :math:`p = \\infty`, the closest scipy function is\n`scipy.spatial.distance.pdist(xn, lambda x, y: np.abs(x - y).max())`.\n\nArgs:\n    input: input tensor of shape :math:`N \\times M`.\n    p: p value for the p-norm distance to calculate between each vector pair\n        :math:`\\in [0, \\infty]`.\n")
cosine_similarity = _add_docstr(torch.cosine_similarity, '\ncosine_similarity(x1, x2, dim=1, eps=1e-8) -> Tensor\n\nReturns cosine similarity between x1 and x2, computed along dim.\n\n.. math ::\n    \\text{similarity} = \\dfrac{x_1 \\cdot x_2}{\\max(\\Vert x_1 \\Vert _2 \\cdot \\Vert x_2 \\Vert _2, \\epsilon)}\n\nArgs:\n    x1 (Tensor): First input.\n    x2 (Tensor): Second input (of size matching x1).\n    dim (int, optional): Dimension of vectors. Default: 1\n    eps (float, optional): Small value to avoid division by zero.\n        Default: 1e-8\n\nShape:\n    - Input: :math:`(\\ast_1, D, \\ast_2)` where D is at position `dim`.\n    - Output: :math:`(\\ast_1, \\ast_2)` where 1 is at position `dim`.\n\nExample::\n\n    >>> input1 = torch.randn(100, 128)\n    >>> input2 = torch.randn(100, 128)\n    >>> output = F.cosine_similarity(input1, input2)\n    >>> print(output)\n')
one_hot = _add_docstr(torch._C._nn.one_hot, '\none_hot(tensor, num_classes=-1) -> LongTensor\n\nTakes LongTensor with index values of shape ``(*)`` and returns a tensor\nof shape ``(*, num_classes)`` that have zeros everywhere except where the\nindex of last dimension matches the corresponding value of the input tensor,\nin which case it will be 1.\n\nSee also `One-hot on Wikipedia`_ .\n\n.. _One-hot on Wikipedia:\n    https://en.wikipedia.org/wiki/One-hot\n\nArguments:\n    tensor (LongTensor): class values of any shape.\n    num_classes (int):  Total number of classes. If set to -1, the number\n        of classes will be inferred as one greater than the largest class\n        value in the input tensor.\n\nReturns:\n    LongTensor that has one more dimension with 1 values at the\n    index of last dimension indicated by the input, and 0 everywhere\n    else.\n\nExamples:\n    >>> F.one_hot(torch.arange(0, 5) % 3)\n    tensor([[1, 0, 0],\n            [0, 1, 0],\n            [0, 0, 1],\n            [1, 0, 0],\n            [0, 1, 0]])\n    >>> F.one_hot(torch.arange(0, 5) % 3, num_classes=5)\n    tensor([[1, 0, 0, 0, 0],\n            [0, 1, 0, 0, 0],\n            [0, 0, 1, 0, 0],\n            [1, 0, 0, 0, 0],\n            [0, 1, 0, 0, 0]])\n    >>> F.one_hot(torch.arange(0, 6).view(3,2) % 3)\n    tensor([[[1, 0, 0],\n             [0, 1, 0]],\n            [[0, 0, 1],\n             [1, 0, 0]],\n            [[0, 1, 0],\n             [0, 0, 1]]])\n')

def triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean'):
    """
    See :class:`~torch.nn.TripletMarginLoss` for details
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.triplet_margin_loss', "triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')", {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'triplet_margin_loss': triplet_margin_loss, '_Reduction': _Reduction, 'anchor': anchor, 'positive': positive, 'negative': negative, 'margin': margin, 'p': p, 'eps': eps, 'swap': swap, 'size_average': size_average, 'reduce': reduce, 'reduction': reduction}, 1)

def normalize(input, p=2, dim=1, eps=1e-12, out=None):
    """Performs :math:`L_p` normalization of inputs over specified dimension.

    For a tensor :attr:`input` of sizes :math:`(n_0, ..., n_{dim}, ..., n_k)`, each
    :math:`n_{dim}` -element vector :math:`v` along dimension :attr:`dim` is transformed as

    .. math::
        v = rac{v}{\max(\lVert v Vert_p, \epsilon)}.

    With the default arguments it uses the Euclidean norm over vectors along dimension :math:`1` for normalization.

    Args:
        input: input tensor of any shape
        p (float): the exponent value in the norm formulation. Default: 2
        dim (int): the dimension to reduce. Default: 1
        eps (float): small value to avoid division by zero. Default: 1e-12
        out (Tensor, optional): the output tensor. If :attr:`out` is used, this
                                operation won't be differentiable.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.normalize', 'normalize(input, p=2, dim=1, eps=1e-12, out=None)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'normalize': normalize, 'input': input, 'p': p, 'dim': dim, 'eps': eps, 'out': out}, 1)

def assert_int_or_pair(arg, arg_name, message):
    assert (isinstance(arg, int) or len(arg) == 2), message.format(arg_name)

def unfold(input, kernel_size, dilation=1, padding=0, stride=1):
    """Extracts sliding local blocks from an batched input tensor.

    .. warning::
        Currently, only 4-D input tensors (batched image-like tensors) are
        supported.

    .. warning::

        More than one element of the unfolded tensor may refer to a single
        memory location. As a result, in-place operations (especially ones that
        are vectorized) may result in incorrect behavior. If you need to write
        to the tensor, please clone it first.


    See :class:`torch.nn.Unfold` for details
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.unfold', 'unfold(input, kernel_size, dilation=1, padding=0, stride=1)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'unfold': unfold, 'assert_int_or_pair': assert_int_or_pair, '_pair': _pair, 'input': input, 'kernel_size': kernel_size, 'dilation': dilation, 'padding': padding, 'stride': stride}, 1)

def fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1):
    """Combines an array of sliding local blocks into a large containing
    tensor.

    .. warning::
        Currently, only 4-D output tensors (batched image-like tensors) are
        supported.

    See :class:`torch.nn.Fold` for details
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.fold', 'fold(input, output_size, kernel_size, dilation=1, padding=0, stride=1)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'fold': fold, 'assert_int_or_pair': assert_int_or_pair, '_pair': _pair, 'input': input, 'output_size': output_size, 'kernel_size': kernel_size, 'dilation': dilation, 'padding': padding, 'stride': stride}, 1)

def _pad_circular(input, padding):
    """
    Arguments
        :param input: tensor of shape :math:`(N, C_{	ext{in}}, H, [W, D]))`
        :param padding: (tuple): m-elem tuple where m is the degree of convolution
    Returns
        :return: tensor of shape :math:`(N, C_{	ext{in}}, [D + 2 * padding[0],
                 H + 2 * padding[1]], W + 2 * padding[2]))`
    """
    input = torch.cat([input, input[:, :, 0:padding[-1]]], dim=2)
    input = torch.cat([input[:, :, -(padding[-1] + padding[-2]):-padding[-1]], input], dim=2)
    if len(padding) > 2:
        input = torch.cat([input, input[:, :, :, 0:padding[-3]]], dim=3)
        input = torch.cat([input[:, :, :, -(padding[-3] + padding[-4]):-padding[-3]], input], dim=3)
    if len(padding) > 4:
        input = torch.cat([input, input[:, :, :, :, 0:padding[-5]]], dim=4)
        input = torch.cat([input[:, :, :, :, -(padding[-5] + padding[-6]):-padding[-5]], input], dim=4)
    return input

def multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None):
    """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer). A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.functional.multi_head_attention_forward', 'multi_head_attention_forward(query, key, value, embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight, out_proj_bias, training=True, key_padding_mask=None, need_weights=True, attn_mask=None, use_separate_proj_weight=False, q_proj_weight=None, k_proj_weight=None, v_proj_weight=None, static_k=None, static_v=None)', {'torch': torch, 'Tensor': Tensor, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'multi_head_attention_forward': multi_head_attention_forward, 'linear': linear, 'pad': pad, 'softmax': softmax, 'dropout': dropout, 'query': query, 'key': key, 'value': value, 'embed_dim_to_check': embed_dim_to_check, 'num_heads': num_heads, 'in_proj_weight': in_proj_weight, 'in_proj_bias': in_proj_bias, 'bias_k': bias_k, 'bias_v': bias_v, 'add_zero_attn': add_zero_attn, 'dropout_p': dropout_p, 'out_proj_weight': out_proj_weight, 'out_proj_bias': out_proj_bias, 'training': training, 'key_padding_mask': key_padding_mask, 'need_weights': need_weights, 'attn_mask': attn_mask, 'use_separate_proj_weight': use_separate_proj_weight, 'q_proj_weight': q_proj_weight, 'k_proj_weight': k_proj_weight, 'v_proj_weight': v_proj_weight, 'static_k': static_k, 'static_v': static_v}, 1)

