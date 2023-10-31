"""Gradient interface"""

import torch
from .modules.utils import _single, _pair, _triple

def _grad_input_padding(grad_output, input_size, stride, padding, kernel_size):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.grad._grad_input_padding', '_grad_input_padding(grad_output, input_size, stride, padding, kernel_size)', {'grad_output': grad_output, 'input_size': input_size, 'stride': stride, 'padding': padding, 'kernel_size': kernel_size}, 1)

def conv1d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1):
    """
    Computes the gradient of conv1d with respect to the input of the convolution.
    This is same as the 1D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1,1,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv1d_input(input.shape, weight, grad_output)

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.grad.conv1d_input', 'conv1d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1)', {'_single': _single, '_grad_input_padding': _grad_input_padding, 'torch': torch, 'input_size': input_size, 'weight': weight, 'grad_output': grad_output, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}, 1)

def conv1d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):
    """
    Computes the gradient of conv1d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1,1,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1, requires_grad=True)
        >>> output = F.conv1d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv1d_weight(input, weight.shape, grad_output)

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.grad.conv1d_weight', 'conv1d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1)', {'_single': _single, 'torch': torch, 'input': input, 'weight_size': weight_size, 'grad_output': grad_output, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}, 1)

def conv2d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1):
    """
    Computes the gradient of conv2d with respect to the input of the convolution.
    This is same as the 2D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weight tensor (out_channels x in_channels/groups x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1,1,3,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1,2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv2d_input(input.shape, weight, grad_output)

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.grad.conv2d_input', 'conv2d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1)', {'_pair': _pair, '_grad_input_padding': _grad_input_padding, 'torch': torch, 'input_size': input_size, 'weight': weight, 'grad_output': grad_output, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}, 1)

def conv2d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):
    """
    Computes the gradient of conv2d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(1,1,3,3, requires_grad=True)
        >>> weight = torch.randn(1,1,1,2, requires_grad=True)
        >>> output = F.conv2d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, filter, grad_output)
        >>> F.grad.conv2d_weight(input, weight.shape, grad_output)

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.grad.conv2d_weight', 'conv2d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1)', {'_pair': _pair, 'torch': torch, 'input': input, 'weight_size': weight_size, 'grad_output': grad_output, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}, 1)

def conv3d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1):
    """
    Computes the gradient of conv3d with respect to the input of the convolution.
    This is same as the 3D transposed convolution operator under the hood but requires
    the shape of the gradient w.r.t. input to be specified explicitly.

    Args:
        input_size : Shape of the input gradient tensor
        weight: weights tensor (out_channels x in_channels/groups x kT x kH x kW)
        grad_output : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_input = torch.autograd.grad(output, input, grad_output)
        >>> F.grad.conv3d_input(input.shape, weight, grad_output)

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.grad.conv3d_input', 'conv3d_input(input_size, weight, grad_output, stride=1, padding=0, dilation=1, groups=1)', {'_triple': _triple, '_grad_input_padding': _grad_input_padding, 'torch': torch, 'input_size': input_size, 'weight': weight, 'grad_output': grad_output, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}, 1)

def conv3d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1):
    """
    Computes the gradient of conv3d with respect to the weight of the convolution.

    Args:
        input: input tensor of shape (minibatch x in_channels x iT x iH x iW)
        weight_size : Shape of the weight gradient tensor
        grad_output : output gradient tensor (minibatch x out_channels x oT x oH x oW)
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1

    Examples::

        >>> input = torch.randn(2, 8, 10, 10, 20, requires_grad=True)
        >>> weight = torch.randn(4, 8, 2, 3, 3, requires_grad=True)
        >>> output = F.conv3d(input, weight)
        >>> grad_output = torch.randn(output.shape)
        >>> grad_weight = torch.autograd.grad(output, weight, grad_output)
        >>> F.grad.conv3d_weight(input, weight.shape, grad_output)

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.grad.conv3d_weight', 'conv3d_weight(input, weight_size, grad_output, stride=1, padding=0, dilation=1, groups=1)', {'_triple': _triple, 'torch': torch, 'input': input, 'weight_size': weight_size, 'grad_output': grad_output, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}, 1)

