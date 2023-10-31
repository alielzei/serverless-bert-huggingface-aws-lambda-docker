from __future__ import division
import math
import warnings
import torch

def _no_grad_uniform_(tensor, a, b):
    with torch.no_grad():
        return tensor.uniform_(a, b)

def _no_grad_normal_(tensor, mean, std):
    with torch.no_grad():
        return tensor.normal_(mean, std)

def _no_grad_fill_(tensor, val):
    with torch.no_grad():
        return tensor.fill_(val)

def _no_grad_zero_(tensor):
    with torch.no_grad():
        return tensor.zero_()

def calculate_gain(nonlinearity, param=None):
    """Return the recommended gain value for the given nonlinearity function.
    The values are as follows:

    ================= ====================================================
    nonlinearity      gain
    ================= ====================================================
    Linear / Identity :math:`1`
    Conv{1,2,3}D      :math:`1`
    Sigmoid           :math:`1`
    Tanh              :math:`rac{5}{3}`
    ReLU              :math:`\sqrt{2}`
    Leaky Relu        :math:`\sqrt{rac{2}{1 + 	ext{negative\_slope}^2}}`
    ================= ====================================================

    Args:
        nonlinearity: the non-linear function (`nn.functional` name)
        param: optional parameter for the non-linear function

    Examples:
        >>> gain = nn.init.calculate_gain('leaky_relu', 0.2)  # leaky_relu with negative_slope=0.2
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init.calculate_gain', 'calculate_gain(nonlinearity, param=None)', {'math': math, 'nonlinearity': nonlinearity, 'param': param}, 1)

def uniform_(tensor, a=0.0, b=1.0):
    """Fills the input Tensor with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.uniform_(w)
    """
    return _no_grad_uniform_(tensor, a, b)

def normal_(tensor, mean=0.0, std=1.0):
    """Fills the input Tensor with values drawn from the normal
    distribution :math:`\mathcal{N}(	ext{mean}, 	ext{std}^2)`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.normal_(w)
    """
    return _no_grad_normal_(tensor, mean, std)

def constant_(tensor, val):
    """Fills the input Tensor with the value :math:`	ext{val}`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        val: the value to fill the tensor with

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.constant_(w, 0.3)
    """
    return _no_grad_fill_(tensor, val)

def ones_(tensor):
    """Fills the input Tensor with the scalar value `1`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.ones_(w)
    """
    return _no_grad_fill_(tensor, 1.0)

def zeros_(tensor):
    """Fills the input Tensor with the scalar value `0`.

    Args:
        tensor: an n-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.zeros_(w)
    """
    return _no_grad_zero_(tensor)

def eye_(tensor):
    """Fills the 2-dimensional input `Tensor` with the identity
    matrix. Preserves the identity of the inputs in `Linear` layers, where as
    many inputs are preserved as possible.

    Args:
        tensor: a 2-dimensional `torch.Tensor`

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.eye_(w)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init.eye_', 'eye_(tensor)', {'torch': torch, 'tensor': tensor}, 1)

def dirac_(tensor, groups=1):
    """Fills the {3, 4, 5}-dimensional input `Tensor` with the Dirac
    delta function. Preserves the identity of the inputs in `Convolutional`
    layers, where as many input channels are preserved as possible. In case
    of groups>1, each group of channels preserves identity

    Args:
        tensor: a {3, 4, 5}-dimensional `torch.Tensor`
        groups (optional): number of groups in the conv layer (default: 1)
    Examples:
        >>> w = torch.empty(3, 16, 5, 5)
        >>> nn.init.dirac_(w)
        >>> w = torch.empty(3, 24, 5, 5)
        >>> nn.init.dirac_(w, 3)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init.dirac_', 'dirac_(tensor, groups=1)', {'torch': torch, 'tensor': tensor, 'groups': groups}, 1)

def _calculate_fan_in_and_fan_out(tensor):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init._calculate_fan_in_and_fan_out', '_calculate_fan_in_and_fan_out(tensor)', {'tensor': tensor}, 2)

def xavier_uniform_(tensor, gain=1.0):
    """Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a uniform
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-a, a)` where

    .. math::
        a = 	ext{gain} 	imes \sqrt{rac{6}{	ext{fan\_in} + 	ext{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init.xavier_uniform_', 'xavier_uniform_(tensor, gain=1.0)', {'_calculate_fan_in_and_fan_out': _calculate_fan_in_and_fan_out, 'math': math, '_no_grad_uniform_': _no_grad_uniform_, 'tensor': tensor, 'gain': gain}, 1)

def xavier_normal_(tensor, gain=1.0):
    """Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, 	ext{std}^2)` where

    .. math::
        	ext{std} = 	ext{gain} 	imes \sqrt{rac{2}{	ext{fan\_in} + 	ext{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init.xavier_normal_', 'xavier_normal_(tensor, gain=1.0)', {'_calculate_fan_in_and_fan_out': _calculate_fan_in_and_fan_out, 'math': math, '_no_grad_normal_': _no_grad_normal_, 'tensor': tensor, 'gain': gain}, 1)

def _calculate_correct_fan(tensor, mode):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init._calculate_correct_fan', '_calculate_correct_fan(tensor, mode)', {'_calculate_fan_in_and_fan_out': _calculate_fan_in_and_fan_out, 'tensor': tensor, 'mode': mode}, 1)

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    """Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-	ext{bound}, 	ext{bound})` where

    .. math::
        	ext{bound} = 	ext{gain} 	imes \sqrt{rac{3}{	ext{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only 
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init.kaiming_uniform_', "kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')", {'_calculate_correct_fan': _calculate_correct_fan, 'calculate_gain': calculate_gain, 'math': math, 'torch': torch, 'tensor': tensor, 'a': a, 'mode': mode, 'nonlinearity': nonlinearity}, 1)

def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    """Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    normal distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, 	ext{std}^2)` where

    .. math::
        	ext{std} = rac{	ext{gain}}{\sqrt{	ext{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only 
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_normal_(w, mode='fan_out', nonlinearity='relu')
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init.kaiming_normal_', "kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')", {'_calculate_correct_fan': _calculate_correct_fan, 'calculate_gain': calculate_gain, 'math': math, 'torch': torch, 'tensor': tensor, 'a': a, 'mode': mode, 'nonlinearity': nonlinearity}, 1)

def orthogonal_(tensor, gain=1):
    """Fills the input `Tensor` with a (semi) orthogonal matrix, as
    described in `Exact solutions to the nonlinear dynamics of learning in deep
    linear neural networks` - Saxe, A. et al. (2013). The input tensor must have
    at least 2 dimensions, and for tensors with more than 2 dimensions the
    trailing dimensions are flattened.

    Args:
        tensor: an n-dimensional `torch.Tensor`, where :math:`n \geq 2`
        gain: optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.orthogonal_(w)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init.orthogonal_', 'orthogonal_(tensor, gain=1)', {'torch': torch, 'tensor': tensor, 'gain': gain}, 1)

def sparse_(tensor, sparsity, std=0.01):
    """Fills the 2D input `Tensor` as a sparse matrix, where the
    non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init.sparse_', 'sparse_(tensor, sparsity, std=0.01)', {'math': math, 'torch': torch, 'tensor': tensor, 'sparsity': sparsity, 'std': std}, 1)

def _make_deprecate(meth):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.init._make_deprecate', '_make_deprecate(meth)', {'warnings': warnings, 'meth': meth}, 1)
uniform = _make_deprecate(uniform_)
normal = _make_deprecate(normal_)
constant = _make_deprecate(constant_)
eye = _make_deprecate(eye_)
dirac = _make_deprecate(dirac_)
xavier_uniform = _make_deprecate(xavier_uniform_)
xavier_normal = _make_deprecate(xavier_normal_)
kaiming_uniform = _make_deprecate(kaiming_uniform_)
kaiming_normal = _make_deprecate(kaiming_normal_)
orthogonal = _make_deprecate(orthogonal_)
sparse = _make_deprecate(sparse_)

