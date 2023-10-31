"""
``torch.autograd`` provides classes and functions implementing automatic
differentiation of arbitrary scalar valued functions. It requires minimal
changes to the existing code - you only need to declare :class:`Tensor` s
for which gradients should be computed with the ``requires_grad=True`` keyword.
"""

import torch
import warnings
from .variable import Variable
from .function import Function, NestedIOFunction
from .gradcheck import gradcheck, gradgradcheck
from .grad_mode import no_grad, enable_grad, set_grad_enabled
from .anomaly_mode import detect_anomaly, set_detect_anomaly
from . import profiler
from . import functional
__all__ = ['Variable', 'Function', 'backward', 'grad_mode']

def _make_grads(outputs, grads):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.__init__._make_grads', '_make_grads(outputs, grads)', {'torch': torch, 'outputs': outputs, 'grads': grads}, 1)

def backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None):
    """Computes the sum of gradients of given tensors w.r.t. graph leaves.

    The graph is differentiated using the chain rule. If any of ``tensors``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, then the Jacobian-vector product would be computed, in this
    case the function additionally requires specifying ``grad_tensors``.
    It should be a sequence of matching length, that contains the "vector"
    in the Jacobian-vector product, usually the gradient of the differentiated
    function w.r.t. corresponding tensors (``None`` is an acceptable value for
    all tensors that don't need gradient tensors).

    This function accumulates gradients in the leaves - you might need to zero
    them before calling it.

    Arguments:
        tensors (sequence of Tensor): Tensors of which the derivative will be
            computed.
        grad_tensors (sequence of (Tensor or None)): The "vector" in the Jacobian-vector
            product, usually gradients w.r.t. each element of corresponding tensors.
            None values can be specified for scalar Tensors or ones that don't require
            grad. If a None value would be acceptable for all grad_tensors, then this
            argument is optional.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to ``False``.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.autograd.__init__.backward', 'backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None)', {'warnings': warnings, 'torch': torch, '_make_grads': _make_grads, 'Variable': Variable, 'tensors': tensors, 'grad_tensors': grad_tensors, 'retain_graph': retain_graph, 'create_graph': create_graph, 'grad_variables': grad_variables}, 0)

def grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False):
    """Computes and returns the sum of gradients of outputs w.r.t. the inputs.

    ``grad_outputs`` should be a sequence of length matching ``output``
    containing the "vector" in Jacobian-vector product, usually the pre-computed
    gradients w.r.t. each of the outputs. If an output doesn't require_grad,
    then the gradient can be ``None``).

    If ``only_inputs`` is ``True``, the function will only return a list of gradients
    w.r.t the specified inputs. If it's ``False``, then gradient w.r.t. all remaining
    leaves will still be computed, and will be accumulated into their ``.grad``
    attribute.

    Arguments:
        outputs (sequence of Tensor): outputs of the differentiated function.
        inputs (sequence of Tensor): Inputs w.r.t. which the gradient will be
            returned (and not accumulated into ``.grad``).
        grad_outputs (sequence of Tensor): The "vector" in the Jacobian-vector product.
            Usually gradients w.r.t. each output. None values can be specified for scalar
            Tensors or ones that don't require grad. If a None value would be acceptable
            for all grad_tensors, then this argument is optional. Default: None.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Default: ``False``.
        allow_unused (bool, optional): If ``False``, specifying inputs that were not
            used when computing outputs (and therefore their grad is always zero)
            is an error. Defaults to ``False``.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.__init__.grad', 'grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)', {'warnings': warnings, 'torch': torch, '_make_grads': _make_grads, 'Variable': Variable, 'outputs': outputs, 'inputs': inputs, 'grad_outputs': grad_outputs, 'retain_graph': retain_graph, 'create_graph': create_graph, 'only_inputs': only_inputs, 'allow_unused': allow_unused}, 1)

def _is_checkpoint_valid():
    return Variable._execution_engine.is_checkpoint_valid()

def variable(*args, **kwargs):
    warnings.warn('torch.autograd.variable(...) is deprecated, use torch.tensor(...) instead')
    return torch.tensor(*args, **kwargs)
if not torch._C._autograd_init():
    raise RuntimeError('autograd initialization failed')

