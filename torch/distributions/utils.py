from functools import update_wrapper
from numbers import Number
import torch
import torch.nn.functional as F

def broadcast_all(*values):
    """
    Given a list of values (possibly containing numbers), returns a list where each
    value is broadcasted based on the following rules:
      - `torch.*Tensor` instances are broadcasted as per :ref:`_broadcasting-semantics`.
      - numbers.Number instances (scalars) are upcast to tensors having
        the same size and type as the first tensor passed to `values`.  If all the
        values are scalars, then they are upcasted to scalar Tensors.

    Args:
        values (list of `numbers.Number` or `torch.*Tensor`)

    Raises:
        ValueError: if any of the values is not a `numbers.Number` or
            `torch.*Tensor` instance
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.utils.broadcast_all', 'broadcast_all(*values)', {'torch': torch, 'Number': Number, 'values': values}, 1)

def _standard_normal(shape, dtype, device):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.utils._standard_normal', '_standard_normal(shape, dtype, device)', {'torch': torch, 'shape': shape, 'dtype': dtype, 'device': device}, 1)

def _sum_rightmost(value, dim):
    """
    Sum out ``dim`` many rightmost dimensions of a given tensor.

    Args:
        value (Tensor): A tensor of ``.dim()`` at least ``dim``.
        dim (int): The number of rightmost dims to sum out.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.utils._sum_rightmost', '_sum_rightmost(value, dim)', {'value': value, 'dim': dim}, 1)

def logits_to_probs(logits, is_binary=False):
    """
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.utils.logits_to_probs', 'logits_to_probs(logits, is_binary=False)', {'torch': torch, 'F': F, 'logits': logits, 'is_binary': is_binary}, 1)

def clamp_probs(probs):
    eps = torch.finfo(probs.dtype).eps
    return probs.clamp(min=eps, max=1 - eps)

def probs_to_logits(probs, is_binary=False):
    """
    Converts a tensor of probabilities into logits. For the binary case,
    this denotes the probability of occurrence of the event indexed by `1`.
    For the multi-dimensional case, the values along the last dimension
    denote the probabilities of occurrence of each of the events.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.utils.probs_to_logits', 'probs_to_logits(probs, is_binary=False)', {'clamp_probs': clamp_probs, 'torch': torch, 'probs': probs, 'is_binary': is_binary}, 1)


class lazy_property(object):
    """
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """
    
    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)
    
    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        with torch.enable_grad():
            value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value


