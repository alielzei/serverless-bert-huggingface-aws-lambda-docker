import warnings
import torch
from torch._six import inf

def clip_grad_norm_(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.utils.clip_grad.clip_grad_norm_', 'clip_grad_norm_(parameters, max_norm, norm_type=2)', {'torch': torch, 'inf': inf, 'parameters': parameters, 'max_norm': max_norm, 'norm_type': norm_type}, 1)

def clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    .. warning::
        This method is now deprecated in favor of
        :func:`torch.nn.utils.clip_grad_norm_`.
    """
    warnings.warn('torch.nn.utils.clip_grad_norm is now deprecated in favor of torch.nn.utils.clip_grad_norm_.', stacklevel=2)
    return clip_grad_norm_(parameters, max_norm, norm_type)

def clip_grad_value_(parameters, clip_value):
    """Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[	ext{-clip\_value}, 	ext{clip\_value}ight]`
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.nn.utils.clip_grad.clip_grad_value_', 'clip_grad_value_(parameters, clip_value)', {'torch': torch, 'parameters': parameters, 'clip_value': clip_value}, 0)

