import threading
import torch
from torch.cuda._utils import _get_device_index
from torch._utils import ExceptionWrapper

def get_a_var(obj):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.parallel.parallel_apply.get_a_var', 'get_a_var(obj)', {'torch': torch, 'get_a_var': get_a_var, 'obj': obj}, 1)

def parallel_apply(modules, inputs, kwargs_tup=None, devices=None):
    """Applies each `module` in :attr:`modules` in parallel on arguments
    contained in :attr:`inputs` (positional) and :attr:`kwargs_tup` (keyword)
    on each of :attr:`devices`.

    Args:
        modules (Module): modules to be parallelized
        inputs (tensor): inputs to the modules
        devices (list of int or torch.device): CUDA devices

    :attr:`modules`, :attr:`inputs`, :attr:`kwargs_tup` (if given), and
    :attr:`devices` (if given) should all have same length. Moreover, each
    element of :attr:`inputs` can either be a single object as the only argument
    to a module, or a collection of positional arguments.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.parallel.parallel_apply.parallel_apply', 'parallel_apply(modules, inputs, kwargs_tup=None, devices=None)', {'_get_device_index': _get_device_index, 'threading': threading, 'torch': torch, 'get_a_var': get_a_var, 'ExceptionWrapper': ExceptionWrapper, 'modules': modules, 'inputs': inputs, 'kwargs_tup': kwargs_tup, 'devices': devices}, 1)

