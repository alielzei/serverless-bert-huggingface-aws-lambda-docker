import torch
import torch.cuda.comm as comm
from torch.cuda._utils import _get_device_index

def _is_script_module(module):
    import torch.jit
    return isinstance(module, torch.jit.ScriptModule)

def _is_script_method(module):
    import torch.jit
    return isinstance(module, torch._C.ScriptMethod)

def _init_script_module():
    import torch.jit
    return torch.jit.ScriptModule()

def _is_jit_enabled():
    import torch.jit
    return torch.jit._enabled

def _replicatable_module(module, memo=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.parallel.replicate._replicatable_module', '_replicatable_module(module, memo=None)', {'_is_jit_enabled': _is_jit_enabled, '_is_script_module': _is_script_module, '_replicatable_module': _replicatable_module, 'module': module, 'memo': memo}, 1)

def _broadcast_coalesced_reshape(tensors, devices, detach=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.parallel.replicate._broadcast_coalesced_reshape', '_broadcast_coalesced_reshape(tensors, devices, detach=False)', {'comm': comm, 'tensors': tensors, 'devices': devices, 'detach': detach}, 1)

def replicate(network, devices, detach=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.parallel.replicate.replicate', 'replicate(network, devices, detach=False)', {'_replicatable_module': _replicatable_module, '_get_device_index': _get_device_index, '_broadcast_coalesced_reshape': _broadcast_coalesced_reshape, '_is_script_module': _is_script_module, 'torch': torch, 'network': network, 'devices': devices, 'detach': detach}, 1)

