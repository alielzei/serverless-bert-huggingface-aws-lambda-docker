import torch
from ._functions import Scatter, Gather

def scatter(inputs, target_gpus, dim=0):
    """
    Slices tensors into approximately equal chunks and
    distributes them across given GPUs. Duplicates
    references to objects that are not tensors.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.parallel.scatter_gather.scatter', 'scatter(inputs, target_gpus, dim=0)', {'torch': torch, 'Scatter': Scatter, 'inputs': inputs, 'target_gpus': target_gpus, 'dim': dim}, 1)

def scatter_kwargs(inputs, kwargs, target_gpus, dim=0):
    """Scatter with support for kwargs dictionary"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.parallel.scatter_gather.scatter_kwargs', 'scatter_kwargs(inputs, kwargs, target_gpus, dim=0)', {'scatter': scatter, 'inputs': inputs, 'kwargs': kwargs, 'target_gpus': target_gpus, 'dim': dim}, 2)

def gather(outputs, target_device, dim=0):
    """
    Gathers tensors from different GPUs on a specified device
      (-1 means the CPU).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.parallel.scatter_gather.gather', 'gather(outputs, target_device, dim=0)', {'torch': torch, 'Gather': Gather, 'outputs': outputs, 'target_device': target_device, 'dim': dim}, 1)

