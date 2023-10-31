import torch
from . import _lazy_init, _lazy_call, device_count, current_device
__all__ = ['get_rng_state', 'get_rng_state_all', 'set_rng_state', 'set_rng_state_all', 'manual_seed', 'manual_seed_all', 'seed', 'seed_all', 'initial_seed']

def get_rng_state(device='cuda'):
    """Returns the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (torch.device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.cuda.random.get_rng_state', "get_rng_state(device='cuda')", {'_lazy_init': _lazy_init, 'torch': torch, 'current_device': current_device, 'device': device}, 1)

def get_rng_state_all():
    """Returns a tuple of ByteTensor representing the random number states of all devices."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.cuda.random.get_rng_state_all', 'get_rng_state_all()', {'device_count': device_count, 'get_rng_state': get_rng_state}, 1)

def set_rng_state(new_state, device='cuda'):
    """Sets the random number generator state of the specified GPU.

    Args:
        new_state (torch.ByteTensor): The desired state
        device (torch.device or int, optional): The device to set the RNG state.
            Default: ``'cuda'`` (i.e., ``torch.device('cuda')``, the current CUDA device).
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.cuda.random.set_rng_state', "set_rng_state(new_state, device='cuda')", {'torch': torch, 'current_device': current_device, '_lazy_call': _lazy_call, 'new_state': new_state, 'device': device}, 0)

def set_rng_state_all(new_states):
    """Sets the random number generator state of all devices.

    Args:
        new_state (tuple of torch.ByteTensor): The desired state for each device"""
    for (i, state) in enumerate(new_states):
        set_rng_state(state, i)

def manual_seed(seed):
    """Sets the seed for generating random numbers for the current GPU.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.

    .. warning::
        If you are working with a multi-GPU model, this function is insufficient
        to get determinism.  To seed all GPUs, use :func:`manual_seed_all`.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.cuda.random.manual_seed', 'manual_seed(seed)', {'current_device': current_device, 'torch': torch, '_lazy_call': _lazy_call, 'seed': seed}, 0)

def manual_seed_all(seed):
    """Sets the seed for generating random numbers on all GPUs.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    Args:
        seed (int): The desired seed.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.cuda.random.manual_seed_all', 'manual_seed_all(seed)', {'device_count': device_count, 'torch': torch, '_lazy_call': _lazy_call, 'seed': seed}, 0)

def seed():
    """Sets the seed for generating random numbers to a random number for the current GPU.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.

    .. warning::
        If you are working with a multi-GPU model, this function will only initialize
        the seed on one GPU.  To initialize all GPUs, use :func:`seed_all`.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.cuda.random.seed', 'seed()', {'current_device': current_device, 'torch': torch, '_lazy_call': _lazy_call}, 0)

def seed_all():
    """Sets the seed for generating random numbers to a random number on all GPUs.
    It's safe to call this function if CUDA is not available; in that
    case, it is silently ignored.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.cuda.random.seed_all', 'seed_all()', {'device_count': device_count, 'torch': torch, '_lazy_call': _lazy_call}, 0)

def initial_seed():
    """Returns the current random seed of the current GPU.

    .. warning::
        This function eagerly initializes CUDA.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.cuda.random.initial_seed', 'initial_seed()', {'_lazy_init': _lazy_init, 'current_device': current_device, 'torch': torch}, 1)

