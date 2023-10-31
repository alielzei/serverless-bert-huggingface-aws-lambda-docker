import contextlib
import warnings
from torch._C import default_generator

def set_rng_state(new_state):
    """Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
    """
    default_generator.set_state(new_state)

def get_rng_state():
    """Returns the random number generator state as a `torch.ByteTensor`."""
    return default_generator.get_state()

def manual_seed(seed):
    """Sets the seed for generating random numbers. Returns a
    `torch.Generator` object.

    Args:
        seed (int): The desired seed.
    """
    seed = int(seed)
    import torch.cuda
    if not torch.cuda._is_in_bad_fork():
        torch.cuda.manual_seed_all(seed)
    return default_generator.manual_seed(seed)

def seed():
    """Sets the seed for generating random numbers to a non-deterministic
    random number. Returns a 64 bit number used to seed the RNG.
    """
    seed = default_generator.seed()
    import torch.cuda
    if not torch.cuda._is_in_bad_fork():
        torch.cuda.manual_seed_all(seed)
    return seed

def initial_seed():
    """Returns the initial seed for generating random numbers as a
    Python `long`.
    """
    return default_generator.initial_seed()
_fork_rng_warned_already = False

@contextlib.contextmanager
def fork_rng(devices=None, enabled=True, _caller='fork_rng', _devices_kw='devices'):
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.

    Arguments:
        devices (iterable of CUDA IDs): CUDA devices for which to fork
            the RNG.  CPU RNG state is always forked.  By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.random.fork_rng', "fork_rng(devices=None, enabled=True, _caller='fork_rng', _devices_kw='devices')", {'torch': torch, 'warnings': warnings, 'contextlib': contextlib, 'devices': devices, 'enabled': enabled, '_caller': _caller, '_devices_kw': _devices_kw}, 1)

