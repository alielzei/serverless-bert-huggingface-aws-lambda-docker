from __future__ import absolute_import, division, print_function, unicode_literals
import torch

def is_available():
    return hasattr(torch._C, '_c10d_init')
if (is_available() and not torch._C._c10d_init()):
    raise RuntimeError('Failed to initialize torch.distributed')
if is_available():
    from .distributed_c10d import *
    from .distributed_c10d import _backend

