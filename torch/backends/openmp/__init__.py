import torch

def is_available():
    """Returns whether PyTorch is built with OpenMP support."""
    return torch._C.has_openmp

