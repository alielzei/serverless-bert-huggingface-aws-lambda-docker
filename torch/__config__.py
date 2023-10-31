import torch

def show():
    """
    Return a human-readable string with descriptions of the
    configuration of PyTorch.
    """
    return torch._C._show_config()

def parallel_info():
    """Returns detailed string with parallelization settings"""
    return torch._C._parallel_info()

