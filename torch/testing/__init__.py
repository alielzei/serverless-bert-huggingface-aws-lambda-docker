"""
The testing package contains testing-specific utilities.
"""

import torch
import random
FileCheck = torch._C.FileCheck
__all__ = ['assert_allclose', 'make_non_contiguous', 'rand_like', 'randn_like']
rand_like = torch.rand_like
randn_like = torch.randn_like

def assert_allclose(actual, expected, rtol=None, atol=None, equal_nan=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing.__init__.assert_allclose', 'assert_allclose(actual, expected, rtol=None, atol=None, equal_nan=True)', {'torch': torch, '_get_default_tolerance': _get_default_tolerance, 'actual': actual, 'expected': expected, 'rtol': rtol, 'atol': atol, 'equal_nan': equal_nan}, 1)

def make_non_contiguous(tensor):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing.__init__.make_non_contiguous', 'make_non_contiguous(tensor)', {'random': random, 'torch': torch, 'tensor': tensor}, 1)

def get_all_dtypes():
    return [torch.uint8, torch.bool, torch.int8, torch.int16, torch.int32, torch.int64, torch.float16, torch.float32, torch.float64, torch.bfloat16]

def get_all_math_dtypes(device):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing.__init__.get_all_math_dtypes', 'get_all_math_dtypes(device)', {'torch': torch, 'device': device}, 1)

def get_all_device_types():
    return (['cpu'] if not torch.cuda.is_available() else ['cpu', 'cuda'])
_default_tolerances = {'float64': (1e-05, 1e-08), 'float32': (0.0001, 1e-05), 'float16': (0.001, 0.001)}

def _get_default_tolerance(a, b=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing.__init__._get_default_tolerance', '_get_default_tolerance(a, b=None)', {'_default_tolerances': _default_tolerances, '_get_default_tolerance': _get_default_tolerance, 'a': a, 'b': b}, 1)

