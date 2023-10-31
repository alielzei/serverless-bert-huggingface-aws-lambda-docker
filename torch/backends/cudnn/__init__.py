import sys
import torch
import warnings
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation
try:
    from torch._C import _cudnn
except ImportError:
    _cudnn = None
__cudnn_version = None
if _cudnn is not None:
    
    def _init():
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('torch.backends.cudnn.__init__._init', '_init()', {'_cudnn': _cudnn}, 1)
else:
    
    def _init():
        return False

def version():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.backends.cudnn.__init__.version', 'version()', {'_init': _init, '__cudnn_version': __cudnn_version}, 1)
CUDNN_TENSOR_TYPES = {'torch.cuda.HalfTensor', 'torch.cuda.FloatTensor', 'torch.cuda.DoubleTensor'}

def is_available():
    """Returns a bool indicating if CUDNN is currently available."""
    return torch._C.has_cudnn

def is_acceptable(tensor):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.backends.cudnn.__init__.is_acceptable', 'is_acceptable(tensor)', {'torch': torch, 'CUDNN_TENSOR_TYPES': CUDNN_TENSOR_TYPES, 'is_available': is_available, 'warnings': warnings, '_init': _init, 'sys': sys, 'tensor': tensor}, 1)
_handles = {}
verbose = False

def set_flags(_enabled, _benchmark, _deterministic, _verbose):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.backends.cudnn.__init__.set_flags', 'set_flags(_enabled, _benchmark, _deterministic, _verbose)', {'torch': torch, '_enabled': _enabled, '_benchmark': _benchmark, '_deterministic': _deterministic, '_verbose': _verbose}, 1)

@contextmanager
def flags(enabled=False, benchmark=False, deterministic=False, verbose=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.backends.cudnn.__init__.flags', 'flags(enabled=False, benchmark=False, deterministic=False, verbose=False)', {'__allow_nonbracketed_mutation': __allow_nonbracketed_mutation, 'set_flags': set_flags, 'contextmanager': contextmanager, 'enabled': enabled, 'benchmark': benchmark, 'deterministic': deterministic, 'verbose': verbose}, 0)


class CudnnModule(PropModule):
    
    def __init__(self, m, name):
        super(CudnnModule, self).__init__(m, name)
    enabled = ContextProp(torch._C._get_cudnn_enabled, torch._C._set_cudnn_enabled)
    deterministic = ContextProp(torch._C._get_cudnn_deterministic, torch._C._set_cudnn_deterministic)
    benchmark = ContextProp(torch._C._get_cudnn_benchmark, torch._C._set_cudnn_benchmark)

sys.modules[__name__] = CudnnModule(sys.modules[__name__], __name__)

