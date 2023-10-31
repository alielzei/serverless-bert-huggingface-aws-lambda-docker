import sys
import torch
from contextlib import contextmanager
from torch.backends import ContextProp, PropModule, __allow_nonbracketed_mutation

def is_available():
    """Returns whether PyTorch is built with MKL-DNN support."""
    return torch._C.has_mkldnn

def set_flags(_enabled):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.backends.mkldnn.__init__.set_flags', 'set_flags(_enabled)', {'torch': torch, '_enabled': _enabled}, 1)

@contextmanager
def flags(enabled=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.backends.mkldnn.__init__.flags', 'flags(enabled=False)', {'__allow_nonbracketed_mutation': __allow_nonbracketed_mutation, 'set_flags': set_flags, 'contextmanager': contextmanager, 'enabled': enabled}, 0)


class MkldnnModule(PropModule):
    
    def __init__(self, m, name):
        super(MkldnnModule, self).__init__(m, name)
    enabled = ContextProp(torch._C._get_mkldnn_enabled, torch._C._set_mkldnn_enabled)

sys.modules[__name__] = MkldnnModule(sys.modules[__name__], __name__)

