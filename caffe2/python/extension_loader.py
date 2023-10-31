from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import ctypes
import sys
_set_global_flags = (hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'))

@contextlib.contextmanager
def DlopenGuard(extra_flags=ctypes.RTLD_GLOBAL):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.extension_loader.DlopenGuard', 'DlopenGuard(extra_flags=ctypes.RTLD_GLOBAL)', {'_set_global_flags': _set_global_flags, 'sys': sys, 'contextlib': contextlib, 'extra_flags': extra_flags}, 0)

