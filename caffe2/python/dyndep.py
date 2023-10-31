from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import ctypes
import os
from threading import Lock
from caffe2.python import core, extension_loader

def InitOpsLibrary(name):
    """Loads a dynamic library that contains custom operators into Caffe2.

    Since Caffe2 uses static variable registration, you can optionally load a
    separate .so file that contains custom operators and registers that into
    the caffe2 core binary. In C++, this is usually done by either declaring
    dependency during compilation time, or via dynload. This allows us to do
    registration similarly on the Python side.

    Args:
        name: a name that ends in .so, such as "my_custom_op.so". Otherwise,
            the command will simply be ignored.
    Returns:
        None
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.dyndep.InitOpsLibrary', 'InitOpsLibrary(name)', {'os': os, '_init_impl': _init_impl, 'name': name}, 1)
_IMPORTED_DYNDEPS = set()
dll_lock = Lock()

def GetImportedOpsLibraries():
    return _IMPORTED_DYNDEPS

def _init_impl(path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.dyndep._init_impl', '_init_impl(path)', {'dll_lock': dll_lock, '_IMPORTED_DYNDEPS': _IMPORTED_DYNDEPS, 'extension_loader': extension_loader, 'ctypes': ctypes, 'core': core, 'path': path}, 0)

