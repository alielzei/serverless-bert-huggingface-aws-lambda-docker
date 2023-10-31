import torch._C
import contextlib
import ctypes
import os
import sys
import types
import torch.jit
import torch._utils_internal
_SET_GLOBAL_FLAGS = (hasattr(sys, 'getdlopenflags') and hasattr(sys, 'setdlopenflags'))

@contextlib.contextmanager
def dl_open_guard():
    """
    Context manager to set the RTLD_GLOBAL dynamic linker flag while we open a
    shared library to load custom operators.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch._ops.dl_open_guard', 'dl_open_guard()', {'_SET_GLOBAL_FLAGS': _SET_GLOBAL_FLAGS, 'sys': sys, 'ctypes': ctypes, 'contextlib': contextlib}, 0)


class _OpNamespace(types.ModuleType):
    """
    An op namespace to dynamically bind Operators into Python.

    Say a user has created a custom Operator called "my_namespace::my_op". To
    call this op, the user will write torch.ops.my_namespace.my_op(...).
    At startup, this operation will not yet be bound into Python. Instead, the
    following sequence of magic tricks will occur:
    1. `torch.ops.my_namespace` will invoke the `__getattr__` magic method
       on the `torch.ops` object, which will create a new `_OpNamespace`
       object called `my_namespace` and set it as an attribute on the `ops`
       object.
    2. `torch.ops.my_namespace.my_op` will then invoke `__getattr__` on
       the `my_namespace` object, which will retrieve the operation via
       `torch.get_operation`, a function bound from C++, and then in a similar
       fashion bind this new object onto the `my_namespace` object.
    3. `torch.ops.my_namespace.my_op(...)` then calls this new operation
        and subsequent accesses will incur no further lookup (the namespace and
        operation will already exist).
    """
    
    def __init__(self, name):
        super(_OpNamespace, self).__init__('torch.ops.' + name)
        self.name = name
    
    def __getattr__(self, op_name):
        qualified_op_name = '{}::{}'.format(self.name, op_name)
        op = torch._C._jit_get_operation(qualified_op_name)
        torch.jit._register_builtin(op, qualified_op_name)
        setattr(self, op_name, op)
        op.__module__ = self.__module__ + '.' + self.name
        return op



class _Ops(types.ModuleType):
    __file__ = os.path.join(os.path.dirname(__file__), '_ops.py')
    
    def __init__(self):
        super(_Ops, self).__init__('torch.ops')
        self.loaded_libraries = set()
    
    def __getattr__(self, name):
        namespace = _OpNamespace(name)
        setattr(self, name, namespace)
        return namespace
    
    def load_library(self, path):
        """
        Loads a shared library from the given path into the current process.

        The library being loaded may run global initialization code to register
        custom operators with the PyTorch JIT runtime. This allows dynamically
        loading custom operators. For this, you should compile your operator
        and the static registration code into a shared library object, and then
        call ``torch.ops.load_library('path/to/libcustom.so')`` to load the
        shared object.

        After the library is loaded, it is added to the
        ``torch.ops.loaded_libraries`` attribute, a set that may be inspected
        for the paths of all libraries loaded using this function.

        Arguments:
            path (str): A path to a shared library to load.
        """
        path = torch._utils_internal.resolve_library_path(path)
        with dl_open_guard():
            ctypes.CDLL(path)
        self.loaded_libraries.add(path)

ops = _Ops()

