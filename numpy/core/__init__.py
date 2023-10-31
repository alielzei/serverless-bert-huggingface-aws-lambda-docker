"""
Contains the core of NumPy: ndarray, ufuncs, dtypes, etc.

Please note that this module is private.  All functions and objects
are available in the main ``numpy`` namespace - use that instead.

"""

from numpy.version import version as __version__
import os
import warnings
env_added = []
for envkey in ['OPENBLAS_MAIN_FREE', 'GOTOBLAS_MAIN_FREE']:
    if envkey not in os.environ:
        os.environ[envkey] = '1'
        env_added.append(envkey)
try:
    try:
        from . import multiarray
    except ImportError as exc:
        import sys
        msg = '\n\nIMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!\n\nImporting the numpy C-extensions failed. This error can happen for\nmany reasons, often due to issues with your setup or how NumPy was\ninstalled.\n\nWe have compiled some common reasons and troubleshooting tips at:\n\n    https://numpy.org/devdocs/user/troubleshooting-importerror.html\n\nPlease note and check the following:\n\n  * The Python version is: Python%d.%d from "%s"\n  * The NumPy version is: "%s"\n\nand make sure that they are the versions you expect.\nPlease carefully study the documentation linked above for further help.\n\nOriginal error was: %s\n' % (sys.version_info[0], sys.version_info[1], sys.executable, __version__, exc)
        raise ImportError(msg)
finally:
    for envkey in env_added:
        del os.environ[envkey]
del envkey
del env_added
del os
from . import umath
if not ((hasattr(multiarray, '_multiarray_umath') and hasattr(umath, '_multiarray_umath'))):
    import sys
    path = sys.modules['numpy'].__path__
    msg = 'Something is wrong with the numpy installation. While importing we detected an older version of numpy in {}. One method of fixing this is to repeatedly uninstall numpy until none is found, then reinstall this version.'
    raise ImportError(msg.format(path))
from . import numerictypes as nt
multiarray.set_typeDict(nt.sctypeDict)
from . import numeric
from .numeric import *
from . import fromnumeric
from .fromnumeric import *
from . import defchararray as char
from . import records
from . import records as rec
from .records import record, recarray, format_parser
from .memmap import *
from .defchararray import chararray
from . import function_base
from .function_base import *
from . import _machar
from ._machar import *
from . import getlimits
from .getlimits import *
from . import shape_base
from .shape_base import *
from . import einsumfunc
from .einsumfunc import *
del nt
from .fromnumeric import amax as max, amin as min, round_ as round
from .numeric import absolute as abs
from . import _add_newdocs
from . import _add_newdocs_scalars
from . import _dtype_ctypes
from . import _internal
from . import _dtype
from . import _methods
__all__ = ['char', 'rec', 'memmap']
__all__ += numeric.__all__
__all__ += ['record', 'recarray', 'format_parser']
__all__ += ['chararray']
__all__ += function_base.__all__
__all__ += getlimits.__all__
__all__ += shape_base.__all__
__all__ += einsumfunc.__all__

def _ufunc_reconstruct(module, name):
    mod = __import__(module, fromlist=[name])
    return getattr(mod, name)

def _ufunc_reduce(func):
    return func.__name__

def _DType_reconstruct(scalar_type):
    return type(dtype(scalar_type))

def _DType_reduce(DType):
    if DType is dtype:
        return 'dtype'
    scalar_type = DType.type
    return (_DType_reconstruct, (scalar_type, ))

def __getattr__(name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.__init__.__getattr__', '__getattr__(name)', {'warnings': warnings, '_machar': _machar, '__name__': __name__, 'name': name}, 1)
import copyreg
copyreg.pickle(ufunc, _ufunc_reduce)
copyreg.pickle(type(dtype), _DType_reduce, _DType_reconstruct)
del copyreg
del _ufunc_reduce
del _DType_reduce
from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester

