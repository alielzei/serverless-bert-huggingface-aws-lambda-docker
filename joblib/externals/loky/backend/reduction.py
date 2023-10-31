import copyreg
import io
import functools
import types
import sys
import os
from multiprocessing import util
from pickle import loads, HIGHEST_PROTOCOL
_dispatch_table = {}

def register(type_, reduce_function):
    _dispatch_table[type_] = reduce_function

def _reduce_method(m):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.reduction._reduce_method', '_reduce_method(m)', {'m': m}, 2)


class _C:
    
    def f(self):
        pass
    
    @classmethod
    def h(cls):
        pass

register(type(_C().f), _reduce_method)
register(type(_C.h), _reduce_method)
if not hasattr(sys, 'pypy_version_info'):
    
    def _reduce_method_descriptor(m):
        return (getattr, (m.__objclass__, m.__name__))
    register(type(list.append), _reduce_method_descriptor)
    register(type(int.__add__), _reduce_method_descriptor)

def _reduce_partial(p):
    return (_rebuild_partial, (p.func, p.args, (p.keywords or {})))

def _rebuild_partial(func, args, keywords):
    return functools.partial(func, *args, **keywords)
register(functools.partial, _reduce_partial)
if sys.platform != 'win32':
    from ._posix_reduction import _mk_inheritable
else:
    from . import _win_reduction
try:
    from joblib.externals import cloudpickle
    DEFAULT_ENV = 'cloudpickle'
except ImportError:
    DEFAULT_ENV = 'pickle'
ENV_LOKY_PICKLER = os.environ.get('LOKY_PICKLER', DEFAULT_ENV)
_LokyPickler = None
_loky_pickler_name = None

def set_loky_pickler(loky_pickler=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.reduction.set_loky_pickler', 'set_loky_pickler(loky_pickler=None)', {'ENV_LOKY_PICKLER': ENV_LOKY_PICKLER, 'util': util, 'types': types, 'copyreg': copyreg, '_dispatch_table': _dispatch_table, 'loky_pickler': loky_pickler}, 1)

def get_loky_pickler_name():
    global _loky_pickler_name
    return _loky_pickler_name

def get_loky_pickler():
    global _LokyPickler
    return _LokyPickler
set_loky_pickler()

def dump(obj, file, reducers=None, protocol=None):
    """Replacement for pickle.dump() using _LokyPickler."""
    global _LokyPickler
    _LokyPickler(file, reducers=reducers, protocol=protocol).dump(obj)

def dumps(obj, reducers=None, protocol=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.reduction.dumps', 'dumps(obj, reducers=None, protocol=None)', {'io': io, 'dump': dump, 'obj': obj, 'reducers': reducers, 'protocol': protocol}, 1)
__all__ = ['dump', 'dumps', 'loads', 'register', 'set_loky_pickler']
if sys.platform == 'win32':
    from multiprocessing.reduction import duplicate
    __all__ += ['duplicate']

