import itertools
import sys
import types
import inspect
PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
PY37 = (sys.version_info[0] == 3 and sys.version_info[1] == 7)
if PY2:
    import __builtin__ as builtins
elif PY3:
    import builtins
if PY2:
    inf = float('inf')
    nan = float('nan')
else:
    import math
    inf = math.inf
    nan = math.nan
if PY2:
    string_classes = basestring
else:
    string_classes = (str, bytes)
if PY2:
    int_classes = (int, long)
else:
    int_classes = int
if PY2:
    FileNotFoundError = IOError
else:
    FileNotFoundError = builtins.FileNotFoundError
if PY2:
    import Queue as queue
else:
    import queue

def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    
    
    class metaclass(meta):
        
        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)
    
    return type.__new__(metaclass, 'temporary_class', (), {})
if hasattr(itertools, 'imap'):
    imap = itertools.imap
else:
    imap = map
if PY3:
    import builtins
    exec_ = getattr(builtins, 'exec')
else:
    
    def exec_(_code_, _globs_=None, _locs_=None):
        """Execute code in a namespace."""
        if _globs_ is None:
            frame = sys._getframe(1)
            _globs_ = frame.f_globals
            if _locs_ is None:
                _locs_ = frame.f_locals
            del frame
        elif _locs_ is None:
            _locs_ = _globs_
        exec('exec _code_ in _globs_, _locs_')
if sys.version_info[:2] == (3, 2):
    exec_('def raise_from(value, from_value):\n    try:\n        if from_value is None:\n            raise value\n        raise value from from_value\n    finally:\n        value = None\n')
elif sys.version_info[:2] > (3, 2):
    exec_('def raise_from(value, from_value):\n    try:\n        raise value from from_value\n    finally:\n        value = None\n')
else:
    
    def raise_from(value, from_value):
        raise value
if PY2:
    import collections
    container_abcs = collections
elif PY3:
    import collections.abc
    container_abcs = collections.abc
if PY2:
    
    def get_function_from_type(cls, name):
        method = getattr(cls, name, None)
        return getattr(method, '__func__', None)
elif PY3:
    
    def get_function_from_type(cls, name):
        return getattr(cls, name, None)
if PY2:
    import StringIO
    StringIO = StringIO.StringIO
elif PY3:
    import io
    StringIO = io.StringIO

def istuple(obj):
    t = type(obj)
    return (isinstance(obj, tuple) or t.__module__ == 'torch.return_types')

def bind_method(fn, obj, obj_type):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._six.bind_method', 'bind_method(fn, obj, obj_type)', {'PY2': PY2, 'inspect': inspect, 'types': types, 'fn': fn, 'obj': obj, 'obj_type': obj_type}, 1)

