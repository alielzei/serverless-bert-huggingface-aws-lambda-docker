from __future__ import unicode_literals
import inspect
import math
import numbers
from future.utils import PY2, PY3, exec_
if PY2:
    from collections import Mapping
else:
    from collections.abc import Mapping
if PY3:
    import builtins
    from collections.abc import Mapping
    
    def apply(f, *args, **kw):
        return f(*args, **kw)
    from past.builtins import str as oldstr
    
    def chr(i):
        """
        Return a byte-string of one character with ordinal i; 0 <= i <= 256
        """
        return oldstr(bytes((i, )))
    
    def cmp(x, y):
        """
        cmp(x, y) -> integer

        Return negative if x<y, zero if x==y, positive if x>y.
        Python2 had looser comparison allowing cmp None and non Numerical types and collections.
        Try to match the old behavior
        """
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('past.builtins.misc.cmp', 'cmp(x, y)', {'numbers': numbers, 'math': math, 'cmp': cmp, 'x': x, 'y': y}, 1)
    from sys import intern
    
    def oct(number):
        """oct(number) -> string

        Return the octal representation of an integer
        """
        return '0' + builtins.oct(number)[2:]
    raw_input = input
    try:
        from importlib import reload
    except ImportError:
        from imp import reload
    unicode = str
    unichr = chr
    xrange = range
else:
    import __builtin__
    from collections import Mapping
    apply = __builtin__.apply
    chr = __builtin__.chr
    cmp = __builtin__.cmp
    execfile = __builtin__.execfile
    intern = __builtin__.intern
    oct = __builtin__.oct
    raw_input = __builtin__.raw_input
    reload = __builtin__.reload
    unicode = __builtin__.unicode
    unichr = __builtin__.unichr
    xrange = __builtin__.xrange
if PY3:
    
    def execfile(filename, myglobals=None, mylocals=None):
        """
        Read and execute a Python script from a file in the given namespaces.
        The globals and locals are dictionaries, defaulting to the current
        globals and locals. If only globals is given, locals defaults to it.
        """
        import custom_funtemplate
        custom_funtemplate.rewrite_template('past.builtins.misc.execfile', 'execfile(filename, myglobals=None, mylocals=None)', {'inspect': inspect, 'Mapping': Mapping, 'exec_': exec_, 'filename': filename, 'myglobals': myglobals, 'mylocals': mylocals}, 0)
if PY3:
    __all__ = ['apply', 'chr', 'cmp', 'execfile', 'intern', 'raw_input', 'reload', 'unichr', 'unicode', 'xrange']
else:
    __all__ = []

