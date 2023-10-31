"""
A module that brings in equivalents of the new and modified Python 3
builtins into Py2. Has no effect on Py3.

See the docs `here <http://python-future.org/what-else.html>`_
(``docs/what-else.rst``) for more information.

"""

from future.builtins.iterators import filter, map, zip
from future.builtins.misc import ascii, chr, hex, input, isinstance, next, oct, open, pow, round, super, max, min
from future.utils import PY3
if PY3:
    import builtins
    bytes = builtins.bytes
    dict = builtins.dict
    int = builtins.int
    list = builtins.list
    object = builtins.object
    range = builtins.range
    str = builtins.str
    __all__ = []
else:
    from future.types import newbytes as bytes, newdict as dict, newint as int, newlist as list, newobject as object, newrange as range, newstr as str
from future import utils
if not utils.PY3:
    __all__ = ['filter', 'map', 'zip', 'ascii', 'chr', 'hex', 'input', 'next', 'oct', 'open', 'pow', 'round', 'super', 'bytes', 'dict', 'int', 'list', 'object', 'range', 'str', 'max', 'min']
else:
    __all__ = []

