"""
A resurrection of some old functions from Python 2 for use in Python 3. These
should be used sparingly, to help with porting efforts, since code using them
is no longer standard Python 3 code.

This module provides the following:

1. Implementations of these builtin functions which have no equivalent on Py3:

- apply
- chr
- cmp
- execfile

2. Aliases:

- intern <- sys.intern
- raw_input <- input
- reduce <- functools.reduce
- reload <- imp.reload
- unichr <- chr
- unicode <- str
- xrange <- range

3. List-producing versions of the corresponding Python 3 iterator-producing functions:

- filter
- map
- range
- zip

4. Forward-ported Py2 types:

- basestring
- dict
- str
- long
- unicode

"""

from future.utils import PY3
from past.builtins.noniterators import filter, map, range, reduce, zip
if PY3:
    from past.types import basestring, olddict as dict, oldstr as str, long, unicode
else:
    from __builtin__ import basestring, dict, str, long, unicode
from past.builtins.misc import apply, chr, cmp, execfile, intern, oct, raw_input, reload, unichr, unicode, xrange
from past import utils
if utils.PY3:
    __all__ = ['filter', 'map', 'range', 'reduce', 'zip', 'basestring', 'dict', 'str', 'long', 'unicode', 'apply', 'chr', 'cmp', 'execfile', 'intern', 'raw_input', 'reload', 'unichr', 'xrange']
else:
    __all__ = []

