"""
This module is designed to be used as follows::

    from past.builtins.noniterators import filter, map, range, reduce, zip

And then, for example::

    assert isinstance(range(5), list)

The list-producing functions this brings in are::

- ``filter``
- ``map``
- ``range``
- ``reduce``
- ``zip``

"""

from __future__ import division, absolute_import, print_function
from itertools import chain, starmap
import itertools
from past.types import basestring
from past.utils import PY3

def flatmap(f, items):
    return chain.from_iterable(map(f, items))
if PY3:
    import builtins
    
    def oldfilter(*args):
        """
        filter(function or None, sequence) -> list, tuple, or string

        Return those items of sequence for which function(item) is true.
        If function is None, return the items that are true.  If sequence
        is a tuple or string, return the same type, else return a list.
        """
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('past.builtins.noniterators.oldfilter', 'oldfilter(*args)', {'basestring': basestring, 'builtins': builtins, 'args': args}, 1)
    
    def oldmap(func, *iterables):
        """
        map(function, sequence[, sequence, ...]) -> list

        Return a list of the results of applying the function to the
        items of the argument sequence(s).  If more than one sequence is
        given, the function is called with an argument list consisting of
        the corresponding item of each sequence, substituting None for
        missing values when not all sequences have the same length.  If
        the function is None, return a list of the items of the sequence
        (or a list of tuples if more than one sequence).

        Test cases:
        >>> oldmap(None, 'hello world')
        ['h', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd']

        >>> oldmap(None, range(4))
        [0, 1, 2, 3]

        More test cases are in test_past.test_builtins.
        """
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('past.builtins.noniterators.oldmap', 'oldmap(func, *iterables)', {'itertools': itertools, 'starmap': starmap, 'chain': chain, 'func': func, 'iterables': iterables}, 1)
    
    def oldrange(*args, **kwargs):
        return list(builtins.range(*args, **kwargs))
    
    def oldzip(*args, **kwargs):
        return list(builtins.zip(*args, **kwargs))
    filter = oldfilter
    map = oldmap
    range = oldrange
    from functools import reduce
    zip = oldzip
    __all__ = ['filter', 'map', 'range', 'reduce', 'zip']
else:
    import __builtin__
    filter = __builtin__.filter
    map = __builtin__.map
    range = __builtin__.range
    reduce = __builtin__.reduce
    zip = __builtin__.zip
    __all__ = []

