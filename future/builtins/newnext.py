"""
This module provides a newnext() function in Python 2 that mimics the
behaviour of ``next()`` in Python 3, falling back to Python 2's behaviour for
compatibility if this fails.

``newnext(iterator)`` calls the iterator's ``__next__()`` method if it exists. If this
doesn't exist, it falls back to calling a ``next()`` method.

For example:

    >>> class Odds(object):
    ...     def __init__(self, start=1):
    ...         self.value = start - 2
    ...     def __next__(self):                 # note the Py3 interface
    ...         self.value += 2
    ...         return self.value
    ...     def __iter__(self):
    ...         return self
    ...
    >>> iterator = Odds()
    >>> next(iterator)
    1
    >>> next(iterator)
    3

If you are defining your own custom iterator class as above, it is preferable
to explicitly decorate the class with the @implements_iterator decorator from
``future.utils`` as follows:

    >>> @implements_iterator
    ... class Odds(object):
    ...     # etc
    ...     pass

This next() function is primarily for consuming iterators defined in Python 3
code elsewhere that we would like to run on Python 2 or 3.
"""

_builtin_next = next
_SENTINEL = object()

def newnext(iterator, default=_SENTINEL):
    """
    next(iterator[, default])

    Return the next item from the iterator. If default is given and the iterator
    is exhausted, it is returned instead of raising StopIteration.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.builtins.newnext.newnext', 'newnext(iterator, default=_SENTINEL)', {'iterator': iterator, 'default': default, '_SENTINEL': _SENTINEL}, 1)
__all__ = ['newnext']

