"""
Miscellaneous function (re)definitions from the Py3.4+ standard library
for Python 2.6/2.7.

- math.ceil                (for Python 2.7)
- collections.OrderedDict  (for Python 2.6)
- collections.Counter      (for Python 2.6)
- collections.ChainMap     (for all versions prior to Python 3.3)
- itertools.count          (for Python 2.6, with step parameter)
- subprocess.check_output  (for Python 2.6)
- reprlib.recursive_repr   (for Python 2.6+)
- functools.cmp_to_key     (for Python 2.6)
"""

from __future__ import absolute_import
import subprocess
from math import ceil as oldceil
from operator import itemgetter as _itemgetter, eq as _eq
import sys
import heapq as _heapq
from _weakref import proxy as _proxy
from itertools import repeat as _repeat, chain as _chain, starmap as _starmap
from socket import getaddrinfo, SOCK_STREAM, error, socket
from future.utils import iteritems, itervalues, PY2, PY26, PY3
if PY2:
    from collections import Mapping, MutableMapping
else:
    from collections.abc import Mapping, MutableMapping

def ceil(x):
    """
    Return the ceiling of x as an int.
    This is the smallest integral value >= x.
    """
    return int(oldceil(x))
from itertools import islice
if PY26:
    
    def count(start=0, step=1):
        while True:
            yield start
            start += step
else:
    from itertools import count
if PY3:
    try:
        from _thread import get_ident
    except ImportError:
        from _dummy_thread import get_ident
else:
    try:
        from thread import get_ident
    except ImportError:
        from dummy_thread import get_ident

def recursive_repr(fillvalue='...'):
    """Decorator to make a repr function return fillvalue for a recursive call"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.misc.recursive_repr', "recursive_repr(fillvalue='...')", {'get_ident': get_ident, 'fillvalue': fillvalue}, 1)


class _Link(object):
    __slots__ = ('prev', 'next', 'key', '__weakref__')



class OrderedDict(dict):
    """Dictionary that remembers insertion order"""
    
    def __init__(*args, **kwds):
        """Initialize an ordered dictionary.  The signature is the same as
        regular dictionaries, but keyword arguments are not recommended because
        their insertion order is arbitrary.

        """
        if not args:
            raise TypeError("descriptor '__init__' of 'OrderedDict' object needs an argument")
        self = args[0]
        args = args[1:]
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        try:
            self.__root
        except AttributeError:
            self.__hardroot = _Link()
            self.__root = root = _proxy(self.__hardroot)
            root.prev = root.next = root
            self.__map = {}
        self.__update(*args, **kwds)
    
    def __setitem__(self, key, value, dict_setitem=dict.__setitem__, proxy=_proxy, Link=_Link):
        """od.__setitem__(i, y) <==> od[i]=y"""
        if key not in self:
            self.__map[key] = link = Link()
            root = self.__root
            last = root.prev
            (link.prev, link.next, link.key) = (last, root, key)
            last.next = link
            root.prev = proxy(link)
        dict_setitem(self, key, value)
    
    def __delitem__(self, key, dict_delitem=dict.__delitem__):
        """od.__delitem__(y) <==> del od[y]"""
        dict_delitem(self, key)
        link = self.__map.pop(key)
        link_prev = link.prev
        link_next = link.next
        link_prev.next = link_next
        link_next.prev = link_prev
    
    def __iter__(self):
        """od.__iter__() <==> iter(od)"""
        root = self.__root
        curr = root.next
        while curr is not root:
            yield curr.key
            curr = curr.next
    
    def __reversed__(self):
        """od.__reversed__() <==> reversed(od)"""
        root = self.__root
        curr = root.prev
        while curr is not root:
            yield curr.key
            curr = curr.prev
    
    def clear(self):
        """od.clear() -> None.  Remove all items from od."""
        root = self.__root
        root.prev = root.next = root
        self.__map.clear()
        dict.clear(self)
    
    def popitem(self, last=True):
        """od.popitem() -> (k, v), return and remove a (key, value) pair.
        Pairs are returned in LIFO order if last is true or FIFO order if false.

        """
        if not self:
            raise KeyError('dictionary is empty')
        root = self.__root
        if last:
            link = root.prev
            link_prev = link.prev
            link_prev.next = root
            root.prev = link_prev
        else:
            link = root.next
            link_next = link.next
            root.next = link_next
            link_next.prev = root
        key = link.key
        del self.__map[key]
        value = dict.pop(self, key)
        return (key, value)
    
    def move_to_end(self, key, last=True):
        """Move an existing element to the end (or beginning if last==False).

        Raises KeyError if the element does not exist.
        When last=True, acts like a fast version of self[key]=self.pop(key).

        """
        link = self.__map[key]
        link_prev = link.prev
        link_next = link.next
        link_prev.next = link_next
        link_next.prev = link_prev
        root = self.__root
        if last:
            last = root.prev
            link.prev = last
            link.next = root
            last.next = root.prev = link
        else:
            first = root.next
            link.prev = root
            link.next = first
            root.next = first.prev = link
    
    def __sizeof__(self):
        sizeof = sys.getsizeof
        n = len(self) + 1
        size = sizeof(self.__dict__)
        size += sizeof(self.__map) * 2
        size += sizeof(self.__hardroot) * n
        size += sizeof(self.__root) * n
        return size
    update = __update = MutableMapping.update
    keys = MutableMapping.keys
    values = MutableMapping.values
    items = MutableMapping.items
    __ne__ = MutableMapping.__ne__
    __marker = object()
    
    def pop(self, key, default=__marker):
        """od.pop(k[,d]) -> v, remove specified key and return the corresponding
        value.  If key is not found, d is returned if given, otherwise KeyError
        is raised.

        """
        if key in self:
            result = self[key]
            del self[key]
            return result
        if default is self.__marker:
            raise KeyError(key)
        return default
    
    def setdefault(self, key, default=None):
        """od.setdefault(k[,d]) -> od.get(k,d), also set od[k]=d if k not in od"""
        if key in self:
            return self[key]
        self[key] = default
        return default
    
    @recursive_repr()
    def __repr__(self):
        """od.__repr__() <==> repr(od)"""
        if not self:
            return '%s()' % (self.__class__.__name__, )
        return '%s(%r)' % (self.__class__.__name__, list(self.items()))
    
    def __reduce__(self):
        """Return state information for pickling"""
        inst_dict = vars(self).copy()
        for k in vars(OrderedDict()):
            inst_dict.pop(k, None)
        return (self.__class__, (), (inst_dict or None), None, iter(self.items()))
    
    def copy(self):
        """od.copy() -> a shallow copy of od"""
        return self.__class__(self)
    
    @classmethod
    def fromkeys(cls, iterable, value=None):
        """OD.fromkeys(S[, v]) -> New ordered dictionary with keys from S.
        If not specified, the value defaults to None.

        """
        self = cls()
        for key in iterable:
            self[key] = value
        return self
    
    def __eq__(self, other):
        """od.__eq__(y) <==> od==y.  Comparison to another OD is order-sensitive
        while comparison to a regular mapping is order-insensitive.

        """
        if isinstance(other, OrderedDict):
            return (dict.__eq__(self, other) and all(map(_eq, self, other)))
        return dict.__eq__(self, other)

try:
    from operator import itemgetter
    from heapq import nlargest
except ImportError:
    pass

def _count_elements(mapping, iterable):
    """Tally elements from the iterable."""
    mapping_get = mapping.get
    for elem in iterable:
        mapping[elem] = mapping_get(elem, 0) + 1


class Counter(dict):
    """Dict subclass for counting hashable items.  Sometimes called a bag
    or multiset.  Elements are stored as dictionary keys and their counts
    are stored as dictionary values.

    >>> c = Counter('abcdeabcdabcaba')  # count elements from a string

    >>> c.most_common(3)                # three most common elements
    [('a', 5), ('b', 4), ('c', 3)]
    >>> sorted(c)                       # list all unique elements
    ['a', 'b', 'c', 'd', 'e']
    >>> ''.join(sorted(c.elements()))   # list elements with repetitions
    'aaaaabbbbcccdde'
    >>> sum(c.values())                 # total of all counts
    15

    >>> c['a']                          # count of letter 'a'
    5
    >>> for elem in 'shazam':           # update counts from an iterable
    ...     c[elem] += 1                # by adding 1 to each element's count
    >>> c['a']                          # now there are seven 'a'
    7
    >>> del c['b']                      # remove all 'b'
    >>> c['b']                          # now there are zero 'b'
    0

    >>> d = Counter('simsalabim')       # make another counter
    >>> c.update(d)                     # add in the second counter
    >>> c['a']                          # now there are nine 'a'
    9

    >>> c.clear()                       # empty the counter
    >>> c
    Counter()

    Note:  If a count is set to zero or reduced to zero, it will remain
    in the counter until the entry is deleted or the counter is cleared:

    >>> c = Counter('aaabbc')
    >>> c['b'] -= 2                     # reduce the count of 'b' by two
    >>> c.most_common()                 # 'b' is still in, but its count is zero
    [('a', 3), ('c', 1), ('b', 0)]

    """
    
    def __init__(*args, **kwds):
        """Create a new, empty Counter object.  And if given, count elements
        from an input iterable.  Or, initialize the count from another mapping
        of elements to their counts.

        >>> c = Counter()                           # a new, empty counter
        >>> c = Counter('gallahad')                 # a new counter from an iterable
        >>> c = Counter({'a': 4, 'b': 2})           # a new counter from a mapping
        >>> c = Counter(a=4, b=2)                   # a new counter from keyword args

        """
        if not args:
            raise TypeError("descriptor '__init__' of 'Counter' object needs an argument")
        self = args[0]
        args = args[1:]
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        super(Counter, self).__init__()
        self.update(*args, **kwds)
    
    def __missing__(self, key):
        """The count of elements not in the Counter is zero."""
        return 0
    
    def most_common(self, n=None):
        """List the n most common elements and their counts from the most
        common to the least.  If n is None, then list all element counts.

        >>> Counter('abcdeabcdabcaba').most_common(3)
        [('a', 5), ('b', 4), ('c', 3)]

        """
        if n is None:
            return sorted(self.items(), key=_itemgetter(1), reverse=True)
        return _heapq.nlargest(n, self.items(), key=_itemgetter(1))
    
    def elements(self):
        """Iterator over elements repeating each as many times as its count.

        >>> c = Counter('ABCABC')
        >>> sorted(c.elements())
        ['A', 'A', 'B', 'B', 'C', 'C']

        # Knuth's example for prime factors of 1836:  2**2 * 3**3 * 17**1
        >>> prime_factors = Counter({2: 2, 3: 3, 17: 1})
        >>> product = 1
        >>> for factor in prime_factors.elements():     # loop over factors
        ...     product *= factor                       # and multiply them
        >>> product
        1836

        Note, if an element's count has been set to zero or is a negative
        number, elements() will ignore it.

        """
        return _chain.from_iterable(_starmap(_repeat, self.items()))
    
    @classmethod
    def fromkeys(cls, iterable, v=None):
        raise NotImplementedError('Counter.fromkeys() is undefined.  Use Counter(iterable) instead.')
    
    def update(*args, **kwds):
        """Like dict.update() but add counts instead of replacing them.

        Source can be an iterable, a dictionary, or another Counter instance.

        >>> c = Counter('which')
        >>> c.update('witch')           # add elements from another iterable
        >>> d = Counter('watch')
        >>> c.update(d)                 # add elements from another counter
        >>> c['h']                      # four 'h' in which, witch, and watch
        4

        """
        if not args:
            raise TypeError("descriptor 'update' of 'Counter' object needs an argument")
        self = args[0]
        args = args[1:]
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        iterable = (args[0] if args else None)
        if iterable is not None:
            if isinstance(iterable, Mapping):
                if self:
                    self_get = self.get
                    for (elem, count) in iterable.items():
                        self[elem] = count + self_get(elem, 0)
                else:
                    super(Counter, self).update(iterable)
            else:
                _count_elements(self, iterable)
        if kwds:
            self.update(kwds)
    
    def subtract(*args, **kwds):
        """Like dict.update() but subtracts counts instead of replacing them.
        Counts can be reduced below zero.  Both the inputs and outputs are
        allowed to contain zero and negative counts.

        Source can be an iterable, a dictionary, or another Counter instance.

        >>> c = Counter('which')
        >>> c.subtract('witch')             # subtract elements from another iterable
        >>> c.subtract(Counter('watch'))    # subtract elements from another counter
        >>> c['h']                          # 2 in which, minus 1 in witch, minus 1 in watch
        0
        >>> c['w']                          # 1 in which, minus 1 in witch, minus 1 in watch
        -1

        """
        if not args:
            raise TypeError("descriptor 'subtract' of 'Counter' object needs an argument")
        self = args[0]
        args = args[1:]
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        iterable = (args[0] if args else None)
        if iterable is not None:
            self_get = self.get
            if isinstance(iterable, Mapping):
                for (elem, count) in iterable.items():
                    self[elem] = self_get(elem, 0) - count
            else:
                for elem in iterable:
                    self[elem] = self_get(elem, 0) - 1
        if kwds:
            self.subtract(kwds)
    
    def copy(self):
        """Return a shallow copy."""
        return self.__class__(self)
    
    def __reduce__(self):
        return (self.__class__, (dict(self), ))
    
    def __delitem__(self, elem):
        """Like dict.__delitem__() but does not raise KeyError for missing values."""
        if elem in self:
            super(Counter, self).__delitem__(elem)
    
    def __repr__(self):
        if not self:
            return '%s()' % self.__class__.__name__
        try:
            items = ', '.join(map('%r: %r'.__mod__, self.most_common()))
            return '%s({%s})' % (self.__class__.__name__, items)
        except TypeError:
            return '{0}({1!r})'.format(self.__class__.__name__, dict(self))
    
    def __add__(self, other):
        """Add counts from two counters.

        >>> Counter('abbb') + Counter('bcc')
        Counter({'b': 4, 'c': 2, 'a': 1})

        """
        if not isinstance(other, Counter):
            return NotImplemented
        result = Counter()
        for (elem, count) in self.items():
            newcount = count + other[elem]
            if newcount > 0:
                result[elem] = newcount
        for (elem, count) in other.items():
            if (elem not in self and count > 0):
                result[elem] = count
        return result
    
    def __sub__(self, other):
        """ Subtract count, but keep only results with positive counts.

        >>> Counter('abbbc') - Counter('bccd')
        Counter({'b': 2, 'a': 1})

        """
        if not isinstance(other, Counter):
            return NotImplemented
        result = Counter()
        for (elem, count) in self.items():
            newcount = count - other[elem]
            if newcount > 0:
                result[elem] = newcount
        for (elem, count) in other.items():
            if (elem not in self and count < 0):
                result[elem] = 0 - count
        return result
    
    def __or__(self, other):
        """Union is the maximum of value in either of the input counters.

        >>> Counter('abbb') | Counter('bcc')
        Counter({'b': 3, 'c': 2, 'a': 1})

        """
        if not isinstance(other, Counter):
            return NotImplemented
        result = Counter()
        for (elem, count) in self.items():
            other_count = other[elem]
            newcount = (other_count if count < other_count else count)
            if newcount > 0:
                result[elem] = newcount
        for (elem, count) in other.items():
            if (elem not in self and count > 0):
                result[elem] = count
        return result
    
    def __and__(self, other):
        """ Intersection is the minimum of corresponding counts.

        >>> Counter('abbb') & Counter('bcc')
        Counter({'b': 1})

        """
        if not isinstance(other, Counter):
            return NotImplemented
        result = Counter()
        for (elem, count) in self.items():
            other_count = other[elem]
            newcount = (count if count < other_count else other_count)
            if newcount > 0:
                result[elem] = newcount
        return result
    
    def __pos__(self):
        """Adds an empty counter, effectively stripping negative and zero counts"""
        return self + Counter()
    
    def __neg__(self):
        """Subtracts from an empty counter.  Strips positive and zero counts,
        and flips the sign on negative counts.

        """
        return Counter() - self
    
    def _keep_positive(self):
        """Internal method to strip elements with a negative or zero count"""
        nonpositive = [elem for (elem, count) in self.items() if not count > 0]
        for elem in nonpositive:
            del self[elem]
        return self
    
    def __iadd__(self, other):
        """Inplace add from another counter, keeping only positive counts.

        >>> c = Counter('abbb')
        >>> c += Counter('bcc')
        >>> c
        Counter({'b': 4, 'c': 2, 'a': 1})

        """
        for (elem, count) in other.items():
            self[elem] += count
        return self._keep_positive()
    
    def __isub__(self, other):
        """Inplace subtract counter, but keep only results with positive counts.

        >>> c = Counter('abbbc')
        >>> c -= Counter('bccd')
        >>> c
        Counter({'b': 2, 'a': 1})

        """
        for (elem, count) in other.items():
            self[elem] -= count
        return self._keep_positive()
    
    def __ior__(self, other):
        """Inplace union is the maximum of value from either counter.

        >>> c = Counter('abbb')
        >>> c |= Counter('bcc')
        >>> c
        Counter({'b': 3, 'c': 2, 'a': 1})

        """
        for (elem, other_count) in other.items():
            count = self[elem]
            if other_count > count:
                self[elem] = other_count
        return self._keep_positive()
    
    def __iand__(self, other):
        """Inplace intersection is the minimum of corresponding counts.

        >>> c = Counter('abbb')
        >>> c &= Counter('bcc')
        >>> c
        Counter({'b': 1})

        """
        for (elem, count) in self.items():
            other_count = other[elem]
            if other_count < count:
                self[elem] = other_count
        return self._keep_positive()


def check_output(*popenargs, **kwargs):
    """
    For Python 2.6 compatibility: see
    http://stackoverflow.com/questions/4814970/
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.misc.check_output', 'check_output(*popenargs, **kwargs)', {'subprocess': subprocess, 'popenargs': popenargs, 'kwargs': kwargs}, 1)

def count(start=0, step=1):
    """
    ``itertools.count`` in Py 2.6 doesn't accept a step
    parameter. This is an enhanced version of ``itertools.count``
    for Py2.6 equivalent to ``itertools.count`` in Python 2.7+.
    """
    while True:
        yield start
        start += step


class ChainMap(MutableMapping):
    """ A ChainMap groups multiple dicts (or other mappings) together
    to create a single, updateable view.

    The underlying mappings are stored in a list.  That list is public and can
    accessed or updated using the *maps* attribute.  There is no other state.

    Lookups search the underlying mappings successively until a key is found.
    In contrast, writes, updates, and deletions only operate on the first
    mapping.

    """
    
    def __init__(self, *maps):
        """Initialize a ChainMap by setting *maps* to the given mappings.
        If no mappings are provided, a single empty dictionary is used.

        """
        self.maps = (list(maps) or [{}])
    
    def __missing__(self, key):
        raise KeyError(key)
    
    def __getitem__(self, key):
        for mapping in self.maps:
            try:
                return mapping[key]
            except KeyError:
                pass
        return self.__missing__(key)
    
    def get(self, key, default=None):
        return (self[key] if key in self else default)
    
    def __len__(self):
        return len(set().union(*self.maps))
    
    def __iter__(self):
        return iter(set().union(*self.maps))
    
    def __contains__(self, key):
        return any((key in m for m in self.maps))
    
    def __bool__(self):
        return any(self.maps)
    __nonzero__ = __bool__
    
    @recursive_repr()
    def __repr__(self):
        return '{0.__class__.__name__}({1})'.format(self, ', '.join(map(repr, self.maps)))
    
    @classmethod
    def fromkeys(cls, iterable, *args):
        """Create a ChainMap with a single dict created from the iterable."""
        return cls(dict.fromkeys(iterable, *args))
    
    def copy(self):
        """New ChainMap or subclass with a new copy of maps[0] and refs to maps[1:]"""
        return self.__class__(self.maps[0].copy(), *self.maps[1:])
    __copy__ = copy
    
    def new_child(self, m=None):
        """
        New ChainMap with a new map followed by all previous maps. If no
        map is provided, an empty dict is used.
        """
        if m is None:
            m = {}
        return self.__class__(m, *self.maps)
    
    @property
    def parents(self):
        """New ChainMap from maps[1:]."""
        return self.__class__(*self.maps[1:])
    
    def __setitem__(self, key, value):
        self.maps[0][key] = value
    
    def __delitem__(self, key):
        try:
            del self.maps[0][key]
        except KeyError:
            raise KeyError('Key not found in the first mapping: {0!r}'.format(key))
    
    def popitem(self):
        """Remove and return an item pair from maps[0]. Raise KeyError is maps[0] is empty."""
        try:
            return self.maps[0].popitem()
        except KeyError:
            raise KeyError('No keys found in the first mapping.')
    
    def pop(self, key, *args):
        """Remove *key* from maps[0] and return its value. Raise KeyError if *key* not in maps[0]."""
        try:
            return self.maps[0].pop(key, *args)
        except KeyError:
            raise KeyError('Key not found in the first mapping: {0!r}'.format(key))
    
    def clear(self):
        """Clear maps[0], leaving maps[1:] intact."""
        self.maps[0].clear()

from socket import _GLOBAL_DEFAULT_TIMEOUT

def create_connection(address, timeout=_GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    """Backport of 3-argument create_connection() for Py2.6.

    Connect to *address* and return the socket object.

    Convenience function.  Connect to *address* (a 2-tuple ``(host,
    port)``) and return the socket object.  Passing the optional
    *timeout* parameter will set the timeout on the socket instance
    before attempting to connect.  If no *timeout* is supplied, the
    global default timeout setting returned by :func:`getdefaulttimeout`
    is used.  If *source_address* is set it must be a tuple of (host, port)
    for the socket to bind as a source address before making the connection.
    An host of '' or port 0 tells the OS to use the default.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.misc.create_connection', 'create_connection(address, timeout=_GLOBAL_DEFAULT_TIMEOUT, source_address=None)', {'getaddrinfo': getaddrinfo, 'SOCK_STREAM': SOCK_STREAM, 'socket': socket, 'error': error, 'address': address, 'timeout': timeout, 'source_address': source_address, '_GLOBAL_DEFAULT_TIMEOUT': _GLOBAL_DEFAULT_TIMEOUT}, 1)

def cmp_to_key(mycmp):
    """Convert a cmp= function into a key= function"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.misc.cmp_to_key', 'cmp_to_key(mycmp)', {'mycmp': mycmp}, 1)
_OrderedDict = OrderedDict
_Counter = Counter
_check_output = check_output
_count = count
_ceil = ceil
__count_elements = _count_elements
_recursive_repr = recursive_repr
_ChainMap = ChainMap
_create_connection = create_connection
_cmp_to_key = cmp_to_key
if sys.version_info >= (2, 7):
    from collections import OrderedDict, Counter
    from itertools import count
    from functools import cmp_to_key
    try:
        from subprocess import check_output
    except ImportError:
        pass
    from socket import create_connection
if sys.version_info >= (3, 0):
    from math import ceil
    from collections import _count_elements
if sys.version_info >= (3, 3):
    from reprlib import recursive_repr
    from collections import ChainMap

