"""
This file is separate from ``_add_newdocs.py`` so that it can be mocked out by
our sphinx ``conf.py`` during doc builds, where we want to avoid showing
platform-dependent information.
"""

import sys
import os
from numpy.core import dtype
from numpy.core import numerictypes as _numerictypes
from numpy.core.function_base import add_newdoc

def numeric_type_aliases(aliases):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._add_newdocs_scalars.numeric_type_aliases', 'numeric_type_aliases(aliases)', {'_numerictypes': _numerictypes, 'aliases': aliases}, 1)
possible_aliases = numeric_type_aliases([('int8', '8-bit signed integer (``-128`` to ``127``)'), ('int16', '16-bit signed integer (``-32_768`` to ``32_767``)'), ('int32', '32-bit signed integer (``-2_147_483_648`` to ``2_147_483_647``)'), ('int64', '64-bit signed integer (``-9_223_372_036_854_775_808`` to ``9_223_372_036_854_775_807``)'), ('intp', 'Signed integer large enough to fit pointer, compatible with C ``intptr_t``'), ('uint8', '8-bit unsigned integer (``0`` to ``255``)'), ('uint16', '16-bit unsigned integer (``0`` to ``65_535``)'), ('uint32', '32-bit unsigned integer (``0`` to ``4_294_967_295``)'), ('uint64', '64-bit unsigned integer (``0`` to ``18_446_744_073_709_551_615``)'), ('uintp', 'Unsigned integer large enough to fit pointer, compatible with C ``uintptr_t``'), ('float16', '16-bit-precision floating-point number type: sign bit, 5 bits exponent, 10 bits mantissa'), ('float32', '32-bit-precision floating-point number type: sign bit, 8 bits exponent, 23 bits mantissa'), ('float64', '64-bit precision floating-point number type: sign bit, 11 bits exponent, 52 bits mantissa'), ('float96', '96-bit extended-precision floating-point number type'), ('float128', '128-bit extended-precision floating-point number type'), ('complex64', 'Complex number type composed of 2 32-bit-precision floating-point numbers'), ('complex128', 'Complex number type composed of 2 64-bit-precision floating-point numbers'), ('complex192', 'Complex number type composed of 2 96-bit extended-precision floating-point numbers'), ('complex256', 'Complex number type composed of 2 128-bit extended-precision floating-point numbers')])

def _get_platform_and_machine():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._add_newdocs_scalars._get_platform_and_machine', '_get_platform_and_machine()', {'os': os, 'sys': sys}, 2)
(_system, _machine) = _get_platform_and_machine()
_doc_alias_string = f':Alias on this platform ({_system} {_machine}):'

def add_newdoc_for_scalar_type(obj, fixed_aliases, doc):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.core._add_newdocs_scalars.add_newdoc_for_scalar_type', 'add_newdoc_for_scalar_type(obj, fixed_aliases, doc)', {'_numerictypes': _numerictypes, 'dtype': dtype, '_doc_alias_string': _doc_alias_string, 'possible_aliases': possible_aliases, 'add_newdoc': add_newdoc, 'obj': obj, 'fixed_aliases': fixed_aliases, 'doc': doc}, 0)
add_newdoc_for_scalar_type('bool_', ['bool8'], "\n    Boolean type (True or False), stored as a byte.\n\n    .. warning::\n\n       The :class:`bool_` type is not a subclass of the :class:`int_` type\n       (the :class:`bool_` is not even a number type). This is different\n       than Python's default implementation of :class:`bool` as a\n       sub-class of :class:`int`.\n    ")
add_newdoc_for_scalar_type('byte', [], '\n    Signed integer type, compatible with C ``char``.\n    ')
add_newdoc_for_scalar_type('short', [], '\n    Signed integer type, compatible with C ``short``.\n    ')
add_newdoc_for_scalar_type('intc', [], '\n    Signed integer type, compatible with C ``int``.\n    ')
add_newdoc_for_scalar_type('int_', [], '\n    Signed integer type, compatible with Python `int` and C ``long``.\n    ')
add_newdoc_for_scalar_type('longlong', [], '\n    Signed integer type, compatible with C ``long long``.\n    ')
add_newdoc_for_scalar_type('ubyte', [], '\n    Unsigned integer type, compatible with C ``unsigned char``.\n    ')
add_newdoc_for_scalar_type('ushort', [], '\n    Unsigned integer type, compatible with C ``unsigned short``.\n    ')
add_newdoc_for_scalar_type('uintc', [], '\n    Unsigned integer type, compatible with C ``unsigned int``.\n    ')
add_newdoc_for_scalar_type('uint', [], '\n    Unsigned integer type, compatible with C ``unsigned long``.\n    ')
add_newdoc_for_scalar_type('ulonglong', [], '\n    Signed integer type, compatible with C ``unsigned long long``.\n    ')
add_newdoc_for_scalar_type('half', [], '\n    Half-precision floating-point number type.\n    ')
add_newdoc_for_scalar_type('single', [], '\n    Single-precision floating-point number type, compatible with C ``float``.\n    ')
add_newdoc_for_scalar_type('double', ['float_'], '\n    Double-precision floating-point number type, compatible with Python `float`\n    and C ``double``.\n    ')
add_newdoc_for_scalar_type('longdouble', ['longfloat'], '\n    Extended-precision floating-point number type, compatible with C\n    ``long double`` but not necessarily with IEEE 754 quadruple-precision.\n    ')
add_newdoc_for_scalar_type('csingle', ['singlecomplex'], '\n    Complex number type composed of two single-precision floating-point\n    numbers.\n    ')
add_newdoc_for_scalar_type('cdouble', ['cfloat', 'complex_'], '\n    Complex number type composed of two double-precision floating-point\n    numbers, compatible with Python `complex`.\n    ')
add_newdoc_for_scalar_type('clongdouble', ['clongfloat', 'longcomplex'], '\n    Complex number type composed of two extended-precision floating-point\n    numbers.\n    ')
add_newdoc_for_scalar_type('object_', [], '\n    Any Python object.\n    ')
add_newdoc_for_scalar_type('str_', ['unicode_'], '\n    A unicode string.\n\n    When used in arrays, this type strips trailing null codepoints.\n\n    Unlike the builtin `str`, this supports the :ref:`python:bufferobjects`, exposing its\n    contents as UCS4:\n\n    >>> m = memoryview(np.str_("abc"))\n    >>> m.format\n    \'3w\'\n    >>> m.tobytes()\n    b\'a\\x00\\x00\\x00b\\x00\\x00\\x00c\\x00\\x00\\x00\'\n    ')
add_newdoc_for_scalar_type('bytes_', ['string_'], '\n    A byte string.\n\n    When used in arrays, this type strips trailing null bytes.\n    ')
add_newdoc_for_scalar_type('void', [], '\n    np.void(length_or_data, /, dtype=None)\n\n    Create a new structured or unstructured void scalar.\n\n    Parameters\n    ----------\n    length_or_data : int, array-like, bytes-like, object\n       One of multiple meanings (see notes).  The length or\n       bytes data of an unstructured void.  Or alternatively,\n       the data to be stored in the new scalar when `dtype`\n       is provided.\n       This can be an array-like, in which case an array may\n       be returned.\n    dtype : dtype, optional\n        If provided the dtype of the new scalar.  This dtype must\n        be "void" dtype (i.e. a structured or unstructured void,\n        see also :ref:`defining-structured-types`).\n\n       ..versionadded:: 1.24\n\n    Notes\n    -----\n    For historical reasons and because void scalars can represent both\n    arbitrary byte data and structured dtypes, the void constructor\n    has three calling conventions:\n\n    1. ``np.void(5)`` creates a ``dtype="V5"`` scalar filled with five\n       ``\\0`` bytes.  The 5 can be a Python or NumPy integer.\n    2. ``np.void(b"bytes-like")`` creates a void scalar from the byte string.\n       The dtype itemsize will match the byte string length, here ``"V10"``.\n    3. When a ``dtype=`` is passed the call is rougly the same as an\n       array creation.  However, a void scalar rather than array is returned.\n\n    Please see the examples which show all three different conventions.\n\n    Examples\n    --------\n    >>> np.void(5)\n    void(b\'\\x00\\x00\\x00\\x00\\x00\')\n    >>> np.void(b\'abcd\')\n    void(b\'\\x61\\x62\\x63\\x64\')\n    >>> np.void((5, 3.2, "eggs"), dtype="i,d,S5")\n    (5, 3.2, b\'eggs\')  # looks like a tuple, but is `np.void`\n    >>> np.void(3, dtype=[(\'x\', np.int8), (\'y\', np.int8)])\n    (3, 3)  # looks like a tuple, but is `np.void`\n\n    ')
add_newdoc_for_scalar_type('datetime64', [], "\n    If created from a 64-bit integer, it represents an offset from\n    ``1970-01-01T00:00:00``.\n    If created from string, the string can be in ISO 8601 date\n    or datetime format.\n\n    >>> np.datetime64(10, 'Y')\n    numpy.datetime64('1980')\n    >>> np.datetime64('1980', 'Y')\n    numpy.datetime64('1980')\n    >>> np.datetime64(10, 'D')\n    numpy.datetime64('1970-01-11')\n\n    See :ref:`arrays.datetime` for more information.\n    ")
add_newdoc_for_scalar_type('timedelta64', [], '\n    A timedelta stored as a 64-bit integer.\n\n    See :ref:`arrays.datetime` for more information.\n    ')
add_newdoc('numpy.core.numerictypes', 'integer', ('is_integer', '\n    integer.is_integer() -> bool\n\n    Return ``True`` if the number is finite with integral value.\n\n    .. versionadded:: 1.22\n\n    Examples\n    --------\n    >>> np.int64(-2).is_integer()\n    True\n    >>> np.uint32(5).is_integer()\n    True\n    '))
for float_name in ('half', 'single', 'double', 'longdouble'):
    add_newdoc('numpy.core.numerictypes', float_name, ('as_integer_ratio', '\n        {ftype}.as_integer_ratio() -> (int, int)\n\n        Return a pair of integers, whose ratio is exactly equal to the original\n        floating point number, and with a positive denominator.\n        Raise `OverflowError` on infinities and a `ValueError` on NaNs.\n\n        >>> np.{ftype}(10.0).as_integer_ratio()\n        (10, 1)\n        >>> np.{ftype}(0.0).as_integer_ratio()\n        (0, 1)\n        >>> np.{ftype}(-.25).as_integer_ratio()\n        (-1, 4)\n        '.format(ftype=float_name)))
    add_newdoc('numpy.core.numerictypes', float_name, ('is_integer', f'\n        {float_name}.is_integer() -> bool\n\n        Return ``True`` if the floating point number is finite with integral\n        value, and ``False`` otherwise.\n\n        .. versionadded:: 1.22\n\n        Examples\n        --------\n        >>> np.{float_name}(-2.0).is_integer()\n        True\n        >>> np.{float_name}(3.2).is_integer()\n        False\n        '))
for int_name in ('int8', 'uint8', 'int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'int64', 'uint64', 'int64', 'uint64'):
    add_newdoc('numpy.core.numerictypes', int_name, ('bit_count', f'\n        {int_name}.bit_count() -> int\n\n        Computes the number of 1-bits in the absolute value of the input.\n        Analogous to the builtin `int.bit_count` or ``popcount`` in C++.\n\n        Examples\n        --------\n        >>> np.{int_name}(127).bit_count()\n        7' + ((f'\n        >>> np.{int_name}(-127).bit_count()\n        7\n        ' if dtype(int_name).char.islower() else ''))))

