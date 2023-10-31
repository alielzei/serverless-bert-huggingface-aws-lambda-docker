"""
A place for internal code

Some things are more easily handled Python.

"""

import ast
import re
import sys
import warnings
from .multiarray import dtype, array, ndarray, promote_types
try:
    import ctypes
except ImportError:
    ctypes = None
IS_PYPY = sys.implementation.name == 'pypy'
if sys.byteorder == 'little':
    _nbo = '<'
else:
    _nbo = '>'

def _makenames_list(adict, align):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._makenames_list', '_makenames_list(adict, align)', {'dtype': dtype, 'adict': adict, 'align': align}, 4)

def _usefields(adict, align):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._usefields', '_usefields(adict, align)', {'_makenames_list': _makenames_list, 'dtype': dtype, 'adict': adict, 'align': align}, 1)

def _array_descr(descriptor):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._array_descr', '_array_descr(descriptor)', {'_array_descr': _array_descr, 'descriptor': descriptor}, 1)

def _reconstruct(subtype, shape, dtype):
    return ndarray.__new__(subtype, shape, dtype)
format_re = re.compile('(?P<order1>[<>|=]?)(?P<repeats> *[(]?[ ,0-9]*[)]? *)(?P<order2>[<>|=]?)(?P<dtype>[A-Za-z0-9.?]*(?:\\[[a-zA-Z0-9,.]+\\])?)')
sep_re = re.compile('\\s*,\\s*')
space_re = re.compile('\\s+$')
_convorder = {'=': _nbo}

def _commastring(astr):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._commastring', '_commastring(astr)', {'format_re': format_re, 'space_re': space_re, 'sep_re': sep_re, '_convorder': _convorder, '_nbo': _nbo, 'ast': ast, 'astr': astr}, 1)


class dummy_ctype:
    
    def __init__(self, cls):
        self._cls = cls
    
    def __mul__(self, other):
        return self
    
    def __call__(self, *other):
        return self._cls(other)
    
    def __eq__(self, other):
        return self._cls == other._cls
    
    def __ne__(self, other):
        return self._cls != other._cls


def _getintp_ctype():
    val = _getintp_ctype.cache
    if val is not None:
        return val
    if ctypes is None:
        import numpy as np
        val = dummy_ctype(np.intp)
    else:
        char = dtype('p').char
        if char == 'i':
            val = ctypes.c_int
        elif char == 'l':
            val = ctypes.c_long
        elif char == 'q':
            val = ctypes.c_longlong
        else:
            val = ctypes.c_long
    _getintp_ctype.cache = val
    return val
_getintp_ctype.cache = None


class _missing_ctypes:
    
    def cast(self, num, obj):
        return num.value
    
    
    class c_void_p:
        
        def __init__(self, ptr):
            self.value = ptr
    



class _ctypes:
    
    def __init__(self, array, ptr=None):
        self._arr = array
        if ctypes:
            self._ctypes = ctypes
            self._data = self._ctypes.c_void_p(ptr)
        else:
            self._ctypes = _missing_ctypes()
            self._data = self._ctypes.c_void_p(ptr)
            self._data._objects = array
        if self._arr.ndim == 0:
            self._zerod = True
        else:
            self._zerod = False
    
    def data_as(self, obj):
        """
        Return the data pointer cast to a particular c-types object.
        For example, calling ``self._as_parameter_`` is equivalent to
        ``self.data_as(ctypes.c_void_p)``. Perhaps you want to use the data as a
        pointer to a ctypes array of floating-point data:
        ``self.data_as(ctypes.POINTER(ctypes.c_double))``.

        The returned pointer will keep a reference to the array.
        """
        ptr = self._ctypes.cast(self._data, obj)
        ptr._arr = self._arr
        return ptr
    
    def shape_as(self, obj):
        """
        Return the shape tuple as an array of some other c-types
        type. For example: ``self.shape_as(ctypes.c_short)``.
        """
        if self._zerod:
            return None
        return (obj * self._arr.ndim)(*self._arr.shape)
    
    def strides_as(self, obj):
        """
        Return the strides tuple as an array of some other
        c-types type. For example: ``self.strides_as(ctypes.c_longlong)``.
        """
        if self._zerod:
            return None
        return (obj * self._arr.ndim)(*self._arr.strides)
    
    @property
    def data(self):
        """
        A pointer to the memory area of the array as a Python integer.
        This memory area may contain data that is not aligned, or not in correct
        byte-order. The memory area may not even be writeable. The array
        flags and data-type of this array should be respected when passing this
        attribute to arbitrary C-code to avoid trouble that can include Python
        crashing. User Beware! The value of this attribute is exactly the same
        as ``self._array_interface_['data'][0]``.

        Note that unlike ``data_as``, a reference will not be kept to the array:
        code like ``ctypes.c_void_p((a + b).ctypes.data)`` will result in a
        pointer to a deallocated array, and should be spelt
        ``(a + b).ctypes.data_as(ctypes.c_void_p)``
        """
        return self._data.value
    
    @property
    def shape(self):
        """
        (c_intp*self.ndim): A ctypes array of length self.ndim where
        the basetype is the C-integer corresponding to ``dtype('p')`` on this
        platform (see `~numpy.ctypeslib.c_intp`). This base-type could be
        `ctypes.c_int`, `ctypes.c_long`, or `ctypes.c_longlong` depending on
        the platform. The ctypes array contains the shape of
        the underlying array.
        """
        return self.shape_as(_getintp_ctype())
    
    @property
    def strides(self):
        """
        (c_intp*self.ndim): A ctypes array of length self.ndim where
        the basetype is the same as for the shape attribute. This ctypes array
        contains the strides information from the underlying array. This strides
        information is important for showing how many bytes must be jumped to
        get to the next element in the array.
        """
        return self.strides_as(_getintp_ctype())
    
    @property
    def _as_parameter_(self):
        """
        Overrides the ctypes semi-magic method

        Enables `c_func(some_array.ctypes)`
        """
        return self.data_as(ctypes.c_void_p)
    
    def get_data(self):
        """Deprecated getter for the `_ctypes.data` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_data" is deprecated. Use "data" instead', DeprecationWarning, stacklevel=2)
        return self.data
    
    def get_shape(self):
        """Deprecated getter for the `_ctypes.shape` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_shape" is deprecated. Use "shape" instead', DeprecationWarning, stacklevel=2)
        return self.shape
    
    def get_strides(self):
        """Deprecated getter for the `_ctypes.strides` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_strides" is deprecated. Use "strides" instead', DeprecationWarning, stacklevel=2)
        return self.strides
    
    def get_as_parameter(self):
        """Deprecated getter for the `_ctypes._as_parameter_` property.

        .. deprecated:: 1.21
        """
        warnings.warn('"get_as_parameter" is deprecated. Use "_as_parameter_" instead', DeprecationWarning, stacklevel=2)
        return self._as_parameter_


def _newnames(datatype, order):
    """
    Given a datatype and an order object, return a new names tuple, with the
    order indicated
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._newnames', '_newnames(datatype, order)', {'datatype': datatype, 'order': order}, 1)

def _copy_fields(ary):
    """Return copy of structured array with padding between fields removed.

    Parameters
    ----------
    ary : ndarray
       Structured array from which to remove padding bytes

    Returns
    -------
    ary_copy : ndarray
       Copy of ary with padding bytes removed
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._copy_fields', '_copy_fields(ary)', {'array': array, 'ary': ary}, 1)

def _promote_fields(dt1, dt2):
    """ Perform type promotion for two structured dtypes.

    Parameters
    ----------
    dt1 : structured dtype
        First dtype.
    dt2 : structured dtype
        Second dtype.

    Returns
    -------
    out : dtype
        The promoted dtype

    Notes
    -----
    If one of the inputs is aligned, the result will be.  The titles of
    both descriptors must match (point to the same field).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._promote_fields', '_promote_fields(dt1, dt2)', {'promote_types': promote_types, 'dtype': dtype, 'dt1': dt1, 'dt2': dt2}, 1)

def _getfield_is_safe(oldtype, newtype, offset):
    """ Checks safety of getfield for object arrays.

    As in _view_is_safe, we need to check that memory containing objects is not
    reinterpreted as a non-object datatype and vice versa.

    Parameters
    ----------
    oldtype : data-type
        Data type of the original ndarray.
    newtype : data-type
        Data type of the field being accessed by ndarray.getfield
    offset : int
        Offset of the field being accessed by ndarray.getfield

    Raises
    ------
    TypeError
        If the field access is invalid

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._getfield_is_safe', '_getfield_is_safe(oldtype, newtype, offset)', {'oldtype': oldtype, 'newtype': newtype, 'offset': offset}, 1)

def _view_is_safe(oldtype, newtype):
    """ Checks safety of a view involving object arrays, for example when
    doing::

        np.zeros(10, dtype=oldtype).view(newtype)

    Parameters
    ----------
    oldtype : data-type
        Data type of original ndarray
    newtype : data-type
        Data type of the view

    Raises
    ------
    TypeError
        If the new type is incompatible with the old type.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._view_is_safe', '_view_is_safe(oldtype, newtype)', {'oldtype': oldtype, 'newtype': newtype}, 1)
_pep3118_native_map = {'?': '?', 'c': 'S1', 'b': 'b', 'B': 'B', 'h': 'h', 'H': 'H', 'i': 'i', 'I': 'I', 'l': 'l', 'L': 'L', 'q': 'q', 'Q': 'Q', 'e': 'e', 'f': 'f', 'd': 'd', 'g': 'g', 'Zf': 'F', 'Zd': 'D', 'Zg': 'G', 's': 'S', 'w': 'U', 'O': 'O', 'x': 'V'}
_pep3118_native_typechars = ''.join(_pep3118_native_map.keys())
_pep3118_standard_map = {'?': '?', 'c': 'S1', 'b': 'b', 'B': 'B', 'h': 'i2', 'H': 'u2', 'i': 'i4', 'I': 'u4', 'l': 'i4', 'L': 'u4', 'q': 'i8', 'Q': 'u8', 'e': 'f2', 'f': 'f', 'd': 'd', 'Zf': 'F', 'Zd': 'D', 's': 'S', 'w': 'U', 'O': 'O', 'x': 'V'}
_pep3118_standard_typechars = ''.join(_pep3118_standard_map.keys())
_pep3118_unsupported_map = {'u': 'UCS-2 strings', '&': 'pointers', 't': 'bitfields', 'X': 'function pointers'}


class _Stream:
    
    def __init__(self, s):
        self.s = s
        self.byteorder = '@'
    
    def advance(self, n):
        res = self.s[:n]
        self.s = self.s[n:]
        return res
    
    def consume(self, c):
        if self.s[:len(c)] == c:
            self.advance(len(c))
            return True
        return False
    
    def consume_until(self, c):
        if callable(c):
            i = 0
            while (i < len(self.s) and not c(self.s[i])):
                i = i + 1
            return self.advance(i)
        else:
            i = self.s.index(c)
            res = self.advance(i)
            self.advance(len(c))
            return res
    
    @property
    def next(self):
        return self.s[0]
    
    def __bool__(self):
        return bool(self.s)


def _dtype_from_pep3118(spec):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._dtype_from_pep3118', '_dtype_from_pep3118(spec)', {'_Stream': _Stream, '__dtype_from_pep3118': __dtype_from_pep3118, 'spec': spec}, 1)

def __dtype_from_pep3118(stream, is_subdtype):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal.__dtype_from_pep3118', '__dtype_from_pep3118(stream, is_subdtype)', {'_pep3118_native_map': _pep3118_native_map, '_pep3118_native_typechars': _pep3118_native_typechars, '_pep3118_standard_map': _pep3118_standard_map, '_pep3118_standard_typechars': _pep3118_standard_typechars, '__dtype_from_pep3118': __dtype_from_pep3118, 'dtype': dtype, '_pep3118_unsupported_map': _pep3118_unsupported_map, '_prod': _prod, '_add_trailing_padding': _add_trailing_padding, '_lcm': _lcm, '_fix_names': _fix_names, 'stream': stream, 'is_subdtype': is_subdtype}, 2)

def _fix_names(field_spec):
    """ Replace names which are None with the next unused f%d name """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.core._internal._fix_names', '_fix_names(field_spec)', {'field_spec': field_spec}, 0)

def _add_trailing_padding(value, padding):
    """Inject the specified number of padding bytes at the end of a dtype"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._add_trailing_padding', '_add_trailing_padding(value, padding)', {'dtype': dtype, 'value': value, 'padding': padding}, 1)

def _prod(a):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._prod', '_prod(a)', {'a': a}, 1)

def _gcd(a, b):
    """Calculate the greatest common divisor of a and b"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal._gcd', '_gcd(a, b)', {'a': a, 'b': b}, 1)

def _lcm(a, b):
    return a // _gcd(a, b) * b

def array_ufunc_errmsg_formatter(dummy, ufunc, method, *inputs, **kwargs):
    """ Format the error message for when __array_ufunc__ gives up. """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._internal.array_ufunc_errmsg_formatter', 'array_ufunc_errmsg_formatter(dummy, ufunc, method, *inputs, **kwargs)', {'dummy': dummy, 'ufunc': ufunc, 'method': method, 'inputs': inputs, 'kwargs': kwargs}, 1)

def array_function_errmsg_formatter(public_api, types):
    """ Format the error message for when __array_ufunc__ gives up. """
    func_name = '{}.{}'.format(public_api.__module__, public_api.__name__)
    return "no implementation found for '{}' on types that implement __array_function__: {}".format(func_name, list(types))

def _ufunc_doc_signature_formatter(ufunc):
    """
    Builds a signature string which resembles PEP 457

    This is used to construct the first line of the docstring
    """
    if ufunc.nin == 1:
        in_args = 'x'
    else:
        in_args = ', '.join((f'x{i + 1}' for i in range(ufunc.nin)))
    if ufunc.nout == 0:
        out_args = ', /, out=()'
    elif ufunc.nout == 1:
        out_args = ', /, out=None'
    else:
        out_args = '[, {positional}], / [, out={default}]'.format(positional=', '.join(('out{}'.format(i + 1) for i in range(ufunc.nout))), default=repr((None, ) * ufunc.nout))
    kwargs = ", casting='same_kind', order='K', dtype=None, subok=True"
    if ufunc.signature is None:
        kwargs = f', where=True{kwargs}[, signature, extobj]'
    else:
        kwargs += '[, signature, extobj, axes, axis]'
    return '{name}({in_args}{out_args}, *{kwargs})'.format(name=ufunc.__name__, in_args=in_args, out_args=out_args, kwargs=kwargs)

def npy_ctypes_check(cls):
    try:
        if IS_PYPY:
            ctype_base = cls.__mro__[-3]
        else:
            ctype_base = cls.__mro__[-2]
        return '_ctypes' in ctype_base.__module__
    except Exception:
        return False

