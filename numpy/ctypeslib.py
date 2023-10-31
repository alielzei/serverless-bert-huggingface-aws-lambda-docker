"""
============================
``ctypes`` Utility Functions
============================

See Also
--------
load_library : Load a C library.
ndpointer : Array restype/argtype with verification.
as_ctypes : Create a ctypes array from an ndarray.
as_array : Create an ndarray from a ctypes array.

References
----------
.. [1] "SciPy Cookbook: ctypes", https://scipy-cookbook.readthedocs.io/items/Ctypes.html

Examples
--------
Load the C library:

>>> _lib = np.ctypeslib.load_library('libmystuff', '.')     #doctest: +SKIP

Our result type, an ndarray that must be of type double, be 1-dimensional
and is C-contiguous in memory:

>>> array_1d_double = np.ctypeslib.ndpointer(
...                          dtype=np.double,
...                          ndim=1, flags='CONTIGUOUS')    #doctest: +SKIP

Our C-function typically takes an array and updates its values
in-place.  For example::

    void foo_func(double* x, int length)
    {
        int i;
        for (i = 0; i < length; i++) {
            x[i] = i*i;
        }
    }

We wrap it using:

>>> _lib.foo_func.restype = None                      #doctest: +SKIP
>>> _lib.foo_func.argtypes = [array_1d_double, c_int] #doctest: +SKIP

Then, we're ready to call ``foo_func``:

>>> out = np.empty(15, dtype=np.double)
>>> _lib.foo_func(out, len(out))                #doctest: +SKIP

"""

__all__ = ['load_library', 'ndpointer', 'c_intp', 'as_ctypes', 'as_array', 'as_ctypes_type']
import os
from numpy import integer, ndarray, dtype as _dtype, asarray, frombuffer
from numpy.core.multiarray import _flagdict, flagsobj
try:
    import ctypes
except ImportError:
    ctypes = None
if ctypes is None:
    
    def _dummy(*args, **kwds):
        """
        Dummy object that raises an ImportError if ctypes is not available.

        Raises
        ------
        ImportError
            If ctypes is not available.

        """
        raise ImportError('ctypes is not available.')
    load_library = _dummy
    as_ctypes = _dummy
    as_array = _dummy
    from numpy import intp as c_intp
    _ndptr_base = object
else:
    import numpy.core._internal as nic
    c_intp = nic._getintp_ctype()
    del nic
    _ndptr_base = ctypes.c_void_p
    
    def load_library(libname, loader_path):
        """
        It is possible to load a library using

        >>> lib = ctypes.cdll[<full_path_name>] # doctest: +SKIP

        But there are cross-platform considerations, such as library file extensions,
        plus the fact Windows will just load the first library it finds with that name.
        NumPy supplies the load_library function as a convenience.

        .. versionchanged:: 1.20.0
            Allow libname and loader_path to take any
            :term:`python:path-like object`.

        Parameters
        ----------
        libname : path-like
            Name of the library, which can have 'lib' as a prefix,
            but without an extension.
        loader_path : path-like
            Where the library can be found.

        Returns
        -------
        ctypes.cdll[libpath] : library object
           A ctypes library object

        Raises
        ------
        OSError
            If there is no library with the expected extension, or the
            library is defective and cannot be loaded.
        """
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.ctypeslib.load_library', 'load_library(libname, loader_path)', {'ctypes': ctypes, 'os': os, 'libname': libname, 'loader_path': loader_path}, 1)

def _num_fromflags(flaglist):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ctypeslib._num_fromflags', '_num_fromflags(flaglist)', {'_flagdict': _flagdict, 'flaglist': flaglist}, 1)
_flagnames = ['C_CONTIGUOUS', 'F_CONTIGUOUS', 'ALIGNED', 'WRITEABLE', 'OWNDATA', 'WRITEBACKIFCOPY']

def _flags_fromnum(num):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ctypeslib._flags_fromnum', '_flags_fromnum(num)', {'_flagnames': _flagnames, '_flagdict': _flagdict, 'num': num}, 1)


class _ndptr(_ndptr_base):
    
    @classmethod
    def from_param(cls, obj):
        if not isinstance(obj, ndarray):
            raise TypeError('argument must be an ndarray')
        if (cls._dtype_ is not None and obj.dtype != cls._dtype_):
            raise TypeError('array must have data type %s' % cls._dtype_)
        if (cls._ndim_ is not None and obj.ndim != cls._ndim_):
            raise TypeError('array must have %d dimension(s)' % cls._ndim_)
        if (cls._shape_ is not None and obj.shape != cls._shape_):
            raise TypeError('array must have shape %s' % str(cls._shape_))
        if (cls._flags_ is not None and obj.flags.num & cls._flags_ != cls._flags_):
            raise TypeError('array must have flags %s' % _flags_fromnum(cls._flags_))
        return obj.ctypes



class _concrete_ndptr(_ndptr):
    """
    Like _ndptr, but with `_shape_` and `_dtype_` specified.

    Notably, this means the pointer has enough information to reconstruct
    the array, which is not generally true.
    """
    
    def _check_retval_(self):
        """
        This method is called when this class is used as the .restype
        attribute for a shared-library function, to automatically wrap the
        pointer into an array.
        """
        return self.contents
    
    @property
    def contents(self):
        """
        Get an ndarray viewing the data pointed to by this pointer.

        This mirrors the `contents` attribute of a normal ctypes pointer
        """
        full_dtype = _dtype((self._dtype_, self._shape_))
        full_ctype = ctypes.c_char * full_dtype.itemsize
        buffer = ctypes.cast(self, ctypes.POINTER(full_ctype)).contents
        return frombuffer(buffer, dtype=full_dtype).squeeze(axis=0)

_pointer_type_cache = {}

def ndpointer(dtype=None, ndim=None, shape=None, flags=None):
    """
    Array-checking restype/argtypes.

    An ndpointer instance is used to describe an ndarray in restypes
    and argtypes specifications.  This approach is more flexible than
    using, for example, ``POINTER(c_double)``, since several restrictions
    can be specified, which are verified upon calling the ctypes function.
    These include data type, number of dimensions, shape and flags.  If a
    given array does not satisfy the specified restrictions,
    a ``TypeError`` is raised.

    Parameters
    ----------
    dtype : data-type, optional
        Array data-type.
    ndim : int, optional
        Number of array dimensions.
    shape : tuple of ints, optional
        Array shape.
    flags : str or tuple of str
        Array flags; may be one or more of:

          - C_CONTIGUOUS / C / CONTIGUOUS
          - F_CONTIGUOUS / F / FORTRAN
          - OWNDATA / O
          - WRITEABLE / W
          - ALIGNED / A
          - WRITEBACKIFCOPY / X

    Returns
    -------
    klass : ndpointer type object
        A type object, which is an ``_ndtpr`` instance containing
        dtype, ndim, shape and flags information.

    Raises
    ------
    TypeError
        If a given array does not satisfy the specified restrictions.

    Examples
    --------
    >>> clib.somefunc.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64,
    ...                                                  ndim=1,
    ...                                                  flags='C_CONTIGUOUS')]
    ... #doctest: +SKIP
    >>> clib.somefunc(np.array([1, 2, 3], dtype=np.float64))
    ... #doctest: +SKIP

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ctypeslib.ndpointer', 'ndpointer(dtype=None, ndim=None, shape=None, flags=None)', {'_dtype': _dtype, 'integer': integer, '_flags_fromnum': _flags_fromnum, 'flagsobj': flagsobj, '_num_fromflags': _num_fromflags, '_pointer_type_cache': _pointer_type_cache, '_concrete_ndptr': _concrete_ndptr, '_ndptr': _ndptr, 'dtype': dtype, 'ndim': ndim, 'shape': shape, 'flags': flags}, 1)
if ctypes is not None:
    
    def _ctype_ndarray(element_type, shape):
        """ Create an ndarray of the given element type and shape """
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.ctypeslib._ctype_ndarray', '_ctype_ndarray(element_type, shape)', {'element_type': element_type, 'shape': shape}, 1)
    
    def _get_scalar_type_map():
        """
        Return a dictionary mapping native endian scalar dtype to ctypes types
        """
        ct = ctypes
        simple_types = [ct.c_byte, ct.c_short, ct.c_int, ct.c_long, ct.c_longlong, ct.c_ubyte, ct.c_ushort, ct.c_uint, ct.c_ulong, ct.c_ulonglong, ct.c_float, ct.c_double, ct.c_bool]
        return {_dtype(ctype): ctype for ctype in simple_types}
    _scalar_type_map = _get_scalar_type_map()
    
    def _ctype_from_dtype_scalar(dtype):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.ctypeslib._ctype_from_dtype_scalar', '_ctype_from_dtype_scalar(dtype)', {'_scalar_type_map': _scalar_type_map, 'dtype': dtype}, 1)
    
    def _ctype_from_dtype_subarray(dtype):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.ctypeslib._ctype_from_dtype_subarray', '_ctype_from_dtype_subarray(dtype)', {'_ctype_from_dtype': _ctype_from_dtype, '_ctype_ndarray': _ctype_ndarray, 'dtype': dtype}, 1)
    
    def _ctype_from_dtype_structured(dtype):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.ctypeslib._ctype_from_dtype_structured', '_ctype_from_dtype_structured(dtype)', {'_ctype_from_dtype': _ctype_from_dtype, 'ctypes': ctypes, 'dtype': dtype}, 1)
    
    def _ctype_from_dtype(dtype):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.ctypeslib._ctype_from_dtype', '_ctype_from_dtype(dtype)', {'_ctype_from_dtype_structured': _ctype_from_dtype_structured, '_ctype_from_dtype_subarray': _ctype_from_dtype_subarray, '_ctype_from_dtype_scalar': _ctype_from_dtype_scalar, 'dtype': dtype}, 1)
    
    def as_ctypes_type(dtype):
        """
        Convert a dtype into a ctypes type.

        Parameters
        ----------
        dtype : dtype
            The dtype to convert

        Returns
        -------
        ctype
            A ctype scalar, union, array, or struct

        Raises
        ------
        NotImplementedError
            If the conversion is not possible

        Notes
        -----
        This function does not losslessly round-trip in either direction.

        ``np.dtype(as_ctypes_type(dt))`` will:

         - insert padding fields
         - reorder fields to be sorted by offset
         - discard field titles

        ``as_ctypes_type(np.dtype(ctype))`` will:

         - discard the class names of `ctypes.Structure`\ s and
           `ctypes.Union`\ s
         - convert single-element `ctypes.Union`\ s into single-element
           `ctypes.Structure`\ s
         - insert padding fields

        """
        return _ctype_from_dtype(_dtype(dtype))
    
    def as_array(obj, shape=None):
        """
        Create a numpy array from a ctypes array or POINTER.

        The numpy array shares the memory with the ctypes object.

        The shape parameter must be given if converting from a ctypes POINTER.
        The shape parameter is ignored if converting from a ctypes array
        """
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.ctypeslib.as_array', 'as_array(obj, shape=None)', {'ctypes': ctypes, '_ctype_ndarray': _ctype_ndarray, 'asarray': asarray, 'obj': obj, 'shape': shape}, 1)
    
    def as_ctypes(obj):
        """Create and return a ctypes object from a numpy array.  Actually
        anything that exposes the __array_interface__ is accepted."""
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.ctypeslib.as_ctypes', 'as_ctypes(obj)', {'as_ctypes_type': as_ctypes_type, '_ctype_ndarray': _ctype_ndarray, 'obj': obj}, 1)

