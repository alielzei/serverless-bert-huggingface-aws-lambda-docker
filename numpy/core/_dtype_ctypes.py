"""
Conversion from ctypes to dtype.

In an ideal world, we could achieve this through the PEP3118 buffer protocol,
something like::

    def dtype_from_ctypes_type(t):
        # needed to ensure that the shape of `t` is within memoryview.format
        class DummyStruct(ctypes.Structure):
            _fields_ = [('a', t)]

        # empty to avoid memory allocation
        ctype_0 = (DummyStruct * 0)()
        mv = memoryview(ctype_0)

        # convert the struct, and slice back out the field
        return _dtype_from_pep3118(mv.format)['a']

Unfortunately, this fails because:

* ctypes cannot handle length-0 arrays with PEP3118 (bpo-32782)
* PEP3118 cannot represent unions, but both numpy and ctypes can
* ctypes cannot handle big-endian structs with PEP3118 (bpo-32780)
"""

import numpy as np

def _from_ctypes_array(t):
    return np.dtype((dtype_from_ctypes_type(t._type_), (t._length_, )))

def _from_ctypes_structure(t):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype_ctypes._from_ctypes_structure', '_from_ctypes_structure(t)', {'dtype_from_ctypes_type': dtype_from_ctypes_type, 'np': np, 't': t}, 1)

def _from_ctypes_scalar(t):
    """
    Return the dtype type with endianness included if it's the case
    """
    if getattr(t, '__ctype_be__', None) is t:
        return np.dtype('>' + t._type_)
    elif getattr(t, '__ctype_le__', None) is t:
        return np.dtype('<' + t._type_)
    else:
        return np.dtype(t._type_)

def _from_ctypes_union(t):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._dtype_ctypes._from_ctypes_union', '_from_ctypes_union(t)', {'dtype_from_ctypes_type': dtype_from_ctypes_type, 'np': np, 't': t}, 1)

def dtype_from_ctypes_type(t):
    """
    Construct a dtype object from a ctypes type
    """
    import _ctypes
    if issubclass(t, _ctypes.Array):
        return _from_ctypes_array(t)
    elif issubclass(t, _ctypes._Pointer):
        raise TypeError('ctypes pointers have no dtype equivalent')
    elif issubclass(t, _ctypes.Structure):
        return _from_ctypes_structure(t)
    elif issubclass(t, _ctypes.Union):
        return _from_ctypes_union(t)
    elif isinstance(getattr(t, '_type_', None), str):
        return _from_ctypes_scalar(t)
    else:
        raise NotImplementedError('Unknown ctypes type {}'.format(t.__name__))

