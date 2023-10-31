from __future__ import annotations
from ._array_object import Array
from ._dtypes import _all_dtypes, _result_type
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Tuple, Union
if TYPE_CHECKING:
    from ._typing import Dtype
    from collections.abc import Sequence
import numpy as np

def astype(x: Array, dtype: Dtype, /, *, copy: bool = True) -> Array:
    if (not copy and dtype == x.dtype):
        return x
    return Array._new(x._array.astype(dtype=dtype, copy=copy))

def broadcast_arrays(*arrays) -> List[Array]:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_arrays <numpy.broadcast_arrays>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    return [Array._new(array) for array in np.broadcast_arrays(*[a._array for a in arrays])]

def broadcast_to(x: Array, /, shape: Tuple[(int, ...)]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.broadcast_to <numpy.broadcast_to>`.

    See its docstring for more information.
    """
    from ._array_object import Array
    return Array._new(np.broadcast_to(x._array, shape))

def can_cast(from_: Union[(Dtype, Array)], to: Dtype, /) -> bool:
    """
    Array API compatible wrapper for :py:func:`np.can_cast <numpy.can_cast>`.

    See its docstring for more information.
    """
    if isinstance(from_, Array):
        from_ = from_.dtype
    elif from_ not in _all_dtypes:
        raise TypeError(f'from_={from_!r}, but should be an array_api array or dtype')
    if to not in _all_dtypes:
        raise TypeError(f'to={to!r}, but should be a dtype')
    try:
        dtype = _result_type(from_, to)
        return to == dtype
    except TypeError:
        return False


@dataclass
class finfo_object:
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float



@dataclass
class iinfo_object:
    bits: int
    max: int
    min: int


def finfo(type: Union[(Dtype, Array)], /) -> finfo_object:
    """
    Array API compatible wrapper for :py:func:`np.finfo <numpy.finfo>`.

    See its docstring for more information.
    """
    fi = np.finfo(type)
    return finfo_object(fi.bits, float(fi.eps), float(fi.max), float(fi.min), float(fi.smallest_normal))

def iinfo(type: Union[(Dtype, Array)], /) -> iinfo_object:
    """
    Array API compatible wrapper for :py:func:`np.iinfo <numpy.iinfo>`.

    See its docstring for more information.
    """
    ii = np.iinfo(type)
    return iinfo_object(ii.bits, ii.max, ii.min)

def result_type(*arrays_and_dtypes) -> Dtype:
    """
    Array API compatible wrapper for :py:func:`np.result_type <numpy.result_type>`.

    See its docstring for more information.
    """
    A = []
    for a in arrays_and_dtypes:
        if isinstance(a, Array):
            a = a.dtype
        elif (isinstance(a, np.ndarray) or a not in _all_dtypes):
            raise TypeError('result_type() inputs must be array_api arrays or dtypes')
        A.append(a)
    if len(A) == 0:
        raise ValueError('at least one array or dtype is required')
    elif len(A) == 1:
        return A[0]
    else:
        t = A[0]
        for t2 in A[1:]:
            t = _result_type(t, t2)
        return t

