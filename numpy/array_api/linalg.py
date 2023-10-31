from __future__ import annotations
from ._dtypes import _floating_dtypes, _numeric_dtypes
from ._manipulation_functions import reshape
from ._array_object import Array
from ..core.numeric import normalize_axis_tuple
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._typing import Literal, Optional, Sequence, Tuple, Union
from typing import NamedTuple
import numpy.linalg
import numpy as np


class EighResult(NamedTuple):
    eigenvalues: Array
    eigenvectors: Array



class QRResult(NamedTuple):
    Q: Array
    R: Array



class SlogdetResult(NamedTuple):
    sign: Array
    logabsdet: Array



class SVDResult(NamedTuple):
    U: Array
    S: Array
    Vh: Array


def cholesky(x: Array, /, *, upper: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.cholesky <numpy.linalg.cholesky>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in cholesky')
    L = np.linalg.cholesky(x._array)
    if upper:
        return Array._new(L).mT
    return Array._new(L)

def cross(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.cross <numpy.cross>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.cross', 'cross(x1: Array, x2: Array, /, axis: int = -1)', {'x1': x1, '_numeric_dtypes': _numeric_dtypes, 'x2': x2, 'Array': Array, 'np': np, 'axis': axis}, 1)

def det(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.det <numpy.linalg.det>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.det', 'det(x: Array, /)', {'x': x, '_floating_dtypes': _floating_dtypes, 'Array': Array, 'np': np}, 1)

def diagonal(x: Array, /, *, offset: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.diagonal <numpy.diagonal>`.

    See its docstring for more information.
    """
    return Array._new(np.diagonal(x._array, offset=offset, axis1=-2, axis2=-1))

def eigh(x: Array, /) -> EighResult:
    """
    Array API compatible wrapper for :py:func:`np.linalg.eigh <numpy.linalg.eigh>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.eigh', 'eigh(x: Array, /)', {'x': x, '_floating_dtypes': _floating_dtypes, 'EighResult': EighResult, 'Array': Array, 'np': np}, 1)

def eigvalsh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.eigvalsh <numpy.linalg.eigvalsh>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.eigvalsh', 'eigvalsh(x: Array, /)', {'x': x, '_floating_dtypes': _floating_dtypes, 'Array': Array, 'np': np}, 1)

def inv(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.inv <numpy.linalg.inv>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in inv')
    return Array._new(np.linalg.inv(x._array))

def matmul(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.matmul <numpy.matmul>`.

    See its docstring for more information.
    """
    if (x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes):
        raise TypeError('Only numeric dtypes are allowed in matmul')
    return Array._new(np.matmul(x1._array, x2._array))

def matrix_norm(x: Array, /, *, keepdims: bool = False, ord: Optional[Union[(int, float, Literal[('fro', 'nuc')])]] = 'fro') -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.norm <numpy.linalg.norm>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.matrix_norm', "matrix_norm(x: Array, /, keepdims: bool = False, ord: Optional[Union[(int, float, Literal[('fro', 'nuc')])]] = 'fro')", {'x': x, '_floating_dtypes': _floating_dtypes, 'Array': Array, 'np': np, 'keepdims': keepdims, 'ord': ord}, 1)

def matrix_power(x: Array, n: int, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.matrix_power <numpy.matrix_power>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.matrix_power', 'matrix_power(x: Array, n: int, /)', {'x': x, '_floating_dtypes': _floating_dtypes, 'Array': Array, 'np': np, 'n': n}, 1)

def matrix_rank(x: Array, /, *, rtol: Optional[Union[(float, Array)]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.matrix_rank <numpy.matrix_rank>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.matrix_rank', 'matrix_rank(x: Array, /, rtol: Optional[Union[(float, Array)]] = None)', {'x': x, 'np': np, 'Array': Array, 'rtol': rtol}, 1)

def matrix_transpose(x: Array, /) -> Array:
    if x.ndim < 2:
        raise ValueError('x must be at least 2-dimensional for matrix_transpose')
    return Array._new(np.swapaxes(x._array, -1, -2))

def outer(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.outer <numpy.outer>`.

    See its docstring for more information.
    """
    if (x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes):
        raise TypeError('Only numeric dtypes are allowed in outer')
    if (x1.ndim != 1 or x2.ndim != 1):
        raise ValueError('The input arrays to outer must be 1-dimensional')
    return Array._new(np.outer(x1._array, x2._array))

def pinv(x: Array, /, *, rtol: Optional[Union[(float, Array)]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.pinv <numpy.linalg.pinv>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in pinv')
    if rtol is None:
        rtol = max(x.shape[-2:]) * np.finfo(x.dtype).eps
    return Array._new(np.linalg.pinv(x._array, rcond=rtol))

def qr(x: Array, /, *, mode: Literal[('reduced', 'complete')] = 'reduced') -> QRResult:
    """
    Array API compatible wrapper for :py:func:`np.linalg.qr <numpy.linalg.qr>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in qr')
    return QRResult(*map(Array._new, np.linalg.qr(x._array, mode=mode)))

def slogdet(x: Array, /) -> SlogdetResult:
    """
    Array API compatible wrapper for :py:func:`np.linalg.slogdet <numpy.linalg.slogdet>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.slogdet', 'slogdet(x: Array, /)', {'x': x, '_floating_dtypes': _floating_dtypes, 'SlogdetResult': SlogdetResult, 'Array': Array, 'np': np}, 1)

def _solve(a, b):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg._solve', '_solve(a, b)', {'a': a, 'b': b}, 1)

def solve(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.solve <numpy.linalg.solve>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.solve', 'solve(x1: Array, x2: Array, /)', {'x1': x1, '_floating_dtypes': _floating_dtypes, 'x2': x2, 'Array': Array, '_solve': _solve}, 1)

def svd(x: Array, /, *, full_matrices: bool = True) -> SVDResult:
    """
    Array API compatible wrapper for :py:func:`np.linalg.svd <numpy.linalg.svd>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.svd', 'svd(x: Array, /, full_matrices: bool = True)', {'x': x, '_floating_dtypes': _floating_dtypes, 'SVDResult': SVDResult, 'Array': Array, 'np': np, 'full_matrices': full_matrices}, 1)

def svdvals(x: Array, /) -> Union[(Array, Tuple[(Array, ...)])]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.svdvals', 'svdvals(x: Array, /)', {'x': x, '_floating_dtypes': _floating_dtypes, 'Array': Array, 'np': np, 'Union': Union}, 1)

def tensordot(x1: Array, x2: Array, /, *, axes: Union[(int, Tuple[(Sequence[int], Sequence[int])])] = 2) -> Array:
    if (x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes):
        raise TypeError('Only numeric dtypes are allowed in tensordot')
    return Array._new(np.tensordot(x1._array, x2._array, axes=axes))

def trace(x: Array, /, *, offset: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.trace <numpy.trace>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.trace', 'trace(x: Array, /, offset: int = 0)', {'x': x, '_numeric_dtypes': _numeric_dtypes, 'Array': Array, 'np': np, 'offset': offset}, 1)

def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    if (x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes):
        raise TypeError('Only numeric dtypes are allowed in vecdot')
    ndim = max(x1.ndim, x2.ndim)
    x1_shape = (1, ) * (ndim - x1.ndim) + tuple(x1.shape)
    x2_shape = (1, ) * (ndim - x2.ndim) + tuple(x2.shape)
    if x1_shape[axis] != x2_shape[axis]:
        raise ValueError('x1 and x2 must have the same size along the given axis')
    (x1_, x2_) = np.broadcast_arrays(x1._array, x2._array)
    x1_ = np.moveaxis(x1_, axis, -1)
    x2_ = np.moveaxis(x2_, axis, -1)
    res = x1_[..., None, :] @ x2_[(..., None)]
    return Array._new(res[(..., 0, 0)])

def vector_norm(x: Array, /, *, axis: Optional[Union[(int, Tuple[(int, ...)])]] = None, keepdims: bool = False, ord: Optional[Union[(int, float)]] = 2) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.norm <numpy.linalg.norm>`.

    See its docstring for more information.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.array_api.linalg.vector_norm', 'vector_norm(x: Array, /, axis: Optional[Union[(int, Tuple[(int, ...)])]] = None, keepdims: bool = False, ord: Optional[Union[(int, float)]] = 2)', {'x': x, '_floating_dtypes': _floating_dtypes, 'normalize_axis_tuple': normalize_axis_tuple, 'np': np, 'Array': Array, 'reshape': reshape, 'axis': axis, 'keepdims': keepdims, 'ord': ord}, 1)
__all__ = ['cholesky', 'cross', 'det', 'diagonal', 'eigh', 'eigvalsh', 'inv', 'matmul', 'matrix_norm', 'matrix_power', 'matrix_rank', 'matrix_transpose', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'svdvals', 'tensordot', 'trace', 'vecdot', 'vector_norm']

