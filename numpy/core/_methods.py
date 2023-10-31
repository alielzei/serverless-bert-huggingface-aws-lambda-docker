"""
Array methods which are called by both the C-code for the method
and the Python code for the NumPy-namespace function

"""

import warnings
from contextlib import nullcontext
from numpy.core import multiarray as mu
from numpy.core import umath as um
from numpy.core.multiarray import asanyarray
from numpy.core import numerictypes as nt
from numpy.core import _exceptions
from numpy.core._ufunc_config import _no_nep50_warning
from numpy._globals import _NoValue
from numpy.compat import pickle, os_fspath
umr_maximum = um.maximum.reduce
umr_minimum = um.minimum.reduce
umr_sum = um.add.reduce
umr_prod = um.multiply.reduce
umr_any = um.logical_or.reduce
umr_all = um.logical_and.reduce
_complex_to_float = {nt.dtype(nt.csingle): nt.dtype(nt.single), nt.dtype(nt.cdouble): nt.dtype(nt.double)}
if nt.dtype(nt.longdouble) != nt.dtype(nt.double):
    _complex_to_float.update({nt.dtype(nt.clongdouble): nt.dtype(nt.longdouble)})

def _amax(a, axis=None, out=None, keepdims=False, initial=_NoValue, where=True):
    return umr_maximum(a, axis, None, out, keepdims, initial, where)

def _amin(a, axis=None, out=None, keepdims=False, initial=_NoValue, where=True):
    return umr_minimum(a, axis, None, out, keepdims, initial, where)

def _sum(a, axis=None, dtype=None, out=None, keepdims=False, initial=_NoValue, where=True):
    return umr_sum(a, axis, dtype, out, keepdims, initial, where)

def _prod(a, axis=None, dtype=None, out=None, keepdims=False, initial=_NoValue, where=True):
    return umr_prod(a, axis, dtype, out, keepdims, initial, where)

def _any(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._methods._any', '_any(a, axis=None, dtype=None, out=None, keepdims=False, where=True)', {'umr_any': umr_any, 'a': a, 'axis': axis, 'dtype': dtype, 'out': out, 'keepdims': keepdims, 'where': where}, 1)

def _all(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._methods._all', '_all(a, axis=None, dtype=None, out=None, keepdims=False, where=True)', {'umr_all': umr_all, 'a': a, 'axis': axis, 'dtype': dtype, 'out': out, 'keepdims': keepdims, 'where': where}, 1)

def _count_reduce_items(arr, axis, keepdims=False, where=True):
    if where is True:
        if axis is None:
            axis = tuple(range(arr.ndim))
        elif not isinstance(axis, tuple):
            axis = (axis, )
        items = 1
        for ax in axis:
            items *= arr.shape[mu.normalize_axis_index(ax, arr.ndim)]
        items = nt.intp(items)
    else:
        from numpy.lib.stride_tricks import broadcast_to
        items = umr_sum(broadcast_to(where, arr.shape), axis, nt.intp, None, keepdims)
    return items

def _clip_dep_is_scalar_nan(a):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._methods._clip_dep_is_scalar_nan', '_clip_dep_is_scalar_nan(a)', {'um': um, 'a': a}, 1)

def _clip_dep_is_byte_swapped(a):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._methods._clip_dep_is_byte_swapped', '_clip_dep_is_byte_swapped(a)', {'mu': mu, 'a': a}, 1)

def _clip_dep_invoke_with_casting(ufunc, *args, out=None, casting=None, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._methods._clip_dep_invoke_with_casting', '_clip_dep_invoke_with_casting(ufunc, *args, out=None, casting=None, **kwargs)', {'_exceptions': _exceptions, 'warnings': warnings, 'ufunc': ufunc, 'out': out, 'casting': casting, 'args': args, 'kwargs': kwargs}, 1)

def _clip(a, min=None, max=None, out=None, *, casting=None, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core._methods._clip', '_clip(a, min=None, max=None, out=None, casting=None, **kwargs)', {'_clip_dep_is_byte_swapped': _clip_dep_is_byte_swapped, '_clip_dep_is_scalar_nan': _clip_dep_is_scalar_nan, 'warnings': warnings, '_clip_dep_invoke_with_casting': _clip_dep_invoke_with_casting, 'um': um, 'a': a, 'min': min, 'max': max, 'out': out, 'casting': casting, 'kwargs': kwargs}, 1)

def _mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    arr = asanyarray(a)
    is_float16_result = False
    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    if (rcount == 0 if where is True else umr_any(rcount == 0, axis=None)):
        warnings.warn('Mean of empty slice.', RuntimeWarning, stacklevel=2)
    if dtype is None:
        if issubclass(arr.dtype.type, (nt.integer, nt.bool_)):
            dtype = mu.dtype('f8')
        elif issubclass(arr.dtype.type, nt.float16):
            dtype = mu.dtype('f4')
            is_float16_result = True
    ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
    if isinstance(ret, mu.ndarray):
        with _no_nep50_warning():
            ret = um.true_divide(ret, rcount, out=ret, casting='unsafe', subok=False)
        if (is_float16_result and out is None):
            ret = arr.dtype.type(ret)
    elif hasattr(ret, 'dtype'):
        if is_float16_result:
            ret = arr.dtype.type(ret / rcount)
        else:
            ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount
    return ret

def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
    arr = asanyarray(a)
    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    if (ddof >= rcount if where is True else umr_any(ddof >= rcount, axis=None)):
        warnings.warn('Degrees of freedom <= 0 for slice', RuntimeWarning, stacklevel=2)
    if (dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool_))):
        dtype = mu.dtype('f8')
    arrmean = umr_sum(arr, axis, dtype, keepdims=True, where=where)
    if rcount.ndim == 0:
        div = rcount
    else:
        div = rcount.reshape(arrmean.shape)
    if isinstance(arrmean, mu.ndarray):
        with _no_nep50_warning():
            arrmean = um.true_divide(arrmean, div, out=arrmean, casting='unsafe', subok=False)
    elif hasattr(arrmean, 'dtype'):
        arrmean = arrmean.dtype.type(arrmean / rcount)
    else:
        arrmean = arrmean / rcount
    x = asanyarray(arr - arrmean)
    if issubclass(arr.dtype.type, (nt.floating, nt.integer)):
        x = um.multiply(x, x, out=x)
    elif x.dtype in _complex_to_float:
        xv = x.view(dtype=(_complex_to_float[x.dtype], (2, )))
        um.multiply(xv, xv, out=xv)
        x = um.add(xv[(..., 0)], xv[(..., 1)], out=x.real).real
    else:
        x = um.multiply(x, um.conjugate(x), out=x).real
    ret = umr_sum(x, axis, dtype, out, keepdims=keepdims, where=where)
    rcount = um.maximum(rcount - ddof, 0)
    if isinstance(ret, mu.ndarray):
        with _no_nep50_warning():
            ret = um.true_divide(ret, rcount, out=ret, casting='unsafe', subok=False)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount
    return ret

def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *, where=True):
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where)
    if isinstance(ret, mu.ndarray):
        ret = um.sqrt(ret, out=ret)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(um.sqrt(ret))
    else:
        ret = um.sqrt(ret)
    return ret

def _ptp(a, axis=None, out=None, keepdims=False):
    return um.subtract(umr_maximum(a, axis, None, out, keepdims), umr_minimum(a, axis, None, None, keepdims), out)

def _dump(self, file, protocol=2):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.core._methods._dump', '_dump(self, file, protocol=2)', {'nullcontext': nullcontext, 'os_fspath': os_fspath, 'pickle': pickle, 'self': self, 'file': file, 'protocol': protocol}, 0)

def _dumps(self, protocol=2):
    return pickle.dumps(self, protocol=protocol)

