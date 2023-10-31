"""
=========
Constants
=========

.. currentmodule:: numpy

NumPy includes several constants:

%(constant_list)s
"""

import re
import textwrap
constants = []

def add_newdoc(module, name, doc):
    constants.append((name, doc))
add_newdoc('numpy', 'pi', '\n    ``pi = 3.1415926535897932384626433...``\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Pi\n\n    ')
add_newdoc('numpy', 'e', "\n    Euler's constant, base of natural logarithms, Napier's constant.\n\n    ``e = 2.71828182845904523536028747135266249775724709369995...``\n\n    See Also\n    --------\n    exp : Exponential function\n    log : Natural logarithm\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/E_%28mathematical_constant%29\n\n    ")
add_newdoc('numpy', 'euler_gamma', '\n    ``γ = 0.5772156649015328606065120900824024310421...``\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Euler-Mascheroni_constant\n\n    ')
add_newdoc('numpy', 'inf', '\n    IEEE 754 floating point representation of (positive) infinity.\n\n    Returns\n    -------\n    y : float\n        A floating point representation of positive infinity.\n\n    See Also\n    --------\n    isinf : Shows which elements are positive or negative infinity\n\n    isposinf : Shows which elements are positive infinity\n\n    isneginf : Shows which elements are negative infinity\n\n    isnan : Shows which elements are Not a Number\n\n    isfinite : Shows which elements are finite (not one of Not a Number,\n    positive infinity and negative infinity)\n\n    Notes\n    -----\n    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754). This means that Not a Number is not equivalent to infinity.\n    Also that positive infinity is not equivalent to negative infinity. But\n    infinity is equivalent to positive infinity.\n\n    `Inf`, `Infinity`, `PINF` and `infty` are aliases for `inf`.\n\n    Examples\n    --------\n    >>> np.inf\n    inf\n    >>> np.array([1]) / 0.\n    array([ Inf])\n\n    ')
add_newdoc('numpy', 'nan', '\n    IEEE 754 floating point representation of Not a Number (NaN).\n\n    Returns\n    -------\n    y : A floating point representation of Not a Number.\n\n    See Also\n    --------\n    isnan : Shows which elements are Not a Number.\n\n    isfinite : Shows which elements are finite (not one of\n    Not a Number, positive infinity and negative infinity)\n\n    Notes\n    -----\n    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754). This means that Not a Number is not equivalent to infinity.\n\n    `NaN` and `NAN` are aliases of `nan`.\n\n    Examples\n    --------\n    >>> np.nan\n    nan\n    >>> np.log(-1)\n    nan\n    >>> np.log([-1, 1, 2])\n    array([        NaN,  0.        ,  0.69314718])\n\n    ')
add_newdoc('numpy', 'newaxis', '\n    A convenient alias for None, useful for indexing arrays.\n\n    Examples\n    --------\n    >>> newaxis is None\n    True\n    >>> x = np.arange(3)\n    >>> x\n    array([0, 1, 2])\n    >>> x[:, newaxis]\n    array([[0],\n    [1],\n    [2]])\n    >>> x[:, newaxis, newaxis]\n    array([[[0]],\n    [[1]],\n    [[2]]])\n    >>> x[:, newaxis] * x\n    array([[0, 0, 0],\n    [0, 1, 2],\n    [0, 2, 4]])\n\n    Outer product, same as ``outer(x, y)``:\n\n    >>> y = np.arange(3, 6)\n    >>> x[:, newaxis] * y\n    array([[ 0,  0,  0],\n    [ 3,  4,  5],\n    [ 6,  8, 10]])\n\n    ``x[newaxis, :]`` is equivalent to ``x[newaxis]`` and ``x[None]``:\n\n    >>> x[newaxis, :].shape\n    (1, 3)\n    >>> x[newaxis].shape\n    (1, 3)\n    >>> x[None].shape\n    (1, 3)\n    >>> x[:, newaxis].shape\n    (3, 1)\n\n    ')
add_newdoc('numpy', 'NZERO', '\n    IEEE 754 floating point representation of negative zero.\n\n    Returns\n    -------\n    y : float\n        A floating point representation of negative zero.\n\n    See Also\n    --------\n    PZERO : Defines positive zero.\n\n    isinf : Shows which elements are positive or negative infinity.\n\n    isposinf : Shows which elements are positive infinity.\n\n    isneginf : Shows which elements are negative infinity.\n\n    isnan : Shows which elements are Not a Number.\n\n    isfinite : Shows which elements are finite - not one of\n               Not a Number, positive infinity and negative infinity.\n\n    Notes\n    -----\n    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754). Negative zero is considered to be a finite number.\n\n    Examples\n    --------\n    >>> np.NZERO\n    -0.0\n    >>> np.PZERO\n    0.0\n\n    >>> np.isfinite([np.NZERO])\n    array([ True])\n    >>> np.isnan([np.NZERO])\n    array([False])\n    >>> np.isinf([np.NZERO])\n    array([False])\n\n    ')
add_newdoc('numpy', 'PZERO', '\n    IEEE 754 floating point representation of positive zero.\n\n    Returns\n    -------\n    y : float\n        A floating point representation of positive zero.\n\n    See Also\n    --------\n    NZERO : Defines negative zero.\n\n    isinf : Shows which elements are positive or negative infinity.\n\n    isposinf : Shows which elements are positive infinity.\n\n    isneginf : Shows which elements are negative infinity.\n\n    isnan : Shows which elements are Not a Number.\n\n    isfinite : Shows which elements are finite - not one of\n               Not a Number, positive infinity and negative infinity.\n\n    Notes\n    -----\n    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754). Positive zero is considered to be a finite number.\n\n    Examples\n    --------\n    >>> np.PZERO\n    0.0\n    >>> np.NZERO\n    -0.0\n\n    >>> np.isfinite([np.PZERO])\n    array([ True])\n    >>> np.isnan([np.PZERO])\n    array([False])\n    >>> np.isinf([np.PZERO])\n    array([False])\n\n    ')
add_newdoc('numpy', 'NAN', '\n    IEEE 754 floating point representation of Not a Number (NaN).\n\n    `NaN` and `NAN` are equivalent definitions of `nan`. Please use\n    `nan` instead of `NAN`.\n\n    See Also\n    --------\n    nan\n\n    ')
add_newdoc('numpy', 'NaN', '\n    IEEE 754 floating point representation of Not a Number (NaN).\n\n    `NaN` and `NAN` are equivalent definitions of `nan`. Please use\n    `nan` instead of `NaN`.\n\n    See Also\n    --------\n    nan\n\n    ')
add_newdoc('numpy', 'NINF', '\n    IEEE 754 floating point representation of negative infinity.\n\n    Returns\n    -------\n    y : float\n        A floating point representation of negative infinity.\n\n    See Also\n    --------\n    isinf : Shows which elements are positive or negative infinity\n\n    isposinf : Shows which elements are positive infinity\n\n    isneginf : Shows which elements are negative infinity\n\n    isnan : Shows which elements are Not a Number\n\n    isfinite : Shows which elements are finite (not one of Not a Number,\n    positive infinity and negative infinity)\n\n    Notes\n    -----\n    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic\n    (IEEE 754). This means that Not a Number is not equivalent to infinity.\n    Also that positive infinity is not equivalent to negative infinity. But\n    infinity is equivalent to positive infinity.\n\n    Examples\n    --------\n    >>> np.NINF\n    -inf\n    >>> np.log(0)\n    -inf\n\n    ')
add_newdoc('numpy', 'PINF', '\n    IEEE 754 floating point representation of (positive) infinity.\n\n    Use `inf` because `Inf`, `Infinity`, `PINF` and `infty` are aliases for\n    `inf`. For more details, see `inf`.\n\n    See Also\n    --------\n    inf\n\n    ')
add_newdoc('numpy', 'infty', '\n    IEEE 754 floating point representation of (positive) infinity.\n\n    Use `inf` because `Inf`, `Infinity`, `PINF` and `infty` are aliases for\n    `inf`. For more details, see `inf`.\n\n    See Also\n    --------\n    inf\n\n    ')
add_newdoc('numpy', 'Inf', '\n    IEEE 754 floating point representation of (positive) infinity.\n\n    Use `inf` because `Inf`, `Infinity`, `PINF` and `infty` are aliases for\n    `inf`. For more details, see `inf`.\n\n    See Also\n    --------\n    inf\n\n    ')
add_newdoc('numpy', 'Infinity', '\n    IEEE 754 floating point representation of (positive) infinity.\n\n    Use `inf` because `Inf`, `Infinity`, `PINF` and `infty` are aliases for\n    `inf`. For more details, see `inf`.\n\n    See Also\n    --------\n    inf\n\n    ')
if __doc__:
    constants_str = []
    constants.sort()
    for (name, doc) in constants:
        s = textwrap.dedent(doc).replace('\n', '\n    ')
        lines = s.split('\n')
        new_lines = []
        for line in lines:
            m = re.match('^(\\s+)[-=]+\\s*$', line)
            if (m and new_lines):
                prev = textwrap.dedent(new_lines.pop())
                new_lines.append('%s.. rubric:: %s' % (m.group(1), prev))
                new_lines.append('')
            else:
                new_lines.append(line)
        s = '\n'.join(new_lines)
        constants_str.append('.. data:: %s\n    %s' % (name, s))
    constants_str = '\n'.join(constants_str)
    __doc__ = __doc__ % dict(constant_list=constants_str)
    del constants_str, name, doc
    del line, lines, new_lines, m, s, prev
del constants, add_newdoc

