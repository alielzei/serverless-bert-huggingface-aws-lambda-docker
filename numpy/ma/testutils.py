"""Miscellaneous functions for testing masked arrays and subclasses

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: testutils.py 3529 2007-11-13 08:01:14Z jarrod.millman $

"""

import operator
import numpy as np
from numpy import ndarray, float_
import numpy.core.umath as umath
import numpy.testing
from numpy.testing import assert_, assert_allclose, assert_array_almost_equal_nulp, assert_raises, build_err_msg
from .core import mask_or, getmask, masked_array, nomask, masked, filled
__all__masked = ['almost', 'approx', 'assert_almost_equal', 'assert_array_almost_equal', 'assert_array_approx_equal', 'assert_array_compare', 'assert_array_equal', 'assert_array_less', 'assert_close', 'assert_equal', 'assert_equal_records', 'assert_mask_equal', 'assert_not_equal', 'fail_if_array_equal']
from unittest import TestCase
__some__from_testing = ['TestCase', 'assert_', 'assert_allclose', 'assert_array_almost_equal_nulp', 'assert_raises']
__all__ = __all__masked + __some__from_testing

def approx(a, b, fill_value=True, rtol=1e-05, atol=1e-08):
    """
    Returns true if all components of a and b are equal to given tolerances.

    If fill_value is True, masked values considered equal. Otherwise,
    masked values are considered unequal.  The relative error rtol should
    be positive and << 1.0 The absolute error atol comes into play for
    those elements of b that are very small or zero; it says how small a
    must be also.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.testutils.approx', 'approx(a, b, fill_value=True, rtol=1e-05, atol=1e-08)', {'mask_or': mask_or, 'getmask': getmask, 'filled': filled, 'np': np, 'masked_array': masked_array, 'float_': float_, 'umath': umath, 'a': a, 'b': b, 'fill_value': fill_value, 'rtol': rtol, 'atol': atol}, 1)

def almost(a, b, decimal=6, fill_value=True):
    """
    Returns True if a and b are equal up to decimal places.

    If fill_value is True, masked values considered equal. Otherwise,
    masked values are considered unequal.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.testutils.almost', 'almost(a, b, decimal=6, fill_value=True)', {'mask_or': mask_or, 'getmask': getmask, 'filled': filled, 'np': np, 'masked_array': masked_array, 'float_': float_, 'a': a, 'b': b, 'decimal': decimal, 'fill_value': fill_value}, 1)

def _assert_equal_on_sequences(actual, desired, err_msg=''):
    """
    Asserts the equality of two non-array sequences.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.testutils._assert_equal_on_sequences', "_assert_equal_on_sequences(actual, desired, err_msg='')", {'assert_equal': assert_equal, 'actual': actual, 'desired': desired, 'err_msg': err_msg}, 1)

def assert_equal_records(a, b):
    """
    Asserts that two records are equal.

    Pretty crude for now.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.testutils.assert_equal_records', 'assert_equal_records(a, b)', {'assert_equal': assert_equal, 'operator': operator, 'masked': masked, 'a': a, 'b': b}, 1)

def assert_equal(actual, desired, err_msg=''):
    """
    Asserts that two items are equal.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.testutils.assert_equal', "assert_equal(actual, desired, err_msg='')", {'assert_equal': assert_equal, '_assert_equal_on_sequences': _assert_equal_on_sequences, 'ndarray': ndarray, 'build_err_msg': build_err_msg, 'masked': masked, 'np': np, 'assert_array_equal': assert_array_equal, 'actual': actual, 'desired': desired, 'err_msg': err_msg}, 1)

def fail_if_equal(actual, desired, err_msg=''):
    """
    Raises an assertion error if two items are equal.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.testutils.fail_if_equal', "fail_if_equal(actual, desired, err_msg='')", {'fail_if_equal': fail_if_equal, 'np': np, 'fail_if_array_equal': fail_if_array_equal, 'build_err_msg': build_err_msg, 'actual': actual, 'desired': desired, 'err_msg': err_msg}, 1)
assert_not_equal = fail_if_equal

def assert_almost_equal(actual, desired, decimal=7, err_msg='', verbose=True):
    """
    Asserts that two items are almost equal.

    The test is equivalent to abs(desired-actual) < 0.5 * 10**(-decimal).

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.testutils.assert_almost_equal', "assert_almost_equal(actual, desired, decimal=7, err_msg='', verbose=True)", {'np': np, 'assert_array_almost_equal': assert_array_almost_equal, 'build_err_msg': build_err_msg, 'actual': actual, 'desired': desired, 'decimal': decimal, 'err_msg': err_msg, 'verbose': verbose}, 1)
assert_close = assert_almost_equal

def assert_array_compare(comparison, x, y, err_msg='', verbose=True, header='', fill_value=True):
    """
    Asserts that comparison between two masked arrays is satisfied.

    The comparison is elementwise.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.testutils.assert_array_compare', "assert_array_compare(comparison, x, y, err_msg='', verbose=True, header='', fill_value=True)", {'mask_or': mask_or, 'getmask': getmask, 'masked_array': masked_array, 'masked': masked, 'build_err_msg': build_err_msg, 'np': np, 'comparison': comparison, 'x': x, 'y': y, 'err_msg': err_msg, 'verbose': verbose, 'header': header, 'fill_value': fill_value}, 1)

def assert_array_equal(x, y, err_msg='', verbose=True):
    """
    Checks the elementwise equality of two masked arrays.

    """
    assert_array_compare(operator.__eq__, x, y, err_msg=err_msg, verbose=verbose, header='Arrays are not equal')

def fail_if_array_equal(x, y, err_msg='', verbose=True):
    """
    Raises an assertion error if two masked arrays are not equal elementwise.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.testutils.fail_if_array_equal', "fail_if_array_equal(x, y, err_msg='', verbose=True)", {'np': np, 'approx': approx, 'assert_array_compare': assert_array_compare, 'x': x, 'y': y, 'err_msg': err_msg, 'verbose': verbose}, 1)

def assert_array_approx_equal(x, y, decimal=6, err_msg='', verbose=True):
    """
    Checks the equality of two masked arrays, up to given number odecimals.

    The equality is checked elementwise.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.testutils.assert_array_approx_equal', "assert_array_approx_equal(x, y, decimal=6, err_msg='', verbose=True)", {'approx': approx, 'assert_array_compare': assert_array_compare, 'x': x, 'y': y, 'decimal': decimal, 'err_msg': err_msg, 'verbose': verbose}, 1)

def assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True):
    """
    Checks the equality of two masked arrays, up to given number odecimals.

    The equality is checked elementwise.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.ma.testutils.assert_array_almost_equal', "assert_array_almost_equal(x, y, decimal=6, err_msg='', verbose=True)", {'almost': almost, 'assert_array_compare': assert_array_compare, 'x': x, 'y': y, 'decimal': decimal, 'err_msg': err_msg, 'verbose': verbose}, 1)

def assert_array_less(x, y, err_msg='', verbose=True):
    """
    Checks that x is smaller than y elementwise.

    """
    assert_array_compare(operator.__lt__, x, y, err_msg=err_msg, verbose=verbose, header='Arrays are not less-ordered')

def assert_mask_equal(m1, m2, err_msg=''):
    """
    Asserts the equality of two masks.

    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.ma.testutils.assert_mask_equal', "assert_mask_equal(m1, m2, err_msg='')", {'nomask': nomask, 'assert_': assert_, 'assert_array_equal': assert_array_equal, 'm1': m1, 'm2': m2, 'err_msg': err_msg}, 0)

