"""
Decorators for labeling and modifying behavior of test objects.

Decorators that merely return a modified version of the original
function object are straightforward. Decorators that return a new
function object need to use
::

  nose.tools.make_decorator(original_function)(decorator)

in returning the decorator, in order to preserve meta-data such as
function name, setup and teardown functions and so on - see
``nose.tools`` for more information.

"""

import collections.abc
import warnings
from .utils import SkipTest, assert_warns, HAS_REFCOUNT
__all__ = ['slow', 'setastest', 'skipif', 'knownfailureif', 'deprecated', 'parametrize', '_needs_refcount']

def slow(t):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Label a test as 'slow'.

    The exact definition of a slow test is obviously both subjective and
    hardware-dependent, but in general any individual test that requires more
    than a second or two should be labeled as slow (the whole suite consists of
    thousands of tests, so even a second is significant).

    Parameters
    ----------
    t : callable
        The test to label as slow.

    Returns
    -------
    t : callable
        The decorated test `t`.

    Examples
    --------
    The `numpy.testing` module includes ``import decorators as dec``.
    A test can be decorated as slow like this::

      from numpy.testing import *

      @dec.slow
      def test_big(self):
          print('Big, slow test')

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.decorators.slow', 'slow(t)', {'warnings': warnings, 't': t}, 1)

def setastest(tf=True):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Signals to nose that this function is or is not a test.

    Parameters
    ----------
    tf : bool
        If True, specifies that the decorated callable is a test.
        If False, specifies that the decorated callable is not a test.
        Default is True.

    Notes
    -----
    This decorator can't use the nose namespace, because it can be
    called from a non-test module. See also ``istest`` and ``nottest`` in
    ``nose.tools``.

    Examples
    --------
    `setastest` can be used in the following way::

      from numpy.testing import dec

      @dec.setastest(False)
      def func_with_test_in_name(arg1, arg2):
          pass

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.decorators.setastest', 'setastest(tf=True)', {'warnings': warnings, 'tf': tf}, 1)

def skipif(skip_condition, msg=None):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Make function raise SkipTest exception if a given condition is true.

    If the condition is a callable, it is used at runtime to dynamically
    make the decision. This is useful for tests that may require costly
    imports, to delay the cost until the test suite is actually executed.

    Parameters
    ----------
    skip_condition : bool or callable
        Flag to determine whether to skip the decorated test.
    msg : str, optional
        Message to give on raising a SkipTest exception. Default is None.

    Returns
    -------
    decorator : function
        Decorator which, when applied to a function, causes SkipTest
        to be raised when `skip_condition` is True, and the function
        to be called normally otherwise.

    Notes
    -----
    The decorator itself is decorated with the ``nose.tools.make_decorator``
    function in order to transmit function name, and various other metadata.

    """
    
    def skip_decorator(f):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.testing._private.decorators.skipif.skip_decorator', 'skip_decorator(f)', {'warnings': warnings, 'skip_condition': skip_condition, 'collections': collections, 'SkipTest': SkipTest, 'f': f}, 1)
    return skip_decorator

def knownfailureif(fail_condition, msg=None):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Make function raise KnownFailureException exception if given condition is true.

    If the condition is a callable, it is used at runtime to dynamically
    make the decision. This is useful for tests that may require costly
    imports, to delay the cost until the test suite is actually executed.

    Parameters
    ----------
    fail_condition : bool or callable
        Flag to determine whether to mark the decorated test as a known
        failure (if True) or not (if False).
    msg : str, optional
        Message to give on raising a KnownFailureException exception.
        Default is None.

    Returns
    -------
    decorator : function
        Decorator, which, when applied to a function, causes
        KnownFailureException to be raised when `fail_condition` is True,
        and the function to be called normally otherwise.

    Notes
    -----
    The decorator itself is decorated with the ``nose.tools.make_decorator``
    function in order to transmit function name, and various other metadata.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.decorators.knownfailureif', 'knownfailureif(fail_condition, msg=None)', {'warnings': warnings, 'collections': collections, 'fail_condition': fail_condition, 'msg': msg}, 1)

def deprecated(conditional=True):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Filter deprecation warnings while running the test suite.

    This decorator can be used to filter DeprecationWarning's, to avoid
    printing them during the test suite run, while checking that the test
    actually raises a DeprecationWarning.

    Parameters
    ----------
    conditional : bool or callable, optional
        Flag to determine whether to mark test as deprecated or not. If the
        condition is a callable, it is used at runtime to dynamically make the
        decision. Default is True.

    Returns
    -------
    decorator : function
        The `deprecated` decorator itself.

    Notes
    -----
    .. versionadded:: 1.4.0

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.decorators.deprecated', 'deprecated(conditional=True)', {'warnings': warnings, 'assert_warns': assert_warns, 'collections': collections, 'conditional': conditional}, 1)

def parametrize(vars, input):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Pytest compatibility class. This implements the simplest level of
    pytest.mark.parametrize for use in nose as an aid in making the transition
    to pytest. It achieves that by adding a dummy var parameter and ignoring
    the doc_func parameter of the base class. It does not support variable
    substitution by name, nor does it support nesting or classes. See the
    pytest documentation for usage.

    .. versionadded:: 1.14.0

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.decorators.parametrize', 'parametrize(vars, input)', {'warnings': warnings, 'vars': vars, 'input': input}, 1)
_needs_refcount = skipif(not HAS_REFCOUNT, 'python has no sys.getrefcount')

