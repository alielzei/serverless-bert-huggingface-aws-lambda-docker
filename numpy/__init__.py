"""
NumPy
=====

Provides
  1. An array object of arbitrary homogeneous items
  2. Fast mathematical operations over arrays
  3. Linear Algebra, Fourier Transforms, Random Number Generation

How to use the documentation
----------------------------
Documentation is available in two forms: docstrings provided
with the code, and a loose standing reference guide, available from
`the NumPy homepage <https://numpy.org>`_.

We recommend exploring the docstrings using
`IPython <https://ipython.org>`_, an advanced Python shell with
TAB-completion and introspection capabilities.  See below for further
instructions.

The docstring examples assume that `numpy` has been imported as ``np``::

  >>> import numpy as np

Code snippets are indicated by three greater-than signs::

  >>> x = 42
  >>> x = x + 1

Use the built-in ``help`` function to view a function's docstring::

  >>> help(np.sort)
  ... # doctest: +SKIP

For some objects, ``np.info(obj)`` may provide additional help.  This is
particularly true if you see the line "Help on ufunc object:" at the top
of the help() page.  Ufuncs are implemented in C, not Python, for speed.
The native Python help() does not know how to view their help, but our
np.info() function does.

To search for documents containing a keyword, do::

  >>> np.lookfor('keyword')
  ... # doctest: +SKIP

General-purpose documents like a glossary and help on the basic concepts
of numpy are available under the ``doc`` sub-module::

  >>> from numpy import doc
  >>> help(doc)
  ... # doctest: +SKIP

Available subpackages
---------------------
lib
    Basic functions used by several sub-packages.
random
    Core Random Tools
linalg
    Core Linear Algebra Tools
fft
    Core FFT routines
polynomial
    Polynomial tools
testing
    NumPy testing tools
distutils
    Enhancements to distutils with support for
    Fortran compilers support and more.

Utilities
---------
test
    Run numpy unittests
show_config
    Show numpy build configuration
dual
    Overwrite certain functions with high-performance SciPy tools.
    Note: `numpy.dual` is deprecated.  Use the functions from NumPy or Scipy
    directly instead of importing them from `numpy.dual`.
matlib
    Make everything matrices.
__version__
    NumPy version string

Viewing documentation using IPython
-----------------------------------

Start IPython and import `numpy` usually under the alias ``np``: `import
numpy as np`.  Then, directly past or use the ``%cpaste`` magic to paste
examples into the shell.  To see which functions are available in `numpy`,
type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
down the list.  To view the docstring for a function, use
``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
the source code).

Copies vs. in-place operation
-----------------------------
Most of the functions in `numpy` return a copy of the array argument
(e.g., `np.sort`).  In-place versions of these functions are often
available as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.
Exceptions to this rule are documented.

"""

import sys
import warnings
from ._globals import ModuleDeprecationWarning, VisibleDeprecationWarning, _NoValue, _CopyMode
try:
    __NUMPY_SETUP__
except NameError:
    __NUMPY_SETUP__ = False
if __NUMPY_SETUP__:
    sys.stderr.write('Running from numpy source directory.\n')
else:
    try:
        from numpy.__config__ import show as show_config
    except ImportError as e:
        msg = 'Error importing numpy: you should not try to import numpy from\n        its source directory; please exit the numpy source tree, and relaunch\n        your python interpreter from there.'
        raise ImportError(msg) from e
    __all__ = ['ModuleDeprecationWarning', 'VisibleDeprecationWarning']
    __deprecated_attrs__ = {}
    from . import _distributor_init
    from . import core
    from .core import *
    from . import compat
    from . import lib
    from .lib import *
    from . import linalg
    from . import fft
    from . import polynomial
    from . import random
    from . import ctypeslib
    from . import ma
    from . import matrixlib as _mat
    from .matrixlib import *
    import builtins as _builtins
    _msg = "module 'numpy' has no attribute '{n}'.\n`np.{n}` was a deprecated alias for the builtin `{n}`. To avoid this error in existing code, use `{n}` by itself. Doing this will not modify any behavior and is safe. {extended_msg}\nThe aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:\n    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations"
    _specific_msg = 'If you specifically wanted the numpy scalar type, use `np.{}` here.'
    _int_extended_msg = 'When replacing `np.{}`, you may wish to use e.g. `np.int64` or `np.int32` to specify the precision. If you wish to review your current use, check the release note link for additional information.'
    _type_info = [('object', ''), ('bool', _specific_msg.format('bool_')), ('float', _specific_msg.format('float64')), ('complex', _specific_msg.format('complex128')), ('str', _specific_msg.format('str_')), ('int', _int_extended_msg.format('int'))]
    __former_attrs__ = {n: _msg.format(n=n, extended_msg=extended_msg) for (n, extended_msg) in _type_info}
    _msg = '`np.{n}` is a deprecated alias for `{an}`.  (Deprecated NumPy 1.24)'
    _type_info = [('bool8', bool_, 'np.bool_'), ('int0', intp, 'np.intp'), ('uint0', uintp, 'np.uintp'), ('str0', str_, 'np.str_'), ('bytes0', bytes_, 'np.bytes_'), ('void0', void, 'np.void'), ('object0', object_, '`np.object0` is a deprecated alias for `np.object_`. `object` can be used instead.  (Deprecated NumPy 1.24)')]
    __future_scalars__ = {'bool', 'long', 'ulong', 'str', 'bytes', 'object'}
    __deprecated_attrs__.update({n: (alias, _msg.format(n=n, an=an)) for (n, alias, an) in _type_info})
    del _msg, _type_info
    from .core import round, abs, max, min
    core.getlimits._register_known_types()
    __all__.extend(['__version__', 'show_config'])
    __all__.extend(core.__all__)
    __all__.extend(_mat.__all__)
    __all__.extend(lib.__all__)
    __all__.extend(['linalg', 'fft', 'random', 'ctypeslib', 'ma'])
    __all__.remove('issubdtype')
    del long, unicode
    __all__.remove('long')
    __all__.remove('unicode')
    __all__.remove('Arrayterator')
    del Arrayterator
    _financial_names = ['fv', 'ipmt', 'irr', 'mirr', 'nper', 'npv', 'pmt', 'ppmt', 'pv', 'rate']
    __expired_functions__ = {name: f'In accordance with NEP 32, the function {name} was removed from NumPy version 1.20.  A replacement for this function is available in the numpy_financial library: https://pypi.org/project/numpy-financial' for name in _financial_names}
    warnings.filterwarnings('ignore', message='numpy.dtype size changed')
    warnings.filterwarnings('ignore', message='numpy.ufunc size changed')
    warnings.filterwarnings('ignore', message='numpy.ndarray size changed')
    oldnumeric = 'removed'
    numarray = 'removed'
    
    def __getattr__(attr):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.__init__.__getattr__', '__getattr__(attr)', {'__expired_functions__': __expired_functions__, '__deprecated_attrs__': __deprecated_attrs__, '__future_scalars__': __future_scalars__, '__former_attrs__': __former_attrs__, '__name__': __name__, 'attr': attr}, 1)
    
    def __dir__():
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.__init__.__dir__', '__dir__()', {}, 1)
    from numpy._pytesttester import PytestTester
    test = PytestTester(__name__)
    del PytestTester
    
    def _sanity_check():
        """
        Quick sanity checks for common bugs caused by environment.
        There are some cases e.g. with wrong BLAS ABI that cause wrong
        results under specific runtime conditions that are not necessarily
        achieved during test suite runs, and it is useful to catch those early.

        See https://github.com/numpy/numpy/issues/8577 and other
        similar bug reports.

        """
        try:
            x = ones(2, dtype=float32)
            if not abs(x.dot(x) - float32(2.0)) < 1e-05:
                raise AssertionError()
        except AssertionError:
            msg = 'The current Numpy installation ({!r}) fails to pass simple sanity checks. This can be caused for example by incorrect BLAS library being linked in, or by mixing package managers (pip, conda, apt, ...). Search closed numpy issues for similar problems.'
            raise RuntimeError(msg.format(__file__)) from None
    _sanity_check()
    del _sanity_check
    
    def _mac_os_check():
        """
        Quick Sanity check for Mac OS look for accelerate build bugs.
        Testing numpy polyfit calls init_dgelsd(LAPACK)
        """
        try:
            c = array([3.0, 2.0, 1.0])
            x = linspace(0, 2, 5)
            y = polyval(c, x)
            _ = polyfit(x, y, 2, cov=True)
        except ValueError:
            pass
    if sys.platform == 'darwin':
        with warnings.catch_warnings(record=True) as w:
            _mac_os_check()
            error_message = ''
            if len(w) > 0:
                error_message = '{}: {}'.format(w[-1].category.__name__, str(w[-1].message))
                msg = 'Polyfit sanity test emitted a warning, most likely due to using a buggy Accelerate backend.\nIf you compiled yourself, more information is available at:\nhttps://numpy.org/doc/stable/user/building.html#accelerated-blas-lapack-libraries\nOtherwise report this to the vendor that provided NumPy.\n{}\n'.format(error_message)
                raise RuntimeError(msg)
    del _mac_os_check
    import os
    use_hugepage = os.environ.get('NUMPY_MADVISE_HUGEPAGE', None)
    if (sys.platform == 'linux' and use_hugepage is None):
        try:
            use_hugepage = 1
            kernel_version = os.uname().release.split('.')[:2]
            kernel_version = tuple((int(v) for v in kernel_version))
            if kernel_version < (4, 6):
                use_hugepage = 0
        except ValueError:
            use_hugepages = 0
    elif use_hugepage is None:
        use_hugepage = 1
    else:
        use_hugepage = int(use_hugepage)
    core.multiarray._set_madvise_hugepage(use_hugepage)
    core.multiarray._multiarray_umath._reload_guard()
    core._set_promotion_state(os.environ.get('NPY_PROMOTION_STATE', 'legacy'))
    
    def _pyinstaller_hooks_dir():
        from pathlib import Path
        return [str(Path(__file__).with_name('_pyinstaller').resolve())]
    del os
from .version import __version__, git_revision as __git_version__
del sys, warnings

