"""
An enhanced distutils, providing support for Fortran compilers, for BLAS,
LAPACK and other common libraries for numerical computing, and more.

Public submodules are::

    misc_util
    system_info
    cpu_info
    log
    exec_command

For details, please see the *Packaging* and *NumPy Distutils User Guide*
sections of the NumPy Reference Guide.

For configuring the preference for and location of libraries like BLAS and
LAPACK, and for setting include paths and similar build options, please see
``site.cfg.example`` in the root of the NumPy repository or sdist.

"""

import warnings
from . import ccompiler
from . import unixccompiler
from .npy_pkg_config import *
warnings.warn('\n\n  `numpy.distutils` is deprecated since NumPy 1.23.0, as a result\n  of the deprecation of `distutils` itself. It will be removed for\n  Python >= 3.12. For older Python versions it will remain present.\n  It is recommended to use `setuptools < 60.0` for those Python versions.\n  For more details, see:\n    https://numpy.org/devdocs/reference/distutils_status_migration.html \n\n', DeprecationWarning, stacklevel=2)
del warnings
try:
    from . import __config__
    from numpy._pytesttester import PytestTester
    test = PytestTester(__name__)
    del PytestTester
except ImportError:
    pass

def customized_fcompiler(plat=None, compiler=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.__init__.customized_fcompiler', 'customized_fcompiler(plat=None, compiler=None)', {'plat': plat, 'compiler': compiler}, 1)

def customized_ccompiler(plat=None, compiler=None, verbose=1):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.__init__.customized_ccompiler', 'customized_ccompiler(plat=None, compiler=None, verbose=1)', {'ccompiler': ccompiler, 'plat': plat, 'compiler': compiler, 'verbose': verbose}, 1)

