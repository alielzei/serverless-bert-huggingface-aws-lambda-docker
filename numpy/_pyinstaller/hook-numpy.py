"""This hook should collect all binary files and any hidden modules that numpy
needs.

Our (some-what inadequate) docs for writing PyInstaller hooks are kept here:
https://pyinstaller.readthedocs.io/en/stable/hooks.html

"""

from PyInstaller.compat import is_conda, is_pure_conda
from PyInstaller.utils.hooks import collect_dynamic_libs, is_module_satisfies
binaries = collect_dynamic_libs('numpy', '.')
if is_pure_conda:
    from PyInstaller.utils.hooks import conda_support
    datas = conda_support.collect_dynamic_libs('numpy', dependencies=True)
hiddenimports = ['numpy.core._dtype_ctypes']
if is_conda:
    hiddenimports.append('six')
excludedimports = ['scipy', 'pytest', 'nose', 'f2py', 'setuptools', 'numpy.f2py', 'distutils', 'numpy.distutils']

