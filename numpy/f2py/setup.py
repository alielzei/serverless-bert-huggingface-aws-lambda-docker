"""
setup.py for installing F2PY

Usage:
   pip install .

Copyright 2001-2005 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Revision: 1.32 $
$Date: 2005/01/30 17:22:14 $
Pearu Peterson

"""

from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from __version__ import version

def configuration(parent_package='', top_path=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.setup.configuration', "configuration(parent_package='', top_path=None)", {'Configuration': Configuration, 'parent_package': parent_package, 'top_path': top_path}, 1)
if __name__ == '__main__':
    config = configuration(top_path='')
    config = config.todict()
    config['classifiers'] = ['Development Status :: 5 - Production/Stable', 'Intended Audience :: Developers', 'Intended Audience :: Science/Research', 'License :: OSI Approved :: NumPy License', 'Natural Language :: English', 'Operating System :: OS Independent', 'Programming Language :: C', 'Programming Language :: Fortran', 'Programming Language :: Python', 'Topic :: Scientific/Engineering', 'Topic :: Software Development :: Code Generators']
    setup(version=version, description='F2PY - Fortran to Python Interface Generator', author='Pearu Peterson', author_email='pearu@cens.ioc.ee', maintainer='Pearu Peterson', maintainer_email='pearu@cens.ioc.ee', license='BSD', platforms='Unix, Windows (mingw|cygwin), Mac OSX', long_description='The Fortran to Python Interface Generator, or F2PY for short, is a\ncommand line tool (f2py) for generating Python C/API modules for\nwrapping Fortran 77/90/95 subroutines, accessing common blocks from\nPython, and calling Python functions from Fortran (call-backs).\nInterfacing subroutines/data from Fortran 90/95 modules is supported.', url='https://numpy.org/doc/stable/f2py/', keywords=['Fortran', 'f2py'], **config)

