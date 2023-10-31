"""

f2py2e - Fortran to Python C/API generator. 2nd Edition.
         See __usage__ below.

Copyright 1999--2011 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy License.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/05/06 08:31:19 $
Pearu Peterson

"""

import sys
import os
import pprint
import re
from pathlib import Path
from . import crackfortran
from . import rules
from . import cb_rules
from . import auxfuncs
from . import cfuncs
from . import f90mod_rules
from . import __version__
from . import capi_maps
f2py_version = __version__.version
numpy_version = __version__.version
errmess = sys.stderr.write
show = pprint.pprint
outmess = auxfuncs.outmess
__usage__ = f"Usage:\n\n1) To construct extension module sources:\n\n      f2py [<options>] <fortran files> [[[only:]||[skip:]] \\\n                                        <fortran functions> ] \\\n                                       [: <fortran files> ...]\n\n2) To compile fortran files and build extension modules:\n\n      f2py -c [<options>, <build_flib options>, <extra options>] <fortran files>\n\n3) To generate signature files:\n\n      f2py -h <filename.pyf> ...< same options as in (1) >\n\nDescription: This program generates a Python C/API file (<modulename>module.c)\n             that contains wrappers for given fortran functions so that they\n             can be called from Python. With the -c option the corresponding\n             extension modules are built.\n\nOptions:\n\n  --2d-numpy       Use numpy.f2py tool with NumPy support. [DEFAULT]\n  --2d-numeric     Use f2py2e tool with Numeric support.\n  --2d-numarray    Use f2py2e tool with Numarray support.\n  --g3-numpy       Use 3rd generation f2py from the separate f2py package.\n                   [NOT AVAILABLE YET]\n\n  -h <filename>    Write signatures of the fortran routines to file <filename>\n                   and exit. You can then edit <filename> and use it instead\n                   of <fortran files>. If <filename>==stdout then the\n                   signatures are printed to stdout.\n  <fortran functions>  Names of fortran routines for which Python C/API\n                   functions will be generated. Default is all that are found\n                   in <fortran files>.\n  <fortran files>  Paths to fortran/signature files that will be scanned for\n                   <fortran functions> in order to determine their signatures.\n  skip:            Ignore fortran functions that follow until `:'.\n  only:            Use only fortran functions that follow until `:'.\n  :                Get back to <fortran files> mode.\n\n  -m <modulename>  Name of the module; f2py generates a Python/C API\n                   file <modulename>module.c or extension module <modulename>.\n                   Default is 'untitled'.\n\n  '-include<header>'  Writes additional headers in the C wrapper, can be passed\n                      multiple times, generates #include <header> each time.\n\n  --[no-]lower     Do [not] lower the cases in <fortran files>. By default,\n                   --lower is assumed with -h key, and --no-lower without -h key.\n\n  --build-dir <dirname>  All f2py generated files are created in <dirname>.\n                   Default is tempfile.mkdtemp().\n\n  --overwrite-signature  Overwrite existing signature file.\n\n  --[no-]latex-doc Create (or not) <modulename>module.tex.\n                   Default is --no-latex-doc.\n  --short-latex    Create 'incomplete' LaTeX document (without commands\n                   \\documentclass, \\tableofcontents, and \\begin{{document}},\n                   \\end{{document}}).\n\n  --[no-]rest-doc Create (or not) <modulename>module.rst.\n                   Default is --no-rest-doc.\n\n  --debug-capi     Create C/API code that reports the state of the wrappers\n                   during runtime. Useful for debugging.\n\n  --[no-]wrap-functions    Create Fortran subroutine wrappers to Fortran 77\n                   functions. --wrap-functions is default because it ensures\n                   maximum portability/compiler independence.\n\n  --include-paths <path1>:<path2>:...   Search include files from the given\n                   directories.\n\n  --help-link [..] List system resources found by system_info.py. See also\n                   --link-<resource> switch below. [..] is optional list\n                   of resources names. E.g. try 'f2py --help-link lapack_opt'.\n\n  --f2cmap <filename>  Load Fortran-to-Python KIND specification from the given\n                   file. Default: .f2py_f2cmap in current directory.\n\n  --quiet          Run quietly.\n  --verbose        Run with extra verbosity.\n  --skip-empty-wrappers   Only generate wrapper files when needed.\n  -v               Print f2py version ID and exit.\n\n\nnumpy.distutils options (only effective with -c):\n\n  --fcompiler=         Specify Fortran compiler type by vendor\n  --compiler=          Specify C compiler type (as defined by distutils)\n\n  --help-fcompiler     List available Fortran compilers and exit\n  --f77exec=           Specify the path to F77 compiler\n  --f90exec=           Specify the path to F90 compiler\n  --f77flags=          Specify F77 compiler flags\n  --f90flags=          Specify F90 compiler flags\n  --opt=               Specify optimization flags\n  --arch=              Specify architecture specific optimization flags\n  --noopt              Compile without optimization\n  --noarch             Compile without arch-dependent optimization\n  --debug              Compile with debugging information\n\nExtra options (only effective with -c):\n\n  --link-<resource>    Link extension module with <resource> as defined\n                       by numpy.distutils/system_info.py. E.g. to link\n                       with optimized LAPACK libraries (vecLib on MacOSX,\n                       ATLAS elsewhere), use --link-lapack_opt.\n                       See also --help-link switch.\n\n  -L/path/to/lib/ -l<libname>\n  -D<define> -U<name>\n  -I/path/to/include/\n  <filename>.o <filename>.so <filename>.a\n\n  Using the following macros may be required with non-gcc Fortran\n  compilers:\n    -DPREPEND_FORTRAN -DNO_APPEND_FORTRAN -DUPPERCASE_FORTRAN\n    -DUNDERSCORE_G77\n\n  When using -DF2PY_REPORT_ATEXIT, a performance report of F2PY\n  interface is printed out at exit (platforms: Linux).\n\n  When using -DF2PY_REPORT_ON_ARRAY_COPY=<int>, a message is\n  sent to stderr whenever F2PY interface makes a copy of an\n  array. Integer <int> sets the threshold for array sizes when\n  a message should be shown.\n\nVersion:     {f2py_version}\nnumpy Version: {numpy_version}\nRequires:    Python 3.5 or higher.\nLicense:     NumPy license (see LICENSE.txt in the NumPy source code)\nCopyright 1999 - 2011 Pearu Peterson all rights reserved.\nhttps://web.archive.org/web/20140822061353/http://cens.ioc.ee/projects/f2py2e"

def scaninputline(inputline):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.f2py2e.scaninputline', 'scaninputline(inputline)', {'f2py_version': f2py_version, 'sys': sys, 'cfuncs': cfuncs, 'outmess': outmess, 'errmess': errmess, 'os': os, '__usage__': __usage__, 'inputline': inputline}, 2)

def callcrackfortran(files, options):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.f2py2e.callcrackfortran', 'callcrackfortran(files, options)', {'rules': rules, 'crackfortran': crackfortran, 'outmess': outmess, 'sys': sys, 'files': files, 'options': options}, 1)

def buildmodules(lst):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.f2py2e.buildmodules', 'buildmodules(lst)', {'cfuncs': cfuncs, 'outmess': outmess, 'cb_rules': cb_rules, 'dict_append': dict_append, 'rules': rules, 'lst': lst}, 1)

def dict_append(d_out, d_in):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.f2py.f2py2e.dict_append', 'dict_append(d_out, d_in)', {'d_out': d_out, 'd_in': d_in}, 0)

def run_main(comline_list):
    """
    Equivalent to running::

        f2py <args>

    where ``<args>=string.join(<list>,' ')``, but in Python.  Unless
    ``-h`` is used, this function returns a dictionary containing
    information on generated modules and their dependencies on source
    files.

    You cannot build extension modules with this function, that is,
    using ``-c`` is not allowed. Use the ``compile`` command instead.

    Examples
    --------
    The command ``f2py -m scalar scalar.f`` can be executed from Python as
    follows.

    .. literalinclude:: ../../source/f2py/code/results/run_main_session.dat
        :language: python

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.f2py2e.run_main', 'run_main(comline_list)', {'crackfortran': crackfortran, 'os': os, 'cfuncs': cfuncs, 'scaninputline': scaninputline, 'auxfuncs': auxfuncs, 'capi_maps': capi_maps, 'callcrackfortran': callcrackfortran, 'outmess': outmess, 'sys': sys, 'errmess': errmess, 'f90mod_rules': f90mod_rules, 'buildmodules': buildmodules, 'dict_append': dict_append, 'comline_list': comline_list}, 1)

def filter_files(prefix, suffix, files, remove_prefix=None):
    """
    Filter files by prefix and suffix.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.f2py2e.filter_files', 'filter_files(prefix, suffix, files, remove_prefix=None)', {'re': re, 'prefix': prefix, 'suffix': suffix, 'files': files, 'remove_prefix': remove_prefix}, 2)

def get_prefix(module):
    p = os.path.dirname(os.path.dirname(module.__file__))
    return p

def run_compile():
    """
    Do it all in one call!
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.f2py.f2py2e.run_compile', 'run_compile()', {'sys': sys, 're': re, 'filter_files': filter_files, 'outmess': outmess, 'os': os}, 0)

def main():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.f2py.f2py2e.main', 'main()', {'sys': sys, 'run_compile': run_compile, 'run_main': run_main}, 1)

