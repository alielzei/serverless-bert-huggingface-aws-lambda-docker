import re
import sys
import subprocess
__doc__ = 'This module generates a DEF file from the symbols in\nan MSVC-compiled DLL import library.  It correctly discriminates between\ndata and functions.  The data is collected from the output of the program\nnm(1).\n\nUsage:\n    python lib2def.py [libname.lib] [output.def]\nor\n    python lib2def.py [libname.lib] > output.def\n\nlibname.lib defaults to python<py_ver>.lib and output.def defaults to stdout\n\nAuthor: Robert Kern <kernr@mail.ncifcrf.gov>\nLast Update: April 30, 1999\n'
__version__ = '0.1a'
py_ver = '%d%d' % tuple(sys.version_info[:2])
DEFAULT_NM = ['nm', '-Cs']
DEF_HEADER = 'LIBRARY         python%s.dll\n;CODE           PRELOAD MOVEABLE DISCARDABLE\n;DATA           PRELOAD SINGLE\n\nEXPORTS\n' % py_ver
FUNC_RE = re.compile('^(.*) in python%s\\.dll' % py_ver, re.MULTILINE)
DATA_RE = re.compile('^_imp__(.*) in python%s\\.dll' % py_ver, re.MULTILINE)

def parse_cmd():
    """Parses the command-line arguments.

libfile, deffile = parse_cmd()"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.lib2def.parse_cmd', 'parse_cmd()', {'sys': sys, 'py_ver': py_ver}, 2)

def getnm(nm_cmd=['nm', '-Cs', 'python%s.lib' % py_ver], shell=True):
    """Returns the output of nm_cmd via a pipe.

nm_output = getnm(nm_cmd = 'nm -Cs py_lib')"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.lib2def.getnm', "getnm(nm_cmd=['nm', '-Cs', 'python%s.lib' % py_ver], shell=True)", {'subprocess': subprocess, 'nm_cmd': nm_cmd, 'shell': shell}, 1)

def parse_nm(nm_output):
    """Returns a tuple of lists: dlist for the list of data
symbols and flist for the list of function symbols.

dlist, flist = parse_nm(nm_output)"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.lib2def.parse_nm', 'parse_nm(nm_output)', {'DATA_RE': DATA_RE, 'FUNC_RE': FUNC_RE, 'nm_output': nm_output}, 2)

def output_def(dlist, flist, header, file=sys.stdout):
    """Outputs the final DEF file to a file defaulting to stdout.

output_def(dlist, flist, header, file = sys.stdout)"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.lib2def.output_def', 'output_def(dlist, flist, header, file=sys.stdout)', {'dlist': dlist, 'flist': flist, 'header': header, 'file': file}, 0)
if __name__ == '__main__':
    (libfile, deffile) = parse_cmd()
    if deffile is None:
        deffile = sys.stdout
    else:
        deffile = open(deffile, 'w')
    nm_cmd = DEFAULT_NM + [str(libfile)]
    nm_output = getnm(nm_cmd, shell=False)
    (dlist, flist) = parse_nm(nm_output)
    output_def(dlist, flist, DEF_HEADER, deffile)

