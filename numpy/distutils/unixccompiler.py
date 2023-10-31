"""
unixccompiler - can handle very long argument lists for ar.

"""

import os
import sys
import subprocess
import shlex
from distutils.errors import CompileError, DistutilsExecError, LibError
from distutils.unixccompiler import UnixCCompiler
from numpy.distutils.ccompiler import replace_method
from numpy.distutils.misc_util import _commandline_dep_string
from numpy.distutils import log

def UnixCCompiler__compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
    """Compile a single source files with a Unix-style compiler."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.unixccompiler.UnixCCompiler__compile', 'UnixCCompiler__compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts)', {'os': os, 'shlex': shlex, 'DistutilsExecError': DistutilsExecError, 'CompileError': CompileError, 'sys': sys, 'subprocess': subprocess, '_commandline_dep_string': _commandline_dep_string, 'self': self, 'obj': obj, 'src': src, 'ext': ext, 'cc_args': cc_args, 'extra_postargs': extra_postargs, 'pp_opts': pp_opts}, 0)
replace_method(UnixCCompiler, '_compile', UnixCCompiler__compile)

def UnixCCompiler_create_static_lib(self, objects, output_libname, output_dir=None, debug=0, target_lang=None):
    """
    Build a static library in a separate sub-process.

    Parameters
    ----------
    objects : list or tuple of str
        List of paths to object files used to build the static library.
    output_libname : str
        The library name as an absolute or relative (if `output_dir` is used)
        path.
    output_dir : str, optional
        The path to the output directory. Default is None, in which case
        the ``output_dir`` attribute of the UnixCCompiler instance.
    debug : bool, optional
        This parameter is not used.
    target_lang : str, optional
        This parameter is not used.

    Returns
    -------
    None

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.unixccompiler.UnixCCompiler_create_static_lib', 'UnixCCompiler_create_static_lib(self, objects, output_libname, output_dir=None, debug=0, target_lang=None)', {'os': os, 'DistutilsExecError': DistutilsExecError, 'LibError': LibError, 'log': log, 'self': self, 'objects': objects, 'output_libname': output_libname, 'output_dir': output_dir, 'debug': debug, 'target_lang': target_lang}, 1)
replace_method(UnixCCompiler, 'create_static_lib', UnixCCompiler_create_static_lib)

