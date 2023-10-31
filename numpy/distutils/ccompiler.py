import os
import re
import sys
import shlex
import time
import subprocess
from copy import copy
from distutils import ccompiler
from distutils.ccompiler import compiler_class, gen_lib_options, get_default_compiler, new_compiler, CCompiler
from distutils.errors import DistutilsExecError, DistutilsModuleError, DistutilsPlatformError, CompileError, UnknownFileError
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion
from numpy.distutils import log
from numpy.distutils.exec_command import filepath_from_subprocess_output, forward_bytes_to_stdout
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, get_num_build_jobs, _commandline_dep_string, sanitize_cxx_flags
import threading
_job_semaphore = None
_global_lock = threading.Lock()
_processing_files = set()

def _needs_build(obj, cc_args, extra_postargs, pp_opts):
    """
    Check if an objects needs to be rebuild based on its dependencies

    Parameters
    ----------
    obj : str
        object file

    Returns
    -------
    bool
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler._needs_build', '_needs_build(obj, cc_args, extra_postargs, pp_opts)', {'os': os, '_commandline_dep_string': _commandline_dep_string, 'shlex': shlex, 'obj': obj, 'cc_args': cc_args, 'extra_postargs': extra_postargs, 'pp_opts': pp_opts}, 1)

def replace_method(klass, method_name, func):
    m = lambda self, *args, **kw: func(self, *args, **kw)
    setattr(klass, method_name, m)

def CCompiler_find_executables(self):
    """
    Does nothing here, but is called by the get_version method and can be
    overridden by subclasses. In particular it is redefined in the `FCompiler`
    class where more documentation can be found.

    """
    pass
replace_method(CCompiler, 'find_executables', CCompiler_find_executables)

def CCompiler_spawn(self, cmd, display=None, env=None):
    """
    Execute a command in a sub-process.

    Parameters
    ----------
    cmd : str
        The command to execute.
    display : str or sequence of str, optional
        The text to add to the log file kept by `numpy.distutils`.
        If not given, `display` is equal to `cmd`.
    env : a dictionary for environment variables, optional

    Returns
    -------
    None

    Raises
    ------
    DistutilsExecError
        If the command failed, i.e. the exit status was not 0.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler.CCompiler_spawn', 'CCompiler_spawn(self, cmd, display=None, env=None)', {'os': os, 'is_sequence': is_sequence, 'log': log, 'subprocess': subprocess, 'sys': sys, 'forward_bytes_to_stdout': forward_bytes_to_stdout, 're': re, 'DistutilsExecError': DistutilsExecError, 'self': self, 'cmd': cmd, 'display': display, 'env': env}, 1)
replace_method(CCompiler, 'spawn', CCompiler_spawn)

def CCompiler_object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
    """
    Return the name of the object files for the given source files.

    Parameters
    ----------
    source_filenames : list of str
        The list of paths to source files. Paths can be either relative or
        absolute, this is handled transparently.
    strip_dir : bool, optional
        Whether to strip the directory from the returned paths. If True,
        the file name prepended by `output_dir` is returned. Default is False.
    output_dir : str, optional
        If given, this path is prepended to the returned paths to the
        object files.

    Returns
    -------
    obj_names : list of str
        The list of paths to the object files corresponding to the source
        files in `source_filenames`.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler.CCompiler_object_filenames', "CCompiler_object_filenames(self, source_filenames, strip_dir=0, output_dir='')", {'os': os, 'UnknownFileError': UnknownFileError, 'self': self, 'source_filenames': source_filenames, 'strip_dir': strip_dir, 'output_dir': output_dir}, 1)
replace_method(CCompiler, 'object_filenames', CCompiler_object_filenames)

def CCompiler_compile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None):
    """
    Compile one or more source files.

    Please refer to the Python distutils API reference for more details.

    Parameters
    ----------
    sources : list of str
        A list of filenames
    output_dir : str, optional
        Path to the output directory.
    macros : list of tuples
        A list of macro definitions.
    include_dirs : list of str, optional
        The directories to add to the default include file search path for
        this compilation only.
    debug : bool, optional
        Whether or not to output debug symbols in or alongside the object
        file(s).
    extra_preargs, extra_postargs : ?
        Extra pre- and post-arguments.
    depends : list of str, optional
        A list of file names that all targets depend on.

    Returns
    -------
    objects : list of str
        A list of object file names, one per source file `sources`.

    Raises
    ------
    CompileError
        If compilation fails.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler.CCompiler_compile', 'CCompiler_compile(self, sources, output_dir=None, macros=None, include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None, depends=None)', {'get_num_build_jobs': get_num_build_jobs, '_global_lock': _global_lock, 'threading': threading, 'log': log, '_needs_build': _needs_build, '_processing_files': _processing_files, 'time': time, 'cyg2win32': cyg2win32, 'self': self, 'sources': sources, 'output_dir': output_dir, 'macros': macros, 'include_dirs': include_dirs, 'debug': debug, 'extra_preargs': extra_preargs, 'extra_postargs': extra_postargs, 'depends': depends}, 1)
replace_method(CCompiler, 'compile', CCompiler_compile)

def CCompiler_customize_cmd(self, cmd, ignore=()):
    """
    Customize compiler using distutils command.

    Parameters
    ----------
    cmd : class instance
        An instance inheriting from `distutils.cmd.Command`.
    ignore : sequence of str, optional
        List of `CCompiler` commands (without ``'set_'``) that should not be
        altered. Strings that are checked for are:
        ``('include_dirs', 'define', 'undef', 'libraries', 'library_dirs',
        'rpath', 'link_objects')``.

    Returns
    -------
    None

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler.CCompiler_customize_cmd', 'CCompiler_customize_cmd(self, cmd, ignore=())', {'log': log, 'self': self, 'cmd': cmd, 'ignore': ignore}, 1)
replace_method(CCompiler, 'customize_cmd', CCompiler_customize_cmd)

def _compiler_to_string(compiler):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler._compiler_to_string', '_compiler_to_string(compiler)', {'compiler': compiler}, 1)

def CCompiler_show_customization(self):
    """
    Print the compiler customizations to stdout.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    Printing is only done if the distutils log threshold is < 2.

    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.ccompiler.CCompiler_show_customization', 'CCompiler_show_customization(self)', {'log': log, '_compiler_to_string': _compiler_to_string, 'self': self}, 0)
replace_method(CCompiler, 'show_customization', CCompiler_show_customization)

def CCompiler_customize(self, dist, need_cxx=0):
    """
    Do any platform-specific customization of a compiler instance.

    This method calls `distutils.sysconfig.customize_compiler` for
    platform-specific customization, as well as optionally remove a flag
    to suppress spurious warnings in case C++ code is being compiled.

    Parameters
    ----------
    dist : object
        This parameter is not used for anything.
    need_cxx : bool, optional
        Whether or not C++ has to be compiled. If so (True), the
        ``"-Wstrict-prototypes"`` option is removed to prevent spurious
        warnings. Default is False.

    Returns
    -------
    None

    Notes
    -----
    All the default options used by distutils can be extracted with::

      from distutils import sysconfig
      sysconfig.get_config_vars('CC', 'CXX', 'OPT', 'BASECFLAGS',
                                'CCSHARED', 'LDSHARED', 'SO')

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler.CCompiler_customize', 'CCompiler_customize(self, dist, need_cxx=0)', {'log': log, 'customize_compiler': customize_compiler, 'os': os, 'CompileError': CompileError, 'self': self, 'dist': dist, 'need_cxx': need_cxx}, 1)
replace_method(CCompiler, 'customize', CCompiler_customize)

def simple_version_match(pat='[-.\\d]+', ignore='', start=''):
    """
    Simple matching of version numbers, for use in CCompiler and FCompiler.

    Parameters
    ----------
    pat : str, optional
        A regular expression matching version numbers.
        Default is ``r'[-.\d]+'``.
    ignore : str, optional
        A regular expression matching patterns to skip.
        Default is ``''``, in which case nothing is skipped.
    start : str, optional
        A regular expression matching the start of where to start looking
        for version numbers.
        Default is ``''``, in which case searching is started at the
        beginning of the version string given to `matcher`.

    Returns
    -------
    matcher : callable
        A function that is appropriate to use as the ``.version_match``
        attribute of a `CCompiler` class. `matcher` takes a single parameter,
        a version string.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler.simple_version_match', "simple_version_match(pat='[-.\\d]+', ignore='', start='')", {'re': re, 'pat': pat, 'ignore': ignore, 'start': start}, 1)

def CCompiler_get_version(self, force=False, ok_status=[0]):
    """
    Return compiler version, or None if compiler is not available.

    Parameters
    ----------
    force : bool, optional
        If True, force a new determination of the version, even if the
        compiler already has a version attribute. Default is False.
    ok_status : list of int, optional
        The list of status values returned by the version look-up process
        for which a version string is returned. If the status value is not
        in `ok_status`, None is returned. Default is ``[0]``.

    Returns
    -------
    version : str or None
        Version string, in the format of `distutils.version.LooseVersion`.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler.CCompiler_get_version', 'CCompiler_get_version(self, force=False, ok_status=[0])', {'re': re, 'subprocess': subprocess, 'filepath_from_subprocess_output': filepath_from_subprocess_output, 'LooseVersion': LooseVersion, 'self': self, 'force': force, 'ok_status': ok_status}, 1)
replace_method(CCompiler, 'get_version', CCompiler_get_version)

def CCompiler_cxx_compiler(self):
    """
    Return the C++ compiler.

    Parameters
    ----------
    None

    Returns
    -------
    cxx : class instance
        The C++ compiler, as a `CCompiler` instance.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler.CCompiler_cxx_compiler', 'CCompiler_cxx_compiler(self)', {'copy': copy, 'sanitize_cxx_flags': sanitize_cxx_flags, 'sys': sys, 'self': self}, 1)
replace_method(CCompiler, 'cxx_compiler', CCompiler_cxx_compiler)
compiler_class['intel'] = ('intelccompiler', 'IntelCCompiler', 'Intel C Compiler for 32-bit applications')
compiler_class['intele'] = ('intelccompiler', 'IntelItaniumCCompiler', 'Intel C Itanium Compiler for Itanium-based applications')
compiler_class['intelem'] = ('intelccompiler', 'IntelEM64TCCompiler', 'Intel C Compiler for 64-bit applications')
compiler_class['intelw'] = ('intelccompiler', 'IntelCCompilerW', 'Intel C Compiler for 32-bit applications on Windows')
compiler_class['intelemw'] = ('intelccompiler', 'IntelEM64TCCompilerW', 'Intel C Compiler for 64-bit applications on Windows')
compiler_class['pathcc'] = ('pathccompiler', 'PathScaleCCompiler', 'PathScale Compiler for SiCortex-based applications')
compiler_class['arm'] = ('armccompiler', 'ArmCCompiler', 'Arm C Compiler')
ccompiler._default_compilers += (('linux.*', 'intel'), ('linux.*', 'intele'), ('linux.*', 'intelem'), ('linux.*', 'pathcc'), ('nt', 'intelw'), ('nt', 'intelemw'))
if sys.platform == 'win32':
    compiler_class['mingw32'] = ('mingw32ccompiler', 'Mingw32CCompiler', 'Mingw32 port of GNU C Compiler for Win32(for MSC built Python)')
    if mingw32():
        log.info('Setting mingw32 as default compiler for nt.')
        ccompiler._default_compilers = (('nt', 'mingw32'), ) + ccompiler._default_compilers
_distutils_new_compiler = new_compiler

def new_compiler(plat=None, compiler=None, verbose=None, dry_run=0, force=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler.new_compiler', 'new_compiler(plat=None, compiler=None, verbose=None, dry_run=0, force=0)', {'log': log, 'os': os, 'get_default_compiler': get_default_compiler, 'compiler_class': compiler_class, 'DistutilsPlatformError': DistutilsPlatformError, 'DistutilsModuleError': DistutilsModuleError, 'sys': sys, 'plat': plat, 'compiler': compiler, 'verbose': verbose, 'dry_run': dry_run, 'force': force}, 1)
ccompiler.new_compiler = new_compiler
_distutils_gen_lib_options = gen_lib_options

def gen_lib_options(compiler, library_dirs, runtime_library_dirs, libraries):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.ccompiler.gen_lib_options', 'gen_lib_options(compiler, library_dirs, runtime_library_dirs, libraries)', {'_distutils_gen_lib_options': _distutils_gen_lib_options, 'is_sequence': is_sequence, 'compiler': compiler, 'library_dirs': library_dirs, 'runtime_library_dirs': runtime_library_dirs, 'libraries': libraries}, 1)
ccompiler.gen_lib_options = gen_lib_options
for _cc in ['msvc9', 'msvc', '_msvc', 'bcpp', 'cygwinc', 'emxc', 'unixc']:
    _m = sys.modules.get('distutils.' + _cc + 'compiler')
    if _m is not None:
        setattr(_m, 'gen_lib_options', gen_lib_options)

