import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
_tdata = tlocal()
_tmpdirs = []

def clean_up_temporary_directory():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.misc_util.clean_up_temporary_directory', 'clean_up_temporary_directory()', {'_tmpdirs': _tmpdirs, 'shutil': shutil}, 0)
atexit.register(clean_up_temporary_directory)
__all__ = ['Configuration', 'get_numpy_include_dirs', 'default_config_dict', 'dict_append', 'appendpath', 'generate_config_py', 'get_cmd', 'allpath', 'get_mathlibs', 'terminal_has_colors', 'red_text', 'green_text', 'yellow_text', 'blue_text', 'cyan_text', 'cyg2win32', 'mingw32', 'all_strings', 'has_f_sources', 'has_cxx_sources', 'filter_sources', 'get_dependencies', 'is_local_src_dir', 'get_ext_source_files', 'get_script_files', 'get_lib_source_files', 'get_data_files', 'dot_join', 'get_frame', 'minrelpath', 'njoin', 'is_sequence', 'is_string', 'as_list', 'gpaths', 'get_language', 'get_build_architecture', 'get_info', 'get_pkg_info', 'get_num_build_jobs', 'sanitize_cxx_flags', 'exec_mod_from_location']


class InstallableLib:
    """
    Container to hold information on an installable library.

    Parameters
    ----------
    name : str
        Name of the installed library.
    build_info : dict
        Dictionary holding build information.
    target_dir : str
        Absolute path specifying where to install the library.

    See Also
    --------
    Configuration.add_installed_library

    Notes
    -----
    The three parameters are stored as attributes with the same names.

    """
    
    def __init__(self, name, build_info, target_dir):
        self.name = name
        self.build_info = build_info
        self.target_dir = target_dir


def get_num_build_jobs():
    """
    Get number of parallel build jobs set by the --parallel command line
    argument of setup.py
    If the command did not receive a setting the environment variable
    NPY_NUM_BUILD_JOBS is checked. If that is unset, return the number of
    processors on the system, with a maximum of 8 (to prevent
    overloading the system if there a lot of CPUs).

    Returns
    -------
    out : int
        number of parallel jobs that can be run

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_num_build_jobs', 'get_num_build_jobs()', {'os': os, 'multiprocessing': multiprocessing}, 1)

def quote_args(args):
    """Quote list of arguments.

    .. deprecated:: 1.22.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.quote_args', 'quote_args(args)', {'args': args}, 1)

def allpath(name):
    """Convert a /-separated pathname to one using the OS's path separator."""
    split = name.split('/')
    return os.path.join(*split)

def rel_path(path, parent_path):
    """Return path relative to parent_path."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.rel_path', 'rel_path(path, parent_path)', {'os': os, 'path': path, 'parent_path': parent_path}, 1)

def get_path_from_frame(frame, parent_path=None):
    """Return path of the module given a frame object from the call stack.

    Returned path is relative to parent_path when given,
    otherwise it is absolute path.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_path_from_frame', 'get_path_from_frame(frame, parent_path=None)', {'os': os, 'sys': sys, 'rel_path': rel_path, 'frame': frame, 'parent_path': parent_path}, 1)

def njoin(*path):
    """Join two or more pathname components +
    - convert a /-separated pathname to one using the OS's path separator.
    - resolve `..` and `.` from path.

    Either passing n arguments as in njoin('a','b'), or a sequence
    of n names as in njoin(['a','b']) is handled, or a mixture of such arguments.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.njoin', 'njoin(*path)', {'is_sequence': is_sequence, 'njoin': njoin, 'is_string': is_string, 'os': os, 'minrelpath': minrelpath, 'path': path}, 1)

def get_mathlibs(path=None):
    """Return the MATHLIB line from numpyconfig.h
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_mathlibs', 'get_mathlibs(path=None)', {'os': os, 'get_numpy_include_dirs': get_numpy_include_dirs, 'DistutilsError': DistutilsError, 'path': path}, 1)

def minrelpath(path):
    """Resolve `..` and '.' from path.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.minrelpath', 'minrelpath(path)', {'is_string': is_string, 'os': os, 'path': path}, 1)

def sorted_glob(fileglob):
    """sorts output of python glob for https://bugs.python.org/issue30461
    to allow extensions to have reproducible build results"""
    return sorted(glob.glob(fileglob))

def _fix_paths(paths, local_path, include_non_existing):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util._fix_paths', '_fix_paths(paths, local_path, include_non_existing)', {'is_sequence': is_sequence, 'is_string': is_string, 'sorted_glob': sorted_glob, 'njoin': njoin, 'os': os, '_fix_paths': _fix_paths, 'minrelpath': minrelpath, 'paths': paths, 'local_path': local_path, 'include_non_existing': include_non_existing}, 1)

def gpaths(paths, local_path='', include_non_existing=True):
    """Apply glob to paths and prepend local_path if needed.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.gpaths', "gpaths(paths, local_path='', include_non_existing=True)", {'is_string': is_string, '_fix_paths': _fix_paths, 'paths': paths, 'local_path': local_path, 'include_non_existing': include_non_existing}, 1)

def make_temp_file(suffix='', prefix='', text=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.make_temp_file', "make_temp_file(suffix='', prefix='', text=True)", {'_tdata': _tdata, 'tempfile': tempfile, '_tmpdirs': _tmpdirs, 'os': os, 'suffix': suffix, 'prefix': prefix, 'text': text}, 2)

def terminal_has_colors():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.terminal_has_colors', 'terminal_has_colors()', {'sys': sys, 'os': os}, 1)
if terminal_has_colors():
    _colour_codes = dict(black=0, red=1, green=2, yellow=3, blue=4, magenta=5, cyan=6, white=7, default=9)
    
    def colour_text(s, fg=None, bg=None, bold=False):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.colour_text', 'colour_text(s, fg=None, bg=None, bold=False)', {'_colour_codes': _colour_codes, 's': s, 'fg': fg, 'bg': bg, 'bold': bold}, 1)
else:
    
    def colour_text(s, fg=None, bg=None):
        return s

def default_text(s):
    return colour_text(s, 'default')

def red_text(s):
    return colour_text(s, 'red')

def green_text(s):
    return colour_text(s, 'green')

def yellow_text(s):
    return colour_text(s, 'yellow')

def cyan_text(s):
    return colour_text(s, 'cyan')

def blue_text(s):
    return colour_text(s, 'blue')

def cyg2win32(path: str) -> str:
    """Convert a path from Cygwin-native to Windows-native.

    Uses the cygpath utility (part of the Base install) to do the
    actual conversion.  Falls back to returning the original path if
    this fails.

    Handles the default ``/cygdrive`` mount prefix as well as the
    ``/proc/cygdrive`` portable prefix, custom cygdrive prefixes such
    as ``/`` or ``/mnt``, and absolute paths such as ``/usr/src/`` or
    ``/home/username``

    Parameters
    ----------
    path : str
       The path to convert

    Returns
    -------
    converted_path : str
        The converted path

    Notes
    -----
    Documentation for cygpath utility:
    https://cygwin.com/cygwin-ug-net/cygpath.html
    Documentation for the C function it wraps:
    https://cygwin.com/cygwin-api/func-cygwin-conv-path.html

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.cyg2win32', 'cyg2win32(path)', {'sys': sys, 'subprocess': subprocess, 'path': path}, 1)

def mingw32():
    """Return true when using mingw32 environment.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.mingw32', 'mingw32()', {'sys': sys, 'os': os}, 1)

def msvc_runtime_version():
    """Return version of MSVC runtime library, as defined by __MSC_VER__ macro"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.msvc_runtime_version', 'msvc_runtime_version()', {'sys': sys}, 1)

def msvc_runtime_library():
    """Return name of MSVC runtime library if Python was built with MSVC >= 7"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.msvc_runtime_library', 'msvc_runtime_library()', {'msvc_runtime_major': msvc_runtime_major}, 1)

def msvc_runtime_major():
    """Return major version of MSVC runtime coded like get_build_msvc_version"""
    major = {1300: 70, 1310: 71, 1400: 80, 1500: 90, 1600: 100, 1900: 140}.get(msvc_runtime_version(), None)
    return major
cxx_ext_match = re.compile('.*\\.(cpp|cxx|cc)\\Z', re.I).match
fortran_ext_match = re.compile('.*\\.(f90|f95|f77|for|ftn|f)\\Z', re.I).match
f90_ext_match = re.compile('.*\\.(f90|f95)\\Z', re.I).match
f90_module_name_match = re.compile('\\s*module\\s*(?P<name>[\\w_]+)', re.I).match

def _get_f90_modules(source):
    """Return a list of Fortran f90 module names that
    given source file defines.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util._get_f90_modules', '_get_f90_modules(source)', {'f90_ext_match': f90_ext_match, 'f90_module_name_match': f90_module_name_match, 'source': source}, 1)

def is_string(s):
    return isinstance(s, str)

def all_strings(lst):
    """Return True if all items in lst are string objects. """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.all_strings', 'all_strings(lst)', {'is_string': is_string, 'lst': lst}, 1)

def is_sequence(seq):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.is_sequence', 'is_sequence(seq)', {'is_string': is_string, 'seq': seq}, 1)

def is_glob_pattern(s):
    return (is_string(s) and (('*' in s or '?' in s)))

def as_list(seq):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.as_list', 'as_list(seq)', {'is_sequence': is_sequence, 'seq': seq}, 1)

def get_language(sources):
    """Determine language value (c,f77,f90) from sources """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_language', 'get_language(sources)', {'f90_ext_match': f90_ext_match, 'fortran_ext_match': fortran_ext_match, 'sources': sources}, 1)

def has_f_sources(sources):
    """Return True if sources contains Fortran files """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.has_f_sources', 'has_f_sources(sources)', {'fortran_ext_match': fortran_ext_match, 'sources': sources}, 1)

def has_cxx_sources(sources):
    """Return True if sources contains C++ files """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.has_cxx_sources', 'has_cxx_sources(sources)', {'cxx_ext_match': cxx_ext_match, 'sources': sources}, 1)

def filter_sources(sources):
    """Return four lists of filenames containing
    C, C++, Fortran, and Fortran 90 module sources,
    respectively.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.filter_sources', 'filter_sources(sources)', {'fortran_ext_match': fortran_ext_match, '_get_f90_modules': _get_f90_modules, 'cxx_ext_match': cxx_ext_match, 'sources': sources}, 4)

def _get_headers(directory_list):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util._get_headers', '_get_headers(directory_list)', {'sorted_glob': sorted_glob, 'os': os, 'directory_list': directory_list}, 1)

def _get_directories(list_of_sources):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util._get_directories', '_get_directories(list_of_sources)', {'os': os, 'list_of_sources': list_of_sources}, 1)

def _commandline_dep_string(cc_args, extra_postargs, pp_opts):
    """
    Return commandline representation used to determine if a file needs
    to be recompiled
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util._commandline_dep_string', '_commandline_dep_string(cc_args, extra_postargs, pp_opts)', {'cc_args': cc_args, 'extra_postargs': extra_postargs, 'pp_opts': pp_opts}, 1)

def get_dependencies(sources):
    return _get_headers(_get_directories(sources))

def is_local_src_dir(directory):
    """Return true if directory is local directory.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.is_local_src_dir', 'is_local_src_dir(directory)', {'is_string': is_string, 'os': os, 'directory': directory}, 1)

def general_source_files(top_path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.misc_util.general_source_files', 'general_source_files(top_path)', {'re': re, 'os': os, 'top_path': top_path}, 0)

def general_source_directories_files(top_path):
    """Return a directory name relative to top_path and
    files contained.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.misc_util.general_source_directories_files', 'general_source_directories_files(top_path)', {'re': re, 'os': os, 'rel_path': rel_path, 'top_path': top_path}, 0)

def get_ext_source_files(ext):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_ext_source_files', 'get_ext_source_files(ext)', {'is_string': is_string, 'get_dependencies': get_dependencies, 'is_local_src_dir': is_local_src_dir, 'general_source_files': general_source_files, 'os': os, 'ext': ext}, 1)

def get_script_files(scripts):
    scripts = [_m for _m in scripts if is_string(_m)]
    return scripts

def get_lib_source_files(lib):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_lib_source_files', 'get_lib_source_files(lib)', {'is_string': is_string, 'get_dependencies': get_dependencies, 'is_local_src_dir': is_local_src_dir, 'general_source_files': general_source_files, 'os': os, 'lib': lib}, 1)

def get_shared_lib_extension(is_python_ext=False):
    """Return the correct file extension for shared libraries.

    Parameters
    ----------
    is_python_ext : bool, optional
        Whether the shared library is a Python extension.  Default is False.

    Returns
    -------
    so_ext : str
        The shared library extension.

    Notes
    -----
    For Python shared libs, `so_ext` will typically be '.so' on Linux and OS X,
    and '.pyd' on Windows.  For Python >= 3.2 `so_ext` has a tag prepended on
    POSIX systems according to PEP 3149.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_shared_lib_extension', 'get_shared_lib_extension(is_python_ext=False)', {'distutils': distutils, 'sys': sys, 'is_python_ext': is_python_ext}, 1)

def get_data_files(data):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_data_files', 'get_data_files(data)', {'is_string': is_string, 'is_local_src_dir': is_local_src_dir, 'general_source_files': general_source_files, 'os': os, 'data': data}, 1)

def dot_join(*args):
    return '.'.join([a for a in args if a])

def get_frame(level=0):
    """Return frame object from call stack with given level.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_frame', 'get_frame(level=0)', {'sys': sys, 'level': level}, 1)


class Configuration:
    _list_keys = ['packages', 'ext_modules', 'data_files', 'include_dirs', 'libraries', 'headers', 'scripts', 'py_modules', 'installed_libraries', 'define_macros']
    _dict_keys = ['package_dir', 'installed_pkg_config']
    _extra_keys = ['name', 'version']
    numpy_include_dirs = []
    
    def __init__(self, package_name=None, parent_name=None, top_path=None, package_path=None, caller_level=1, setup_name='setup.py', **attrs):
        """Construct configuration instance of a package.

        package_name -- name of the package
                        Ex.: 'distutils'
        parent_name  -- name of the parent package
                        Ex.: 'numpy'
        top_path     -- directory of the toplevel package
                        Ex.: the directory where the numpy package source sits
        package_path -- directory of package. Will be computed by magic from the
                        directory of the caller module if not specified
                        Ex.: the directory where numpy.distutils is
        caller_level -- frame level to caller namespace, internal parameter.
        """
        self.name = dot_join(parent_name, package_name)
        self.version = None
        caller_frame = get_frame(caller_level)
        self.local_path = get_path_from_frame(caller_frame, top_path)
        if top_path is None:
            top_path = self.local_path
            self.local_path = ''
        if package_path is None:
            package_path = self.local_path
        elif os.path.isdir(njoin(self.local_path, package_path)):
            package_path = njoin(self.local_path, package_path)
        if not os.path.isdir((package_path or '.')):
            raise ValueError('%r is not a directory' % (package_path, ))
        self.top_path = top_path
        self.package_path = package_path
        self.path_in_package = os.path.join(*self.name.split('.'))
        self.list_keys = self._list_keys[:]
        self.dict_keys = self._dict_keys[:]
        for n in self.list_keys:
            v = copy.copy(attrs.get(n, []))
            setattr(self, n, as_list(v))
        for n in self.dict_keys:
            v = copy.copy(attrs.get(n, {}))
            setattr(self, n, v)
        known_keys = self.list_keys + self.dict_keys
        self.extra_keys = self._extra_keys[:]
        for n in attrs.keys():
            if n in known_keys:
                continue
            a = attrs[n]
            setattr(self, n, a)
            if isinstance(a, list):
                self.list_keys.append(n)
            elif isinstance(a, dict):
                self.dict_keys.append(n)
            else:
                self.extra_keys.append(n)
        if os.path.exists(njoin(package_path, '__init__.py')):
            self.packages.append(self.name)
            self.package_dir[self.name] = package_path
        self.options = dict(ignore_setup_xxx_py=False, assume_default_configuration=False, delegate_options_to_subpackages=False, quiet=False)
        caller_instance = None
        for i in range(1, 3):
            try:
                f = get_frame(i)
            except ValueError:
                break
            try:
                caller_instance = eval('self', f.f_globals, f.f_locals)
                break
            except NameError:
                pass
        if isinstance(caller_instance, self.__class__):
            if caller_instance.options['delegate_options_to_subpackages']:
                self.set_options(**caller_instance.options)
        self.setup_name = setup_name
    
    def todict(self):
        """
        Return a dictionary compatible with the keyword arguments of distutils
        setup function.

        Examples
        --------
        >>> setup(**config.todict())                           #doctest: +SKIP
        """
        self._optimize_data_files()
        d = {}
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        for n in known_keys:
            a = getattr(self, n)
            if a:
                d[n] = a
        return d
    
    def info(self, message):
        if not self.options['quiet']:
            print(message)
    
    def warn(self, message):
        sys.stderr.write('Warning: %s\n' % (message, ))
    
    def set_options(self, **options):
        """
        Configure Configuration instance.

        The following options are available:
         - ignore_setup_xxx_py
         - assume_default_configuration
         - delegate_options_to_subpackages
         - quiet

        """
        for (key, value) in options.items():
            if key in self.options:
                self.options[key] = value
            else:
                raise ValueError('Unknown option: ' + key)
    
    def get_distribution(self):
        """Return the distutils distribution object for self."""
        from numpy.distutils.core import get_distribution
        return get_distribution()
    
    def _wildcard_get_subpackage(self, subpackage_name, parent_name, caller_level=1):
        l = subpackage_name.split('.')
        subpackage_path = njoin([self.local_path] + l)
        dirs = [_m for _m in sorted_glob(subpackage_path) if os.path.isdir(_m)]
        config_list = []
        for d in dirs:
            if not os.path.isfile(njoin(d, '__init__.py')):
                continue
            if 'build' in d.split(os.sep):
                continue
            n = '.'.join(d.split(os.sep)[-len(l):])
            c = self.get_subpackage(n, parent_name=parent_name, caller_level=caller_level + 1)
            config_list.extend(c)
        return config_list
    
    def _get_configuration_from_setup_py(self, setup_py, subpackage_name, subpackage_path, parent_name, caller_level=1):
        sys.path.insert(0, os.path.dirname(setup_py))
        try:
            setup_name = os.path.splitext(os.path.basename(setup_py))[0]
            n = dot_join(self.name, subpackage_name, setup_name)
            setup_module = exec_mod_from_location('_'.join(n.split('.')), setup_py)
            if not hasattr(setup_module, 'configuration'):
                if not self.options['assume_default_configuration']:
                    self.warn('Assuming default configuration (%s does not define configuration())' % setup_module)
                config = Configuration(subpackage_name, parent_name, self.top_path, subpackage_path, caller_level=caller_level + 1)
            else:
                pn = dot_join(*[parent_name] + subpackage_name.split('.')[:-1])
                args = (pn, )
                if setup_module.configuration.__code__.co_argcount > 1:
                    args = args + (self.top_path, )
                config = setup_module.configuration(*args)
            if config.name != dot_join(parent_name, subpackage_name):
                self.warn('Subpackage %r configuration returned as %r' % (dot_join(parent_name, subpackage_name), config.name))
        finally:
            del sys.path[0]
        return config
    
    def get_subpackage(self, subpackage_name, subpackage_path=None, parent_name=None, caller_level=1):
        """Return list of subpackage configurations.

        Parameters
        ----------
        subpackage_name : str or None
            Name of the subpackage to get the configuration. '*' in
            subpackage_name is handled as a wildcard.
        subpackage_path : str
            If None, then the path is assumed to be the local path plus the
            subpackage_name. If a setup.py file is not found in the
            subpackage_path, then a default configuration is used.
        parent_name : str
            Parent name.
        """
        if subpackage_name is None:
            if subpackage_path is None:
                raise ValueError('either subpackage_name or subpackage_path must be specified')
            subpackage_name = os.path.basename(subpackage_path)
        l = subpackage_name.split('.')
        if (subpackage_path is None and '*' in subpackage_name):
            return self._wildcard_get_subpackage(subpackage_name, parent_name, caller_level=caller_level + 1)
        assert '*' not in subpackage_name, repr((subpackage_name, subpackage_path, parent_name))
        if subpackage_path is None:
            subpackage_path = njoin([self.local_path] + l)
        else:
            subpackage_path = njoin([subpackage_path] + l[:-1])
            subpackage_path = self.paths([subpackage_path])[0]
        setup_py = njoin(subpackage_path, self.setup_name)
        if not self.options['ignore_setup_xxx_py']:
            if not os.path.isfile(setup_py):
                setup_py = njoin(subpackage_path, 'setup_%s.py' % subpackage_name)
        if not os.path.isfile(setup_py):
            if not self.options['assume_default_configuration']:
                self.warn('Assuming default configuration (%s/{setup_%s,setup}.py was not found)' % (os.path.dirname(setup_py), subpackage_name))
            config = Configuration(subpackage_name, parent_name, self.top_path, subpackage_path, caller_level=caller_level + 1)
        else:
            config = self._get_configuration_from_setup_py(setup_py, subpackage_name, subpackage_path, parent_name, caller_level=caller_level + 1)
        if config:
            return [config]
        else:
            return []
    
    def add_subpackage(self, subpackage_name, subpackage_path=None, standalone=False):
        """Add a sub-package to the current Configuration instance.

        This is useful in a setup.py script for adding sub-packages to a
        package.

        Parameters
        ----------
        subpackage_name : str
            name of the subpackage
        subpackage_path : str
            if given, the subpackage path such as the subpackage is in
            subpackage_path / subpackage_name. If None,the subpackage is
            assumed to be located in the local path / subpackage_name.
        standalone : bool
        """
        if standalone:
            parent_name = None
        else:
            parent_name = self.name
        config_list = self.get_subpackage(subpackage_name, subpackage_path, parent_name=parent_name, caller_level=2)
        if not config_list:
            self.warn('No configuration returned, assuming unavailable.')
        for config in config_list:
            d = config
            if isinstance(config, Configuration):
                d = config.todict()
            assert isinstance(d, dict), repr(type(d))
            self.info('Appending %s configuration to %s' % (d.get('name'), self.name))
            self.dict_append(**d)
        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized, it may be too late to add a subpackage ' + subpackage_name)
    
    def add_data_dir(self, data_path):
        """Recursively add files under data_path to data_files list.

        Recursively add files under data_path to the list of data_files to be
        installed (and distributed). The data_path can be either a relative
        path-name, or an absolute path-name, or a 2-tuple where the first
        argument shows where in the install directory the data directory
        should be installed to.

        Parameters
        ----------
        data_path : seq or str
            Argument can be either

                * 2-sequence (<datadir suffix>, <path to data directory>)
                * path to data directory where python datadir suffix defaults
                  to package dir.

        Notes
        -----
        Rules for installation paths::

            foo/bar -> (foo/bar, foo/bar) -> parent/foo/bar
            (gun, foo/bar) -> parent/gun
            foo/* -> (foo/a, foo/a), (foo/b, foo/b) -> parent/foo/a, parent/foo/b
            (gun, foo/*) -> (gun, foo/a), (gun, foo/b) -> gun
            (gun/*, foo/*) -> parent/gun/a, parent/gun/b
            /foo/bar -> (bar, /foo/bar) -> parent/bar
            (gun, /foo/bar) -> parent/gun
            (fun/*/gun/*, sun/foo/bar) -> parent/fun/foo/gun/bar

        Examples
        --------
        For example suppose the source directory contains fun/foo.dat and
        fun/bar/car.dat:

        >>> self.add_data_dir('fun')                       #doctest: +SKIP
        >>> self.add_data_dir(('sun', 'fun'))              #doctest: +SKIP
        >>> self.add_data_dir(('gun', '/full/path/to/fun'))#doctest: +SKIP

        Will install data-files to the locations::

            <package install directory>/
              fun/
                foo.dat
                bar/
                  car.dat
              sun/
                foo.dat
                bar/
                  car.dat
              gun/
                foo.dat
                car.dat

        """
        if is_sequence(data_path):
            (d, data_path) = data_path
        else:
            d = None
        if is_sequence(data_path):
            [self.add_data_dir((d, p)) for p in data_path]
            return
        if not is_string(data_path):
            raise TypeError('not a string: %r' % (data_path, ))
        if d is None:
            if os.path.isabs(data_path):
                return self.add_data_dir((os.path.basename(data_path), data_path))
            return self.add_data_dir((data_path, data_path))
        paths = self.paths(data_path, include_non_existing=False)
        if is_glob_pattern(data_path):
            if is_glob_pattern(d):
                pattern_list = allpath(d).split(os.sep)
                pattern_list.reverse()
                rl = list(range(len(pattern_list) - 1))
                rl.reverse()
                for i in rl:
                    if not pattern_list[i]:
                        del pattern_list[i]
                for path in paths:
                    if not os.path.isdir(path):
                        print('Not a directory, skipping', path)
                        continue
                    rpath = rel_path(path, self.local_path)
                    path_list = rpath.split(os.sep)
                    path_list.reverse()
                    target_list = []
                    i = 0
                    for s in pattern_list:
                        if is_glob_pattern(s):
                            if i >= len(path_list):
                                raise ValueError('cannot fill pattern %r with %r' % (d, path))
                            target_list.append(path_list[i])
                        else:
                            assert s == path_list[i], repr((s, path_list[i], data_path, d, path, rpath))
                            target_list.append(s)
                        i += 1
                    if path_list[i:]:
                        self.warn('mismatch of pattern_list=%s and path_list=%s' % (pattern_list, path_list))
                    target_list.reverse()
                    self.add_data_dir((os.sep.join(target_list), path))
            else:
                for path in paths:
                    self.add_data_dir((d, path))
            return
        assert not is_glob_pattern(d), repr(d)
        dist = self.get_distribution()
        if (dist is not None and dist.data_files is not None):
            data_files = dist.data_files
        else:
            data_files = self.data_files
        for path in paths:
            for (d1, f) in list(general_source_directories_files(path)):
                target_path = os.path.join(self.path_in_package, d, d1)
                data_files.append((target_path, f))
    
    def _optimize_data_files(self):
        data_dict = {}
        for (p, files) in self.data_files:
            if p not in data_dict:
                data_dict[p] = set()
            for f in files:
                data_dict[p].add(f)
        self.data_files[:] = [(p, list(files)) for (p, files) in data_dict.items()]
    
    def add_data_files(self, *files):
        """Add data files to configuration data_files.

        Parameters
        ----------
        files : sequence
            Argument(s) can be either

                * 2-sequence (<datadir prefix>,<path to data file(s)>)
                * paths to data files where python datadir prefix defaults
                  to package dir.

        Notes
        -----
        The form of each element of the files sequence is very flexible
        allowing many combinations of where to get the files from the package
        and where they should ultimately be installed on the system. The most
        basic usage is for an element of the files argument sequence to be a
        simple filename. This will cause that file from the local path to be
        installed to the installation path of the self.name package (package
        path). The file argument can also be a relative path in which case the
        entire relative path will be installed into the package directory.
        Finally, the file can be an absolute path name in which case the file
        will be found at the absolute path name but installed to the package
        path.

        This basic behavior can be augmented by passing a 2-tuple in as the
        file argument. The first element of the tuple should specify the
        relative path (under the package install directory) where the
        remaining sequence of files should be installed to (it has nothing to
        do with the file-names in the source distribution). The second element
        of the tuple is the sequence of files that should be installed. The
        files in this sequence can be filenames, relative paths, or absolute
        paths. For absolute paths the file will be installed in the top-level
        package installation directory (regardless of the first argument).
        Filenames and relative path names will be installed in the package
        install directory under the path name given as the first element of
        the tuple.

        Rules for installation paths:

          #. file.txt -> (., file.txt)-> parent/file.txt
          #. foo/file.txt -> (foo, foo/file.txt) -> parent/foo/file.txt
          #. /foo/bar/file.txt -> (., /foo/bar/file.txt) -> parent/file.txt
          #. ``*``.txt -> parent/a.txt, parent/b.txt
          #. foo/``*``.txt`` -> parent/foo/a.txt, parent/foo/b.txt
          #. ``*/*.txt`` -> (``*``, ``*``/``*``.txt) -> parent/c/a.txt, parent/d/b.txt
          #. (sun, file.txt) -> parent/sun/file.txt
          #. (sun, bar/file.txt) -> parent/sun/file.txt
          #. (sun, /foo/bar/file.txt) -> parent/sun/file.txt
          #. (sun, ``*``.txt) -> parent/sun/a.txt, parent/sun/b.txt
          #. (sun, bar/``*``.txt) -> parent/sun/a.txt, parent/sun/b.txt
          #. (sun/``*``, ``*``/``*``.txt) -> parent/sun/c/a.txt, parent/d/b.txt

        An additional feature is that the path to a data-file can actually be
        a function that takes no arguments and returns the actual path(s) to
        the data-files. This is useful when the data files are generated while
        building the package.

        Examples
        --------
        Add files to the list of data_files to be included with the package.

            >>> self.add_data_files('foo.dat',
            ...     ('fun', ['gun.dat', 'nun/pun.dat', '/tmp/sun.dat']),
            ...     'bar/cat.dat',
            ...     '/full/path/to/can.dat')                   #doctest: +SKIP

        will install these data files to::

            <package install directory>/
             foo.dat
             fun/
               gun.dat
               nun/
                 pun.dat
             sun.dat
             bar/
               car.dat
             can.dat

        where <package install directory> is the package (or sub-package)
        directory such as '/usr/lib/python2.4/site-packages/mypackage' ('C:
        \Python2.4 \Lib \site-packages \mypackage') or
        '/usr/lib/python2.4/site- packages/mypackage/mysubpackage' ('C:
        \Python2.4 \Lib \site-packages \mypackage \mysubpackage').
        """
        if len(files) > 1:
            for f in files:
                self.add_data_files(f)
            return
        assert len(files) == 1
        if is_sequence(files[0]):
            (d, files) = files[0]
        else:
            d = None
        if is_string(files):
            filepat = files
        elif is_sequence(files):
            if len(files) == 1:
                filepat = files[0]
            else:
                for f in files:
                    self.add_data_files((d, f))
                return
        else:
            raise TypeError(repr(type(files)))
        if d is None:
            if hasattr(filepat, '__call__'):
                d = ''
            elif os.path.isabs(filepat):
                d = ''
            else:
                d = os.path.dirname(filepat)
            self.add_data_files((d, files))
            return
        paths = self.paths(filepat, include_non_existing=False)
        if is_glob_pattern(filepat):
            if is_glob_pattern(d):
                pattern_list = d.split(os.sep)
                pattern_list.reverse()
                for path in paths:
                    path_list = path.split(os.sep)
                    path_list.reverse()
                    path_list.pop()
                    target_list = []
                    i = 0
                    for s in pattern_list:
                        if is_glob_pattern(s):
                            target_list.append(path_list[i])
                            i += 1
                        else:
                            target_list.append(s)
                    target_list.reverse()
                    self.add_data_files((os.sep.join(target_list), path))
            else:
                self.add_data_files((d, paths))
            return
        assert not is_glob_pattern(d), repr((d, filepat))
        dist = self.get_distribution()
        if (dist is not None and dist.data_files is not None):
            data_files = dist.data_files
        else:
            data_files = self.data_files
        data_files.append((os.path.join(self.path_in_package, d), paths))
    
    def add_define_macros(self, macros):
        """Add define macros to configuration

        Add the given sequence of macro name and value duples to the beginning
        of the define_macros list This list will be visible to all extension
        modules of the current package.
        """
        dist = self.get_distribution()
        if dist is not None:
            if not hasattr(dist, 'define_macros'):
                dist.define_macros = []
            dist.define_macros.extend(macros)
        else:
            self.define_macros.extend(macros)
    
    def add_include_dirs(self, *paths):
        """Add paths to configuration include directories.

        Add the given sequence of paths to the beginning of the include_dirs
        list. This list will be visible to all extension modules of the
        current package.
        """
        include_dirs = self.paths(paths)
        dist = self.get_distribution()
        if dist is not None:
            if dist.include_dirs is None:
                dist.include_dirs = []
            dist.include_dirs.extend(include_dirs)
        else:
            self.include_dirs.extend(include_dirs)
    
    def add_headers(self, *files):
        """Add installable headers to configuration.

        Add the given sequence of files to the beginning of the headers list.
        By default, headers will be installed under <python-
        include>/<self.name.replace('.','/')>/ directory. If an item of files
        is a tuple, then its first argument specifies the actual installation
        location relative to the <python-include> path.

        Parameters
        ----------
        files : str or seq
            Argument(s) can be either:

                * 2-sequence (<includedir suffix>,<path to header file(s)>)
                * path(s) to header file(s) where python includedir suffix will
                  default to package name.
        """
        headers = []
        for path in files:
            if is_string(path):
                [headers.append((self.name, p)) for p in self.paths(path)]
            else:
                if (not isinstance(path, (tuple, list)) or len(path) != 2):
                    raise TypeError(repr(path))
                [headers.append((path[0], p)) for p in self.paths(path[1])]
        dist = self.get_distribution()
        if dist is not None:
            if dist.headers is None:
                dist.headers = []
            dist.headers.extend(headers)
        else:
            self.headers.extend(headers)
    
    def paths(self, *paths, **kws):
        """Apply glob to paths and prepend local_path if needed.

        Applies glob.glob(...) to each path in the sequence (if needed) and
        pre-pends the local_path if needed. Because this is called on all
        source lists, this allows wildcard characters to be specified in lists
        of sources for extension modules and libraries and scripts and allows
        path-names be relative to the source directory.

        """
        include_non_existing = kws.get('include_non_existing', True)
        return gpaths(paths, local_path=self.local_path, include_non_existing=include_non_existing)
    
    def _fix_paths_dict(self, kw):
        for k in kw.keys():
            v = kw[k]
            if k in ['sources', 'depends', 'include_dirs', 'library_dirs', 'module_dirs', 'extra_objects']:
                new_v = self.paths(v)
                kw[k] = new_v
    
    def add_extension(self, name, sources, **kw):
        """Add extension to configuration.

        Create and add an Extension instance to the ext_modules list. This
        method also takes the following optional keyword arguments that are
        passed on to the Extension constructor.

        Parameters
        ----------
        name : str
            name of the extension
        sources : seq
            list of the sources. The list of sources may contain functions
            (called source generators) which must take an extension instance
            and a build directory as inputs and return a source file or list of
            source files or None. If None is returned then no sources are
            generated. If the Extension instance has no sources after
            processing all source generators, then no extension module is
            built.
        include_dirs :
        define_macros :
        undef_macros :
        library_dirs :
        libraries :
        runtime_library_dirs :
        extra_objects :
        extra_compile_args :
        extra_link_args :
        extra_f77_compile_args :
        extra_f90_compile_args :
        export_symbols :
        swig_opts :
        depends :
            The depends list contains paths to files or directories that the
            sources of the extension module depend on. If any path in the
            depends list is newer than the extension module, then the module
            will be rebuilt.
        language :
        f2py_options :
        module_dirs :
        extra_info : dict or list
            dict or list of dict of keywords to be appended to keywords.

        Notes
        -----
        The self.paths(...) method is applied to all lists that may contain
        paths.
        """
        ext_args = copy.copy(kw)
        ext_args['name'] = dot_join(self.name, name)
        ext_args['sources'] = sources
        if 'extra_info' in ext_args:
            extra_info = ext_args['extra_info']
            del ext_args['extra_info']
            if isinstance(extra_info, dict):
                extra_info = [extra_info]
            for info in extra_info:
                assert isinstance(info, dict), repr(info)
                dict_append(ext_args, **info)
        self._fix_paths_dict(ext_args)
        libraries = ext_args.get('libraries', [])
        libnames = []
        ext_args['libraries'] = []
        for libname in libraries:
            if isinstance(libname, tuple):
                self._fix_paths_dict(libname[1])
            if '@' in libname:
                (lname, lpath) = libname.split('@', 1)
                lpath = os.path.abspath(njoin(self.local_path, lpath))
                if os.path.isdir(lpath):
                    c = self.get_subpackage(None, lpath, caller_level=2)
                    if isinstance(c, Configuration):
                        c = c.todict()
                    for l in [l[0] for l in c.get('libraries', [])]:
                        llname = l.split('__OF__', 1)[0]
                        if llname == lname:
                            c.pop('name', None)
                            dict_append(ext_args, **c)
                            break
                    continue
            libnames.append(libname)
        ext_args['libraries'] = libnames + ext_args['libraries']
        ext_args['define_macros'] = self.define_macros + ext_args.get('define_macros', [])
        from numpy.distutils.core import Extension
        ext = Extension(**ext_args)
        self.ext_modules.append(ext)
        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized, it may be too late to add an extension ' + name)
        return ext
    
    def add_library(self, name, sources, **build_info):
        """
        Add library to configuration.

        Parameters
        ----------
        name : str
            Name of the extension.
        sources : sequence
            List of the sources. The list of sources may contain functions
            (called source generators) which must take an extension instance
            and a build directory as inputs and return a source file or list of
            source files or None. If None is returned then no sources are
            generated. If the Extension instance has no sources after
            processing all source generators, then no extension module is
            built.
        build_info : dict, optional
            The following keys are allowed:

                * depends
                * macros
                * include_dirs
                * extra_compiler_args
                * extra_f77_compile_args
                * extra_f90_compile_args
                * f2py_options
                * language

        """
        self._add_library(name, sources, None, build_info)
        dist = self.get_distribution()
        if dist is not None:
            self.warn('distutils distribution has been initialized, it may be too late to add a library ' + name)
    
    def _add_library(self, name, sources, install_dir, build_info):
        """Common implementation for add_library and add_installed_library. Do
        not use directly"""
        build_info = copy.copy(build_info)
        build_info['sources'] = sources
        if not 'depends' in build_info:
            build_info['depends'] = []
        self._fix_paths_dict(build_info)
        self.libraries.append((name, build_info))
    
    def add_installed_library(self, name, sources, install_dir, build_info=None):
        """
        Similar to add_library, but the specified library is installed.

        Most C libraries used with `distutils` are only used to build python
        extensions, but libraries built through this method will be installed
        so that they can be reused by third-party packages.

        Parameters
        ----------
        name : str
            Name of the installed library.
        sources : sequence
            List of the library's source files. See `add_library` for details.
        install_dir : str
            Path to install the library, relative to the current sub-package.
        build_info : dict, optional
            The following keys are allowed:

                * depends
                * macros
                * include_dirs
                * extra_compiler_args
                * extra_f77_compile_args
                * extra_f90_compile_args
                * f2py_options
                * language

        Returns
        -------
        None

        See Also
        --------
        add_library, add_npy_pkg_config, get_info

        Notes
        -----
        The best way to encode the options required to link against the specified
        C libraries is to use a "libname.ini" file, and use `get_info` to
        retrieve the required options (see `add_npy_pkg_config` for more
        information).

        """
        if not build_info:
            build_info = {}
        install_dir = os.path.join(self.package_path, install_dir)
        self._add_library(name, sources, install_dir, build_info)
        self.installed_libraries.append(InstallableLib(name, build_info, install_dir))
    
    def add_npy_pkg_config(self, template, install_dir, subst_dict=None):
        """
        Generate and install a npy-pkg config file from a template.

        The config file generated from `template` is installed in the
        given install directory, using `subst_dict` for variable substitution.

        Parameters
        ----------
        template : str
            The path of the template, relatively to the current package path.
        install_dir : str
            Where to install the npy-pkg config file, relatively to the current
            package path.
        subst_dict : dict, optional
            If given, any string of the form ``@key@`` will be replaced by
            ``subst_dict[key]`` in the template file when installed. The install
            prefix is always available through the variable ``@prefix@``, since the
            install prefix is not easy to get reliably from setup.py.

        See also
        --------
        add_installed_library, get_info

        Notes
        -----
        This works for both standard installs and in-place builds, i.e. the
        ``@prefix@`` refer to the source directory for in-place builds.

        Examples
        --------
        ::

            config.add_npy_pkg_config('foo.ini.in', 'lib', {'foo': bar})

        Assuming the foo.ini.in file has the following content::

            [meta]
            Name=@foo@
            Version=1.0
            Description=dummy description

            [default]
            Cflags=-I@prefix@/include
            Libs=

        The generated file will have the following content::

            [meta]
            Name=bar
            Version=1.0
            Description=dummy description

            [default]
            Cflags=-Iprefix_dir/include
            Libs=

        and will be installed as foo.ini in the 'lib' subpath.

        When cross-compiling with numpy distutils, it might be necessary to
        use modified npy-pkg-config files.  Using the default/generated files
        will link with the host libraries (i.e. libnpymath.a).  For
        cross-compilation you of-course need to link with target libraries,
        while using the host Python installation.

        You can copy out the numpy/core/lib/npy-pkg-config directory, add a
        pkgdir value to the .ini files and set NPY_PKG_CONFIG_PATH environment
        variable to point to the directory with the modified npy-pkg-config
        files.

        Example npymath.ini modified for cross-compilation::

            [meta]
            Name=npymath
            Description=Portable, core math library implementing C99 standard
            Version=0.1

            [variables]
            pkgname=numpy.core
            pkgdir=/build/arm-linux-gnueabi/sysroot/usr/lib/python3.7/site-packages/numpy/core
            prefix=${pkgdir}
            libdir=${prefix}/lib
            includedir=${prefix}/include

            [default]
            Libs=-L${libdir} -lnpymath
            Cflags=-I${includedir}
            Requires=mlib

            [msvc]
            Libs=/LIBPATH:${libdir} npymath.lib
            Cflags=/INCLUDE:${includedir}
            Requires=mlib

        """
        if subst_dict is None:
            subst_dict = {}
        template = os.path.join(self.package_path, template)
        if self.name in self.installed_pkg_config:
            self.installed_pkg_config[self.name].append((template, install_dir, subst_dict))
        else:
            self.installed_pkg_config[self.name] = [(template, install_dir, subst_dict)]
    
    def add_scripts(self, *files):
        """Add scripts to configuration.

        Add the sequence of files to the beginning of the scripts list.
        Scripts will be installed under the <prefix>/bin/ directory.

        """
        scripts = self.paths(files)
        dist = self.get_distribution()
        if dist is not None:
            if dist.scripts is None:
                dist.scripts = []
            dist.scripts.extend(scripts)
        else:
            self.scripts.extend(scripts)
    
    def dict_append(self, **dict):
        for key in self.list_keys:
            a = getattr(self, key)
            a.extend(dict.get(key, []))
        for key in self.dict_keys:
            a = getattr(self, key)
            a.update(dict.get(key, {}))
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        for key in dict.keys():
            if key not in known_keys:
                a = getattr(self, key, None)
                if (a and a == dict[key]):
                    continue
                self.warn('Inheriting attribute %r=%r from %r' % (key, dict[key], dict.get('name', '?')))
                setattr(self, key, dict[key])
                self.extra_keys.append(key)
            elif key in self.extra_keys:
                self.info('Ignoring attempt to set %r (from %r to %r)' % (key, getattr(self, key), dict[key]))
            elif key in known_keys:
                pass
            else:
                raise ValueError("Don't know about key=%r" % key)
    
    def __str__(self):
        from pprint import pformat
        known_keys = self.list_keys + self.dict_keys + self.extra_keys
        s = '<' + 5 * '-' + '\n'
        s += 'Configuration of ' + self.name + ':\n'
        known_keys.sort()
        for k in known_keys:
            a = getattr(self, k, None)
            if a:
                s += '%s = %s\n' % (k, pformat(a))
        s += 5 * '-' + '>'
        return s
    
    def get_config_cmd(self):
        """
        Returns the numpy.distutils config command instance.
        """
        cmd = get_cmd('config')
        cmd.ensure_finalized()
        cmd.dump_source = 0
        cmd.noisy = 0
        old_path = os.environ.get('PATH')
        if old_path:
            path = os.pathsep.join(['.', old_path])
            os.environ['PATH'] = path
        return cmd
    
    def get_build_temp_dir(self):
        """
        Return a path to a temporary directory where temporary files should be
        placed.
        """
        cmd = get_cmd('build')
        cmd.ensure_finalized()
        return cmd.build_temp
    
    def have_f77c(self):
        """Check for availability of Fortran 77 compiler.

        Use it inside source generating function to ensure that
        setup distribution instance has been initialized.

        Notes
        -----
        True if a Fortran 77 compiler is available (because a simple Fortran 77
        code was able to be compiled successfully).
        """
        simple_fortran_subroutine = '\n        subroutine simple\n        end\n        '
        config_cmd = self.get_config_cmd()
        flag = config_cmd.try_compile(simple_fortran_subroutine, lang='f77')
        return flag
    
    def have_f90c(self):
        """Check for availability of Fortran 90 compiler.

        Use it inside source generating function to ensure that
        setup distribution instance has been initialized.

        Notes
        -----
        True if a Fortran 90 compiler is available (because a simple Fortran
        90 code was able to be compiled successfully)
        """
        simple_fortran_subroutine = '\n        subroutine simple\n        end\n        '
        config_cmd = self.get_config_cmd()
        flag = config_cmd.try_compile(simple_fortran_subroutine, lang='f90')
        return flag
    
    def append_to(self, extlib):
        """Append libraries, include_dirs to extension or library item.
        """
        if is_sequence(extlib):
            (lib_name, build_info) = extlib
            dict_append(build_info, libraries=self.libraries, include_dirs=self.include_dirs)
        else:
            from numpy.distutils.core import Extension
            assert isinstance(extlib, Extension), repr(extlib)
            extlib.libraries.extend(self.libraries)
            extlib.include_dirs.extend(self.include_dirs)
    
    def _get_svn_revision(self, path):
        """Return path's SVN revision number.
        """
        try:
            output = subprocess.check_output(['svnversion'], cwd=path)
        except (subprocess.CalledProcessError, OSError):
            pass
        else:
            m = re.match(b'(?P<revision>\\d+)', output)
            if m:
                return int(m.group('revision'))
        if (sys.platform == 'win32' and os.environ.get('SVN_ASP_DOT_NET_HACK', None)):
            entries = njoin(path, '_svn', 'entries')
        else:
            entries = njoin(path, '.svn', 'entries')
        if os.path.isfile(entries):
            with open(entries) as f:
                fstr = f.read()
            if fstr[:5] == '<?xml':
                m = re.search('revision="(?P<revision>\\d+)"', fstr)
                if m:
                    return int(m.group('revision'))
            else:
                m = re.search('dir[\\n\\r]+(?P<revision>\\d+)', fstr)
                if m:
                    return int(m.group('revision'))
        return None
    
    def _get_hg_revision(self, path):
        """Return path's Mercurial revision number.
        """
        try:
            output = subprocess.check_output(['hg', 'identify', '--num'], cwd=path)
        except (subprocess.CalledProcessError, OSError):
            pass
        else:
            m = re.match(b'(?P<revision>\\d+)', output)
            if m:
                return int(m.group('revision'))
        branch_fn = njoin(path, '.hg', 'branch')
        branch_cache_fn = njoin(path, '.hg', 'branch.cache')
        if os.path.isfile(branch_fn):
            branch0 = None
            with open(branch_fn) as f:
                revision0 = f.read().strip()
            branch_map = {}
            with open(branch_cache_fn, 'r') as f:
                for line in f:
                    (branch1, revision1) = line.split()[:2]
                    if revision1 == revision0:
                        branch0 = branch1
                    try:
                        revision1 = int(revision1)
                    except ValueError:
                        continue
                    branch_map[branch1] = revision1
            return branch_map.get(branch0)
        return None
    
    def get_version(self, version_file=None, version_variable=None):
        """Try to get version string of a package.

        Return a version string of the current package or None if the version
        information could not be detected.

        Notes
        -----
        This method scans files named
        __version__.py, <packagename>_version.py, version.py, and
        __svn_version__.py for string variables version, __version__, and
        <packagename>_version, until a version number is found.
        """
        version = getattr(self, 'version', None)
        if version is not None:
            return version
        if version_file is None:
            files = ['__version__.py', self.name.split('.')[-1] + '_version.py', 'version.py', '__svn_version__.py', '__hg_version__.py']
        else:
            files = [version_file]
        if version_variable is None:
            version_vars = ['version', '__version__', self.name.split('.')[-1] + '_version']
        else:
            version_vars = [version_variable]
        for f in files:
            fn = njoin(self.local_path, f)
            if os.path.isfile(fn):
                info = ('.py', 'U', 1)
                name = os.path.splitext(os.path.basename(fn))[0]
                n = dot_join(self.name, name)
                try:
                    version_module = exec_mod_from_location('_'.join(n.split('.')), fn)
                except ImportError as e:
                    self.warn(str(e))
                    version_module = None
                if version_module is None:
                    continue
                for a in version_vars:
                    version = getattr(version_module, a, None)
                    if version is not None:
                        break
                try:
                    version = version_module.get_versions()['version']
                except AttributeError:
                    pass
                if version is not None:
                    break
        if version is not None:
            self.version = version
            return version
        revision = self._get_svn_revision(self.local_path)
        if revision is None:
            revision = self._get_hg_revision(self.local_path)
        if revision is not None:
            version = str(revision)
            self.version = version
        return version
    
    def make_svn_version_py(self, delete=True):
        """Appends a data function to the data_files list that will generate
        __svn_version__.py file to the current package directory.

        Generate package __svn_version__.py file from SVN revision number,
        it will be removed after python exits but will be available
        when sdist, etc commands are executed.

        Notes
        -----
        If __svn_version__.py existed before, nothing is done.

        This is
        intended for working with source directories that are in an SVN
        repository.
        """
        target = njoin(self.local_path, '__svn_version__.py')
        revision = self._get_svn_revision(self.local_path)
        if (os.path.isfile(target) or revision is None):
            return
        else:
            
            def generate_svn_version_py():
                if not os.path.isfile(target):
                    version = str(revision)
                    self.info('Creating %s (version=%r)' % (target, version))
                    with open(target, 'w') as f:
                        f.write('version = %r\n' % version)
                
                def rm_file(f=target, p=self.info):
                    if delete:
                        try:
                            os.remove(f)
                            p('removed ' + f)
                        except OSError:
                            pass
                        try:
                            os.remove(f + 'c')
                            p('removed ' + f + 'c')
                        except OSError:
                            pass
                atexit.register(rm_file)
                return target
            self.add_data_files(('', generate_svn_version_py()))
    
    def make_hg_version_py(self, delete=True):
        """Appends a data function to the data_files list that will generate
        __hg_version__.py file to the current package directory.

        Generate package __hg_version__.py file from Mercurial revision,
        it will be removed after python exits but will be available
        when sdist, etc commands are executed.

        Notes
        -----
        If __hg_version__.py existed before, nothing is done.

        This is intended for working with source directories that are
        in an Mercurial repository.
        """
        target = njoin(self.local_path, '__hg_version__.py')
        revision = self._get_hg_revision(self.local_path)
        if (os.path.isfile(target) or revision is None):
            return
        else:
            
            def generate_hg_version_py():
                if not os.path.isfile(target):
                    version = str(revision)
                    self.info('Creating %s (version=%r)' % (target, version))
                    with open(target, 'w') as f:
                        f.write('version = %r\n' % version)
                
                def rm_file(f=target, p=self.info):
                    if delete:
                        try:
                            os.remove(f)
                            p('removed ' + f)
                        except OSError:
                            pass
                        try:
                            os.remove(f + 'c')
                            p('removed ' + f + 'c')
                        except OSError:
                            pass
                atexit.register(rm_file)
                return target
            self.add_data_files(('', generate_hg_version_py()))
    
    def make_config_py(self, name='__config__'):
        """Generate package __config__.py file containing system_info
        information used during building the package.

        This file is installed to the
        package installation directory.

        """
        self.py_modules.append((self.name, name, generate_config_py))
    
    def get_info(self, *names):
        """Get resources information.

        Return information (from system_info.get_info) for all of the names in
        the argument list in a single dictionary.
        """
        from .system_info import get_info, dict_append
        info_dict = {}
        for a in names:
            dict_append(info_dict, **get_info(a))
        return info_dict


def get_cmd(cmdname, _cache={}):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_cmd', 'get_cmd(cmdname, _cache={})', {'distutils': distutils, 'cmdname': cmdname, '_cache': _cache}, 1)

def get_numpy_include_dirs():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_numpy_include_dirs', 'get_numpy_include_dirs()', {'Configuration': Configuration}, 1)

def get_npy_pkg_dir():
    """Return the path where to find the npy-pkg-config directory.

    If the NPY_PKG_CONFIG_PATH environment variable is set, the value of that
    is returned.  Otherwise, a path inside the location of the numpy module is
    returned.

    The NPY_PKG_CONFIG_PATH can be useful when cross-compiling, maintaining
    customized npy-pkg-config .ini files for the cross-compilation
    environment, and using them when cross-compiling.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_npy_pkg_dir', 'get_npy_pkg_dir()', {'os': os, 'importlib': importlib}, 1)

def get_pkg_info(pkgname, dirs=None):
    """
    Return library info for the given package.

    Parameters
    ----------
    pkgname : str
        Name of the package (should match the name of the .ini file, without
        the extension, e.g. foo for the file foo.ini).
    dirs : sequence, optional
        If given, should be a sequence of additional directories where to look
        for npy-pkg-config files. Those directories are searched prior to the
        NumPy directory.

    Returns
    -------
    pkginfo : class instance
        The `LibraryInfo` instance containing the build information.

    Raises
    ------
    PkgNotFound
        If the package is not found.

    See Also
    --------
    Configuration.add_npy_pkg_config, Configuration.add_installed_library,
    get_info

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_pkg_info', 'get_pkg_info(pkgname, dirs=None)', {'get_npy_pkg_dir': get_npy_pkg_dir, 'pkgname': pkgname, 'dirs': dirs}, 1)

def get_info(pkgname, dirs=None):
    """
    Return an info dict for a given C library.

    The info dict contains the necessary options to use the C library.

    Parameters
    ----------
    pkgname : str
        Name of the package (should match the name of the .ini file, without
        the extension, e.g. foo for the file foo.ini).
    dirs : sequence, optional
        If given, should be a sequence of additional directories where to look
        for npy-pkg-config files. Those directories are searched prior to the
        NumPy directory.

    Returns
    -------
    info : dict
        The dictionary with build information.

    Raises
    ------
    PkgNotFound
        If the package is not found.

    See Also
    --------
    Configuration.add_npy_pkg_config, Configuration.add_installed_library,
    get_pkg_info

    Examples
    --------
    To get the necessary information for the npymath library from NumPy:

    >>> npymath_info = np.distutils.misc_util.get_info('npymath')
    >>> npymath_info                                    #doctest: +SKIP
    {'define_macros': [], 'libraries': ['npymath'], 'library_dirs':
    ['.../numpy/core/lib'], 'include_dirs': ['.../numpy/core/include']}

    This info dict can then be used as input to a `Configuration` instance::

      config.add_extension('foo', sources=['foo.c'], extra_info=npymath_info)

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.get_info', 'get_info(pkgname, dirs=None)', {'get_pkg_info': get_pkg_info, 'pkgname': pkgname, 'dirs': dirs}, 1)

def is_bootstrapping():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.is_bootstrapping', 'is_bootstrapping()', {}, 1)

def default_config_dict(name=None, parent_name=None, local_path=None):
    """Return a configuration dictionary for usage in
    configuration() function defined in file setup_<name>.py.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.default_config_dict', 'default_config_dict(name=None, parent_name=None, local_path=None)', {'Configuration': Configuration, 'name': name, 'parent_name': parent_name, 'local_path': local_path}, 1)

def dict_append(d, **kws):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.misc_util.dict_append', 'dict_append(d, **kws)', {'d': d, 'kws': kws}, 0)

def appendpath(prefix, path):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.appendpath', 'appendpath(prefix, path)', {'os': os, 'njoin': njoin, 'prefix': prefix, 'path': path}, 1)

def generate_config_py(target):
    """Generate config.py file containing system_info information
    used during building the package.

    Usage:
        config['py_modules'].append((packagename, '__config__',generate_config_py))
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.generate_config_py', 'generate_config_py(target)', {'os': os, 'sys': sys, 'textwrap': textwrap, 'target': target}, 1)

def msvc_version(compiler):
    """Return version major and minor of compiler instance if it is
    MSVC, raise an exception otherwise."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.msvc_version', 'msvc_version(compiler)', {'compiler': compiler}, 1)

def get_build_architecture():
    from distutils.msvccompiler import get_build_architecture
    return get_build_architecture()
_cxx_ignore_flags = {'-Werror=implicit-function-declaration', '-std=c99'}

def sanitize_cxx_flags(cxxflags):
    """
    Some flags are valid for C but not C++. Prune them.
    """
    return [flag for flag in cxxflags if flag not in _cxx_ignore_flags]

def exec_mod_from_location(modname, modfile):
    """
    Use importlib machinery to import a module `modname` from the file
    `modfile`. Depending on the `spec.loader`, the module may not be
    registered in sys.modules.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.misc_util.exec_mod_from_location', 'exec_mod_from_location(modname, modfile)', {'importlib': importlib, 'modname': modname, 'modfile': modfile}, 1)

