"""
Support code for building Python extensions on Windows.

    # NT stuff
    # 1. Make sure libpython<version>.a exists for gcc.  If not, build it.
    # 2. Force windows to use gcc (we're struggling with MSVC and g77 support)
    # 3. Force windows to use g77

"""

import os
import platform
import sys
import subprocess
import re
import textwrap
import numpy.distutils.ccompiler
from numpy.distutils import log
import distutils.cygwinccompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import get_build_version as get_build_msvc_version
from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import msvc_runtime_library, msvc_runtime_version, msvc_runtime_major, get_build_architecture

def get_msvcr_replacement():
    """Replacement for outdated version of get_msvcr from cygwinccompiler"""
    msvcr = msvc_runtime_library()
    return ([] if msvcr is None else [msvcr])
_START = re.compile('\\[Ordinal/Name Pointer\\] Table')
_TABLE = re.compile('^\\s+\\[([\\s*[0-9]*)\\] ([a-zA-Z0-9_]*)')


class Mingw32CCompiler(distutils.cygwinccompiler.CygwinCCompiler):
    """ A modified MingW32 compiler compatible with an MSVC built Python.

    """
    compiler_type = 'mingw32'
    
    def __init__(self, verbose=0, dry_run=0, force=0):
        distutils.cygwinccompiler.CygwinCCompiler.__init__(self, verbose, dry_run, force)
        build_import_library()
        msvcr_success = build_msvcr_library()
        msvcr_dbg_success = build_msvcr_library(debug=True)
        if (msvcr_success or msvcr_dbg_success):
            self.define_macro('NPY_MINGW_USE_CUSTOM_MSVCR')
        msvcr_version = msvc_runtime_version()
        if msvcr_version:
            self.define_macro('__MSVCRT_VERSION__', '0x%04i' % msvcr_version)
        if get_build_architecture() == 'AMD64':
            self.set_executables(compiler='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall', compiler_so='gcc -g -DDEBUG -DMS_WIN64 -O0 -Wall -Wstrict-prototypes', linker_exe='gcc -g', linker_so='gcc -g -shared')
        else:
            self.set_executables(compiler='gcc -O2 -Wall', compiler_so='gcc -O2 -Wall -Wstrict-prototypes', linker_exe='g++ ', linker_so='g++ -shared')
        self.compiler_cxx = ['g++']
        return
    
    def link(self, target_desc, objects, output_filename, output_dir, libraries, library_dirs, runtime_library_dirs, export_symbols=None, debug=0, extra_preargs=None, extra_postargs=None, build_temp=None, target_lang=None):
        runtime_library = msvc_runtime_library()
        if runtime_library:
            if not libraries:
                libraries = []
            libraries.append(runtime_library)
        args = (self, target_desc, objects, output_filename, output_dir, libraries, library_dirs, runtime_library_dirs, None, debug, extra_preargs, extra_postargs, build_temp, target_lang)
        func = UnixCCompiler.link
        func(*args[:func.__code__.co_argcount])
        return
    
    def object_filenames(self, source_filenames, strip_dir=0, output_dir=''):
        if output_dir is None:
            output_dir = ''
        obj_names = []
        for src_name in source_filenames:
            (base, ext) = os.path.splitext(os.path.normcase(src_name))
            (drv, base) = os.path.splitdrive(base)
            if drv:
                base = base[1:]
            if ext not in self.src_extensions + ['.rc', '.res']:
                raise UnknownFileError("unknown file type '%s' (from '%s')" % (ext, src_name))
            if strip_dir:
                base = os.path.basename(base)
            if (ext == '.res' or ext == '.rc'):
                obj_names.append(os.path.join(output_dir, base + ext + self.obj_extension))
            else:
                obj_names.append(os.path.join(output_dir, base + self.obj_extension))
        return obj_names


def find_python_dll():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler.find_python_dll', 'find_python_dll()', {'sys': sys, 'os': os, 'platform': platform}, 1)

def dump_table(dll):
    st = subprocess.check_output(['objdump.exe', '-p', dll])
    return st.split(b'\n')

def generate_def(dll, dfile):
    """Given a dll file location,  get all its exported symbols and dump them
    into the given def file.

    The .def file will be overwritten"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler.generate_def', 'generate_def(dll, dfile)', {'dump_table': dump_table, '_START': _START, '_TABLE': _TABLE, 'log': log, 'os': os, 'dll': dll, 'dfile': dfile}, 0)

def find_dll(dll_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler.find_dll', 'find_dll(dll_name)', {'get_build_architecture': get_build_architecture, 'os': os, 'sys': sys, 'dll_name': dll_name}, 1)

def build_msvcr_library(debug=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler.build_msvcr_library', 'build_msvcr_library(debug=False)', {'os': os, 'msvc_runtime_major': msvc_runtime_major, 'log': log, 'msvc_runtime_library': msvc_runtime_library, 'sys': sys, 'find_dll': find_dll, 'generate_def': generate_def, 'subprocess': subprocess, 'debug': debug}, 1)

def build_import_library():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler.build_import_library', 'build_import_library()', {'os': os, 'get_build_architecture': get_build_architecture, '_build_import_library_amd64': _build_import_library_amd64, '_build_import_library_x86': _build_import_library_x86}, 1)

def _check_for_import_lib():
    """Check if an import library for the Python runtime already exists."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler._check_for_import_lib', '_check_for_import_lib()', {'sys': sys, 'os': os}, 2)

def _build_import_library_amd64():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler._build_import_library_amd64', '_build_import_library_amd64()', {'_check_for_import_lib': _check_for_import_lib, 'log': log, 'find_python_dll': find_python_dll, 'sys': sys, 'os': os, 'generate_def': generate_def, 'subprocess': subprocess}, 1)

def _build_import_library_x86():
    """ Build the import libraries for Mingw32-gcc on Windows
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler._build_import_library_x86', '_build_import_library_x86()', {'_check_for_import_lib': _check_for_import_lib, 'log': log, 'sys': sys, 'os': os, 'find_python_dll': find_python_dll, 'subprocess': subprocess}, 1)
_MSVCRVER_TO_FULLVER = {}
if sys.platform == 'win32':
    try:
        import msvcrt
        _MSVCRVER_TO_FULLVER['80'] = '8.0.50727.42'
        _MSVCRVER_TO_FULLVER['90'] = '9.0.21022.8'
        _MSVCRVER_TO_FULLVER['100'] = '10.0.30319.460'
        crt_ver = getattr(msvcrt, 'CRT_ASSEMBLY_VERSION', None)
        if crt_ver is not None:
            (maj, min) = re.match('(\\d+)\\.(\\d)', crt_ver).groups()
            _MSVCRVER_TO_FULLVER[maj + min] = crt_ver
            del maj, min
        del crt_ver
    except ImportError:
        log.warn('Cannot import msvcrt: using manifest will not be possible')

def msvc_manifest_xml(maj, min):
    """Given a major and minor version of the MSVCR, returns the
    corresponding XML file."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler.msvc_manifest_xml', 'msvc_manifest_xml(maj, min)', {'_MSVCRVER_TO_FULLVER': _MSVCRVER_TO_FULLVER, 'textwrap': textwrap, 'maj': maj, 'min': min}, 1)

def manifest_rc(name, type='dll'):
    """Return the rc file used to generate the res file which will be embedded
    as manifest for given manifest file name, of given type ('dll' or
    'exe').

    Parameters
    ----------
    name : str
            name of the manifest file to embed
    type : str {'dll', 'exe'}
            type of the binary which will embed the manifest

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler.manifest_rc', "manifest_rc(name, type='dll')", {'name': name, 'type': type}, 1)

def check_embedded_msvcr_match_linked(msver):
    """msver is the ms runtime version used for the MANIFEST."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler.check_embedded_msvcr_match_linked', 'check_embedded_msvcr_match_linked(msver)', {'msvc_runtime_major': msvc_runtime_major, 'msver': msver}, 0)

def configtest_name(config):
    base = os.path.basename(config._gen_temp_sourcefile('yo', [], 'c'))
    return os.path.splitext(base)[0]

def manifest_name(config):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler.manifest_name', 'manifest_name(config)', {'configtest_name': configtest_name, 'config': config}, 1)

def rc_name(config):
    root = configtest_name(config)
    return root + '.rc'

def generate_manifest(config):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.mingw32ccompiler.generate_manifest', 'generate_manifest(config)', {'get_build_msvc_version': get_build_msvc_version, 'check_embedded_msvcr_match_linked': check_embedded_msvcr_match_linked, 'msvc_manifest_xml': msvc_manifest_xml, 'manifest_name': manifest_name, 'config': config}, 0)

