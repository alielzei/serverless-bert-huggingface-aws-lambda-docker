"""
Build a c-extension module on-the-fly in tests.
See build_and_import_extensions for usage hints

"""

import os
import pathlib
import sys
import sysconfig
__all__ = ['build_and_import_extension', 'compile_extension_module']

def build_and_import_extension(modname, functions, *, prologue='', build_dir=None, include_dirs=[], more_init=''):
    """
    Build and imports a c-extension module `modname` from a list of function
    fragments `functions`.


    Parameters
    ----------
    functions : list of fragments
        Each fragment is a sequence of func_name, calling convention, snippet.
    prologue : string
        Code to precede the rest, usually extra ``#include`` or ``#define``
        macros.
    build_dir : pathlib.Path
        Where to build the module, usually a temporary directory
    include_dirs : list
        Extra directories to find include files when compiling
    more_init : string
        Code to appear in the module PyMODINIT_FUNC

    Returns
    -------
    out: module
        The module will have been loaded and is ready for use

    Examples
    --------
    >>> functions = [("test_bytes", "METH_O", "
        if ( !PyBytesCheck(args)) {
            Py_RETURN_FALSE;
        }
        Py_RETURN_TRUE;
    ")]
    >>> mod = build_and_import_extension("testme", functions)
    >>> assert not mod.test_bytes(u'abc')
    >>> assert mod.test_bytes(b'abc')
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.extbuild.build_and_import_extension', "build_and_import_extension(modname, functions, prologue='', build_dir=None, include_dirs=[], more_init='')", {'_make_methods': _make_methods, 'pathlib': pathlib, '_make_source': _make_source, 'compile_extension_module': compile_extension_module, 'importlib': importlib, 'modname': modname, 'functions': functions, 'prologue': prologue, 'build_dir': build_dir, 'include_dirs': include_dirs, 'more_init': more_init}, 1)

def compile_extension_module(name, builddir, include_dirs, source_string, libraries=[], library_dirs=[]):
    """
    Build an extension module and return the filename of the resulting
    native code file.

    Parameters
    ----------
    name : string
        name of the module, possibly including dots if it is a module inside a
        package.
    builddir : pathlib.Path
        Where to build the module, usually a temporary directory
    include_dirs : list
        Extra directories to find include files when compiling
    libraries : list
        Libraries to link into the extension module
    library_dirs: list
        Where to find the libraries, ``-L`` passed to the linker
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.extbuild.compile_extension_module', 'compile_extension_module(name, builddir, include_dirs, source_string, libraries=[], library_dirs=[])', {'_convert_str_to_file': _convert_str_to_file, 'sysconfig': sysconfig, '_c_compile': _c_compile, 'name': name, 'builddir': builddir, 'include_dirs': include_dirs, 'source_string': source_string, 'libraries': libraries, 'library_dirs': library_dirs}, 1)

def _convert_str_to_file(source, dirname):
    """Helper function to create a file ``source.c`` in `dirname` that contains
    the string in `source`. Returns the file name
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.extbuild._convert_str_to_file', '_convert_str_to_file(source, dirname)', {'source': source, 'dirname': dirname}, 1)

def _make_methods(functions, modname):
    """ Turns the name, signature, code in functions into complete functions
    and lists them in a methods_table. Then turns the methods_table into a
    ``PyMethodDef`` structure and returns the resulting code fragment ready
    for compilation
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.extbuild._make_methods', '_make_methods(functions, modname)', {'functions': functions, 'modname': modname}, 1)

def _make_source(name, init, body):
    """ Combines the code fragments into source code ready to be compiled
    """
    code = '\n    #include <Python.h>\n\n    %(body)s\n\n    PyMODINIT_FUNC\n    PyInit_%(name)s(void) {\n    %(init)s\n    }\n    ' % dict(name=name, init=init, body=body)
    return code

def _c_compile(cfile, outputfilename, include_dirs=[], libraries=[], library_dirs=[]):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.extbuild._c_compile', '_c_compile(cfile, outputfilename, include_dirs=[], libraries=[], library_dirs=[])', {'sys': sys, 'os': os, 'get_so_suffix': get_so_suffix, 'build': build, 'cfile': cfile, 'outputfilename': outputfilename, 'include_dirs': include_dirs, 'libraries': libraries, 'library_dirs': library_dirs}, 1)

def build(cfile, outputfilename, compile_extra, link_extra, include_dirs, libraries, library_dirs):
    """cd into the directory where the cfile is, use distutils to build"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.testing._private.extbuild.build', 'build(cfile, outputfilename, compile_extra, link_extra, include_dirs, libraries, library_dirs)', {'os': os, 'cfile': cfile, 'outputfilename': outputfilename, 'compile_extra': compile_extra, 'link_extra': link_extra, 'include_dirs': include_dirs, 'libraries': libraries, 'library_dirs': library_dirs}, 0)

def get_so_suffix():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.extbuild.get_so_suffix', 'get_so_suffix()', {'sysconfig': sysconfig}, 1)

