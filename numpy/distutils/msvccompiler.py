import os
from distutils.msvccompiler import MSVCCompiler as _MSVCCompiler
from .system_info import platform_bits

def _merge(old, new):
    """Concatenate two environment paths avoiding repeats.

    Here `old` is the environment string before the base class initialize
    function is called and `new` is the string after the call. The new string
    will be a fixed string if it is not obtained from the current environment,
    or the same as the old string if obtained from the same environment. The aim
    here is not to append the new string if it is already contained in the old
    string so as to limit the growth of the environment string.

    Parameters
    ----------
    old : string
        Previous environment string.
    new : string
        New environment string.

    Returns
    -------
    ret : string
        Updated environment string.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.msvccompiler._merge', '_merge(old, new)', {'old': old, 'new': new}, 1)


class MSVCCompiler(_MSVCCompiler):
    
    def __init__(self, verbose=0, dry_run=0, force=0):
        _MSVCCompiler.__init__(self, verbose, dry_run, force)
    
    def initialize(self):
        environ_lib = os.getenv('lib', '')
        environ_include = os.getenv('include', '')
        _MSVCCompiler.initialize(self)
        os.environ['lib'] = _merge(environ_lib, os.environ['lib'])
        os.environ['include'] = _merge(environ_include, os.environ['include'])
        if platform_bits == 32:
            self.compile_options += ['/arch:SSE2']
            self.compile_options_debug += ['/arch:SSE2']


def lib_opts_if_msvc(build_cmd):
    """ Add flags if we are using MSVC compiler

    We can't see `build_cmd` in our scope, because we have not initialized
    the distutils build command, so use this deferred calculation to run
    when we are building the library.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.msvccompiler.lib_opts_if_msvc', 'lib_opts_if_msvc(build_cmd)', {'build_cmd': build_cmd}, 1)

