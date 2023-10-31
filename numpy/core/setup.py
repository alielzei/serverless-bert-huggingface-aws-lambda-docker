import os
import sys
import sysconfig
import pickle
import copy
import warnings
import textwrap
import glob
from os.path import join
from numpy.distutils import log
from numpy.distutils.msvccompiler import lib_opts_if_msvc
from distutils.dep_util import newer
from sysconfig import get_config_var
from numpy.compat import npy_load_module
from setup_common import *
NPY_RELAXED_STRIDES_CHECKING = os.environ.get('NPY_RELAXED_STRIDES_CHECKING', '1') != '0'
if not NPY_RELAXED_STRIDES_CHECKING:
    raise SystemError('Support for NPY_RELAXED_STRIDES_CHECKING=0 has been remove as of NumPy 1.23.  This error will eventually be removed entirely.')
NPY_RELAXED_STRIDES_DEBUG = os.environ.get('NPY_RELAXED_STRIDES_DEBUG', '0') != '0'
NPY_RELAXED_STRIDES_DEBUG = (NPY_RELAXED_STRIDES_DEBUG and NPY_RELAXED_STRIDES_CHECKING)
NPY_DISABLE_SVML = os.environ.get('NPY_DISABLE_SVML', '0') == '1'


class CallOnceOnly:
    
    def __init__(self):
        self._check_types = None
        self._check_ieee_macros = None
        self._check_complex = None
    
    def check_types(self, *a, **kw):
        if self._check_types is None:
            out = check_types(*a, **kw)
            self._check_types = pickle.dumps(out)
        else:
            out = copy.deepcopy(pickle.loads(self._check_types))
        return out
    
    def check_ieee_macros(self, *a, **kw):
        if self._check_ieee_macros is None:
            out = check_ieee_macros(*a, **kw)
            self._check_ieee_macros = pickle.dumps(out)
        else:
            out = copy.deepcopy(pickle.loads(self._check_ieee_macros))
        return out
    
    def check_complex(self, *a, **kw):
        if self._check_complex is None:
            out = check_complex(*a, **kw)
            self._check_complex = pickle.dumps(out)
        else:
            out = copy.deepcopy(pickle.loads(self._check_complex))
        return out


def can_link_svml():
    """SVML library is supported only on x86_64 architecture and currently
    only on linux
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup.can_link_svml', 'can_link_svml()', {'NPY_DISABLE_SVML': NPY_DISABLE_SVML, 'sysconfig': sysconfig, 'sys': sys}, 1)

def check_svml_submodule(svmlpath):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup.check_svml_submodule', 'check_svml_submodule(svmlpath)', {'os': os, 'svmlpath': svmlpath}, 1)

def pythonlib_dir():
    """return path where libpython* is."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup.pythonlib_dir', 'pythonlib_dir()', {'sys': sys, 'os': os, 'get_config_var': get_config_var}, 1)

def is_npy_no_signal():
    """Return True if the NPY_NO_SIGNAL symbol must be defined in configuration
    header."""
    return sys.platform == 'win32'

def is_npy_no_smp():
    """Return True if the NPY_NO_SMP symbol must be defined in public
    header (when SMP support cannot be reliably enabled)."""
    return 'NPY_NOSMP' in os.environ

def win32_checks(deflist):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.core.setup.win32_checks', 'win32_checks(deflist)', {'os': os, 'sys': sys, 'deflist': deflist}, 0)

def check_math_capabilities(config, ext, moredefs, mathlibs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup.check_math_capabilities', 'check_math_capabilities(config, ext, moredefs, mathlibs)', {'FUNC_CALL_ARGS': FUNC_CALL_ARGS, 'fname2def': fname2def, 'sys': sys, 'MANDATORY_FUNCS': MANDATORY_FUNCS, 'OPTIONAL_FUNCS_MAYBE': OPTIONAL_FUNCS_MAYBE, 'OPTIONAL_FILE_FUNCS': OPTIONAL_FILE_FUNCS, 'OPTIONAL_MISC_FUNCS': OPTIONAL_MISC_FUNCS, 'OPTIONAL_HEADERS': OPTIONAL_HEADERS, 'os': os, 'OPTIONAL_LOCALE_FUNCS': OPTIONAL_LOCALE_FUNCS, 'OPTIONAL_INTRINSICS': OPTIONAL_INTRINSICS, 'OPTIONAL_FUNCTION_ATTRIBUTES': OPTIONAL_FUNCTION_ATTRIBUTES, 'sysconfig': sysconfig, 'OPTIONAL_FUNCTION_ATTRIBUTES_AVX': OPTIONAL_FUNCTION_ATTRIBUTES_AVX, 'OPTIONAL_FUNCTION_ATTRIBUTES_WITH_INTRINSICS_AVX': OPTIONAL_FUNCTION_ATTRIBUTES_WITH_INTRINSICS_AVX, 'OPTIONAL_VARIABLE_ATTRIBUTES': OPTIONAL_VARIABLE_ATTRIBUTES, 'config': config, 'ext': ext, 'moredefs': moredefs, 'mathlibs': mathlibs}, 1)

def check_complex(config, mathlibs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup.check_complex', 'check_complex(config, mathlibs)', {'C99_COMPLEX_TYPES': C99_COMPLEX_TYPES, 'type2def': type2def, 'C99_COMPLEX_FUNCS': C99_COMPLEX_FUNCS, 'fname2def': fname2def, 'config': config, 'mathlibs': mathlibs}, 2)

def check_ieee_macros(config):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup.check_ieee_macros', 'check_ieee_macros(config)', {'fname2def': fname2def, 'config': config}, 2)

def check_types(config_cmd, ext, build_dir):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup.check_types', 'check_types(config_cmd, ext, build_dir)', {'sys': sys, 'sym2def': sym2def, 'pythonlib_dir': pythonlib_dir, 'config_cmd': config_cmd, 'ext': ext, 'build_dir': build_dir}, 2)

def check_mathlib(config_cmd):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup.check_mathlib', 'check_mathlib(config_cmd)', {'os': os, 'config_cmd': config_cmd}, 1)

def visibility_define(config):
    """Return the define value to use for NPY_VISIBILITY_HIDDEN (may be empty
    string)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup.visibility_define', 'visibility_define(config)', {'config': config}, 1)

def configuration(parent_package='', top_path=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.core.setup.configuration', "configuration(parent_package='', top_path=None)", {'check_api_version': check_api_version, 'C_API_VERSION': C_API_VERSION, 'CallOnceOnly': CallOnceOnly, 'os': os, 'newer': newer, '__file__': __file__, 'log': log, 'check_mathlib': check_mathlib, 'check_math_capabilities': check_math_capabilities, 'is_npy_no_signal': is_npy_no_signal, 'sys': sys, 'win32_checks': win32_checks, 'can_link_svml': can_link_svml, 'NPY_RELAXED_STRIDES_DEBUG': NPY_RELAXED_STRIDES_DEBUG, 'check_long_double_representation': check_long_double_representation, 'check_for_right_shift_internal_compiler_error': check_for_right_shift_internal_compiler_error, 'textwrap': textwrap, 'is_npy_no_smp': is_npy_no_smp, 'visibility_define': visibility_define, 'C_ABI_VERSION': C_ABI_VERSION, 'lib_opts_if_msvc': lib_opts_if_msvc, 'check_svml_submodule': check_svml_submodule, 'glob': glob, 'parent_package': parent_package, 'top_path': top_path}, 1)
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)

