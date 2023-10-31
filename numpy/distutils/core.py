import sys
from distutils.core import Distribution
if 'setuptools' in sys.modules:
    have_setuptools = True
    from setuptools import setup as old_setup
    from setuptools.command import easy_install
    try:
        from setuptools.command import bdist_egg
    except ImportError:
        have_setuptools = False
else:
    from distutils.core import setup as old_setup
    have_setuptools = False
import warnings
import distutils.core
import distutils.dist
from numpy.distutils.extension import Extension
from numpy.distutils.numpy_distribution import NumpyDistribution
from numpy.distutils.command import config, config_compiler, build, build_py, build_ext, build_clib, build_src, build_scripts, sdist, install_data, install_headers, install, bdist_rpm, install_clib
from numpy.distutils.misc_util import is_sequence, is_string
numpy_cmdclass = {'build': build.build, 'build_src': build_src.build_src, 'build_scripts': build_scripts.build_scripts, 'config_cc': config_compiler.config_cc, 'config_fc': config_compiler.config_fc, 'config': config.config, 'build_ext': build_ext.build_ext, 'build_py': build_py.build_py, 'build_clib': build_clib.build_clib, 'sdist': sdist.sdist, 'install_data': install_data.install_data, 'install_headers': install_headers.install_headers, 'install_clib': install_clib.install_clib, 'install': install.install, 'bdist_rpm': bdist_rpm.bdist_rpm}
if have_setuptools:
    from numpy.distutils.command import develop, egg_info
    numpy_cmdclass['bdist_egg'] = bdist_egg.bdist_egg
    numpy_cmdclass['develop'] = develop.develop
    numpy_cmdclass['easy_install'] = easy_install.easy_install
    numpy_cmdclass['egg_info'] = egg_info.egg_info

def _dict_append(d, **kws):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.core._dict_append', '_dict_append(d, **kws)', {'_dict_append': _dict_append, 'is_string': is_string, 'd': d, 'kws': kws}, 0)

def _command_line_ok(_cache=None):
    """ Return True if command line does not contain any
    help or display requests.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.core._command_line_ok', '_command_line_ok(_cache=None)', {'Distribution': Distribution, 'sys': sys, '_cache': _cache}, 1)

def get_distribution(always=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.core.get_distribution', 'get_distribution(always=False)', {'distutils': distutils, 'NumpyDistribution': NumpyDistribution, 'always': always}, 1)

def setup(**attr):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.core.setup', 'setup(**attr)', {'numpy_cmdclass': numpy_cmdclass, 'distutils': distutils, 'setup': setup, '_command_line_ok': _command_line_ok, '_dict_append': _dict_append, 'is_sequence': is_sequence, '_check_append_ext_library': _check_append_ext_library, 'is_string': is_string, '_check_append_library': _check_append_library, 'NumpyDistribution': NumpyDistribution, 'old_setup': old_setup, 'attr': attr}, 1)

def _check_append_library(libraries, item):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.core._check_append_library', '_check_append_library(libraries, item)', {'is_sequence': is_sequence, 'warnings': warnings, 'libraries': libraries, 'item': item}, 1)

def _check_append_ext_library(libraries, lib_name, build_info):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.core._check_append_ext_library', '_check_append_ext_library(libraries, lib_name, build_info)', {'is_sequence': is_sequence, 'warnings': warnings, 'libraries': libraries, 'lib_name': lib_name, 'build_info': build_info}, 1)

