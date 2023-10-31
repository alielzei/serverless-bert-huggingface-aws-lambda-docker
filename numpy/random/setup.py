import os
import sys
from os.path import join
from numpy.distutils.system_info import platform_bits
from numpy.distutils.msvccompiler import lib_opts_if_msvc

def configuration(parent_package='', top_path=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.random.setup.configuration', "configuration(parent_package='', top_path=None)", {'sys': sys, 'os': os, 'lib_opts_if_msvc': lib_opts_if_msvc, 'parent_package': parent_package, 'top_path': top_path}, 1)
if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration)

