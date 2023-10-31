"""This module implements additional tests ala autoconf which can be useful.

"""

import textwrap

def check_inline(cmd):
    """Return the inline identifier (may be empty)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.command.autodist.check_inline', 'check_inline(cmd)', {'textwrap': textwrap, 'cmd': cmd}, 1)

def check_restrict(cmd):
    """Return the restrict identifier (may be empty)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.command.autodist.check_restrict', 'check_restrict(cmd)', {'textwrap': textwrap, 'cmd': cmd}, 1)

def check_compiler_gcc(cmd):
    """Check if the compiler is GCC."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.command.autodist.check_compiler_gcc', 'check_compiler_gcc(cmd)', {'textwrap': textwrap, 'cmd': cmd}, 1)

def check_gcc_version_at_least(cmd, major, minor=0, patchlevel=0):
    """
    Check that the gcc version is at least the specified version."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.command.autodist.check_gcc_version_at_least', 'check_gcc_version_at_least(cmd, major, minor=0, patchlevel=0)', {'textwrap': textwrap, 'cmd': cmd, 'major': major, 'minor': minor, 'patchlevel': patchlevel}, 1)

def check_gcc_function_attribute(cmd, attribute, name):
    """Return True if the given function attribute is supported."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.command.autodist.check_gcc_function_attribute', 'check_gcc_function_attribute(cmd, attribute, name)', {'textwrap': textwrap, 'cmd': cmd, 'attribute': attribute, 'name': name}, 1)

def check_gcc_function_attribute_with_intrinsics(cmd, attribute, name, code, include):
    """Return True if the given function attribute is supported with
    intrinsics."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.command.autodist.check_gcc_function_attribute_with_intrinsics', 'check_gcc_function_attribute_with_intrinsics(cmd, attribute, name, code, include)', {'textwrap': textwrap, 'cmd': cmd, 'attribute': attribute, 'name': name, 'code': code, 'include': include}, 1)

def check_gcc_variable_attribute(cmd, attribute):
    """Return True if the given variable attribute is supported."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.command.autodist.check_gcc_variable_attribute', 'check_gcc_variable_attribute(cmd, attribute)', {'textwrap': textwrap, 'cmd': cmd, 'attribute': attribute}, 1)

