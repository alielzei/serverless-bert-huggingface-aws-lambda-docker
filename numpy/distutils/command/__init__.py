"""distutils.command

Package containing implementation of all the standard Distutils
commands.

"""


def test_na_writable_attributes_deletion():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.command.__init__.test_na_writable_attributes_deletion', 'test_na_writable_attributes_deletion()', {'np': np, 'assert_raises': assert_raises}, 0)
__revision__ = '$Id: __init__.py,v 1.3 2005/05/16 11:08:49 pearu Exp $'
distutils_all = ['clean', 'install_clib', 'install_scripts', 'bdist', 'bdist_dumb', 'bdist_wininst']
__import__('distutils.command', globals(), locals(), distutils_all)
__all__ = ['build', 'config_compiler', 'config', 'build_src', 'build_py', 'build_ext', 'build_clib', 'build_scripts', 'install', 'install_data', 'install_headers', 'install_lib', 'bdist_rpm', 'sdist'] + distutils_all

