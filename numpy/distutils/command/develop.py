""" Override the develop command from setuptools so we can ensure that our
generated files (from build_src or build_scripts) are properly converted to real
files with filenames.

"""

from setuptools.command.develop import develop as old_develop


class develop(old_develop):
    __doc__ = old_develop.__doc__
    
    def install_for_development(self):
        self.reinitialize_command('build_src', inplace=1)
        self.run_command('build_scripts')
        old_develop.install_for_development(self)


