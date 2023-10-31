"""
Nose test running.

This module implements ``test()`` and ``bench()`` functions for NumPy modules.

"""

import os
import sys
import warnings
import numpy as np
from .utils import import_nose, suppress_warnings
__all__ = ['get_package_name', 'run_module_suite', 'NoseTester', '_numpy_tester', 'get_package_name', 'import_nose', 'suppress_warnings']

def get_package_name(filepath):
    """
    Given a path where a package is installed, determine its name.

    Parameters
    ----------
    filepath : str
        Path to a file. If the determination fails, "numpy" is returned.

    Examples
    --------
    >>> np.testing.nosetester.get_package_name('nonsense')
    'numpy'

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.nosetester.get_package_name', 'get_package_name(filepath)', {'os': os, 'filepath': filepath}, 1)

def run_module_suite(file_to_run=None, argv=None):
    """
    Run a test module.

    Equivalent to calling ``$ nosetests <argv> <file_to_run>`` from
    the command line

    Parameters
    ----------
    file_to_run : str, optional
        Path to test module, or None.
        By default, run the module from which this function is called.
    argv : list of strings
        Arguments to be passed to the nose test runner. ``argv[0]`` is
        ignored. All command line arguments accepted by ``nosetests``
        will work. If it is the default value None, sys.argv is used.

        .. versionadded:: 1.9.0

    Examples
    --------
    Adding the following::

        if __name__ == "__main__" :
            run_module_suite(argv=sys.argv)

    at the end of a test module will run the tests when that module is
    called in the python interpreter.

    Alternatively, calling::

    >>> run_module_suite(file_to_run="numpy/tests/test_matlib.py")  # doctest: +SKIP

    from an interpreter will run all the test routine in 'test_matlib.py'.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.testing._private.nosetester.run_module_suite', 'run_module_suite(file_to_run=None, argv=None)', {'sys': sys, 'import_nose': import_nose, 'file_to_run': file_to_run, 'argv': argv}, 0)


class NoseTester:
    """
    Nose test runner.

    This class is made available as numpy.testing.Tester, and a test function
    is typically added to a package's __init__.py like so::

      from numpy.testing import Tester
      test = Tester().test

    Calling this test function finds and runs all tests associated with the
    package and all its sub-packages.

    Attributes
    ----------
    package_path : str
        Full path to the package to test.
    package_name : str
        Name of the package to test.

    Parameters
    ----------
    package : module, str or None, optional
        The package to test. If a string, this should be the full path to
        the package. If None (default), `package` is set to the module from
        which `NoseTester` is initialized.
    raise_warnings : None, str or sequence of warnings, optional
        This specifies which warnings to configure as 'raise' instead
        of being shown once during the test execution.  Valid strings are:

          - "develop" : equals ``(Warning,)``
          - "release" : equals ``()``, don't raise on any warnings.

        Default is "release".
    depth : int, optional
        If `package` is None, then this can be used to initialize from the
        module of the caller of (the caller of (...)) the code that
        initializes `NoseTester`. Default of 0 means the module of the
        immediate caller; higher values are useful for utility routines that
        want to initialize `NoseTester` objects on behalf of other code.

    """
    
    def __init__(self, package=None, raise_warnings='release', depth=0, check_fpu_mode=False):
        if raise_warnings is None:
            raise_warnings = 'release'
        package_name = None
        if package is None:
            f = sys._getframe(1 + depth)
            package_path = f.f_locals.get('__file__', None)
            if package_path is None:
                raise AssertionError
            package_path = os.path.dirname(package_path)
            package_name = f.f_locals.get('__name__', None)
        elif isinstance(package, type(os)):
            package_path = os.path.dirname(package.__file__)
            package_name = getattr(package, '__name__', None)
        else:
            package_path = str(package)
        self.package_path = package_path
        if package_name is None:
            package_name = get_package_name(package_path)
        self.package_name = package_name
        self.raise_warnings = raise_warnings
        self.check_fpu_mode = check_fpu_mode
    
    def _test_argv(self, label, verbose, extra_argv):
        """ Generate argv for nosetest command

        Parameters
        ----------
        label : {'fast', 'full', '', attribute identifier}, optional
            see ``test`` docstring
        verbose : int, optional
            Verbosity value for test outputs, in the range 1-10. Default is 1.
        extra_argv : list, optional
            List with any extra arguments to pass to nosetests.

        Returns
        -------
        argv : list
            command line arguments that will be passed to nose
        """
        argv = [__file__, self.package_path, '-s']
        if (label and label != 'full'):
            if not isinstance(label, str):
                raise TypeError('Selection label should be a string')
            if label == 'fast':
                label = 'not slow'
            argv += ['-A', label]
        argv += ['--verbosity', str(verbose)]
        argv += ['--exe']
        if extra_argv:
            argv += extra_argv
        return argv
    
    def _show_system_info(self):
        nose = import_nose()
        import numpy
        print(f'NumPy version {numpy.__version__}')
        relaxed_strides = numpy.ones((10, 1), order='C').flags.f_contiguous
        print('NumPy relaxed strides checking option:', relaxed_strides)
        npdir = os.path.dirname(numpy.__file__)
        print(f'NumPy is installed in {npdir}')
        if 'scipy' in self.package_name:
            import scipy
            print(f'SciPy version {scipy.__version__}')
            spdir = os.path.dirname(scipy.__file__)
            print(f'SciPy is installed in {spdir}')
        pyversion = sys.version.replace('\n', '')
        print(f'Python version {pyversion}')
        print('nose version %d.%d.%d' % nose.__versioninfo__)
    
    def _get_custom_doctester(self):
        """ Return instantiated plugin for doctests

        Allows subclassing of this class to override doctester

        A return value of None means use the nose builtin doctest plugin
        """
        from .noseclasses import NumpyDoctest
        return NumpyDoctest()
    
    def prepare_test_args(self, label='fast', verbose=1, extra_argv=None, doctests=False, coverage=False, timer=False):
        """
        Run tests for module using nose.

        This method does the heavy lifting for the `test` method. It takes all
        the same arguments, for details see `test`.

        See Also
        --------
        test

        """
        import_nose()
        argv = self._test_argv(label, verbose, extra_argv)
        if coverage:
            argv += [f'--cover-package={self.package_name}', '--with-coverage', '--cover-tests', '--cover-erase']
        if timer:
            if timer is True:
                argv += ['--with-timer']
            elif isinstance(timer, int):
                argv += ['--with-timer', '--timer-top-n', str(timer)]
        import nose.plugins.builtin
        from nose.plugins import EntryPointPluginManager
        from .noseclasses import KnownFailurePlugin, Unplugger, FPUModeCheckPlugin
        plugins = [KnownFailurePlugin()]
        plugins += [p() for p in nose.plugins.builtin.plugins]
        if self.check_fpu_mode:
            plugins += [FPUModeCheckPlugin()]
            argv += ['--with-fpumodecheckplugin']
        try:
            entrypoint_manager = EntryPointPluginManager()
            entrypoint_manager.loadPlugins()
            plugins += [p for p in entrypoint_manager.plugins]
        except ImportError:
            pass
        doctest_argv = '--with-doctest' in argv
        if (doctests == False and doctest_argv):
            doctests = True
        plug = self._get_custom_doctester()
        if plug is None:
            if (doctests and not doctest_argv):
                argv += ['--with-doctest']
        else:
            if doctest_argv:
                argv.remove('--with-doctest')
            plugins += [Unplugger('doctest'), plug]
            if doctests:
                argv += ['--with-' + plug.name]
        return (argv, plugins)
    
    def test(self, label='fast', verbose=1, extra_argv=None, doctests=False, coverage=False, raise_warnings=None, timer=False):
        """
        Run tests for module using nose.

        Parameters
        ----------
        label : {'fast', 'full', '', attribute identifier}, optional
            Identifies the tests to run. This can be a string to pass to
            the nosetests executable with the '-A' option, or one of several
            special values.  Special values are:

            * 'fast' - the default - which corresponds to the ``nosetests -A``
              option of 'not slow'.
            * 'full' - fast (as above) and slow tests as in the
              'no -A' option to nosetests - this is the same as ''.
            * None or '' - run all tests.
            * attribute_identifier - string passed directly to nosetests as '-A'.

        verbose : int, optional
            Verbosity value for test outputs, in the range 1-10. Default is 1.
        extra_argv : list, optional
            List with any extra arguments to pass to nosetests.
        doctests : bool, optional
            If True, run doctests in module. Default is False.
        coverage : bool, optional
            If True, report coverage of NumPy code. Default is False.
            (This requires the
            `coverage module <https://pypi.org/project/coverage/>`_).
        raise_warnings : None, str or sequence of warnings, optional
            This specifies which warnings to configure as 'raise' instead
            of being shown once during the test execution. Valid strings are:

            * "develop" : equals ``(Warning,)``
            * "release" : equals ``()``, do not raise on any warnings.
        timer : bool or int, optional
            Timing of individual tests with ``nose-timer`` (which needs to be
            installed).  If True, time tests and report on all of them.
            If an integer (say ``N``), report timing results for ``N`` slowest
            tests.

        Returns
        -------
        result : object
            Returns the result of running the tests as a
            ``nose.result.TextTestResult`` object.

        Notes
        -----
        Each NumPy module exposes `test` in its namespace to run all tests for it.
        For example, to run all tests for numpy.lib:

        >>> np.lib.test() #doctest: +SKIP

        Examples
        --------
        >>> result = np.lib.test() #doctest: +SKIP
        Running unit tests for numpy.lib
        ...
        Ran 976 tests in 3.933s

        OK

        >>> result.errors #doctest: +SKIP
        []
        >>> result.knownfail #doctest: +SKIP
        []
        """
        verbose = min(verbose, 3)
        from . import utils
        utils.verbose = verbose
        (argv, plugins) = self.prepare_test_args(label, verbose, extra_argv, doctests, coverage, timer)
        if doctests:
            print(f'Running unit tests and doctests for {self.package_name}')
        else:
            print(f'Running unit tests for {self.package_name}')
        self._show_system_info()
        import doctest
        doctest.master = None
        if raise_warnings is None:
            raise_warnings = self.raise_warnings
        _warn_opts = dict(develop=(Warning, ), release=())
        if isinstance(raise_warnings, str):
            raise_warnings = _warn_opts[raise_warnings]
        with suppress_warnings('location') as sup:
            warnings.resetwarnings()
            warnings.filterwarnings('always')
            for warningtype in raise_warnings:
                warnings.filterwarnings('error', category=warningtype)
            sup.filter(message='Not importing directory')
            sup.filter(message='numpy.dtype size changed')
            sup.filter(message='numpy.ufunc size changed')
            sup.filter(category=np.ModuleDeprecationWarning)
            sup.filter(message='.*boolean negative.*')
            sup.filter(message='.*boolean subtract.*')
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                from ...distutils import cpuinfo
            sup.filter(category=UserWarning, module=cpuinfo)
            warnings.filterwarnings('ignore', message='.*getargspec.*', category=DeprecationWarning, module='nose\\.')
            from .noseclasses import NumpyTestProgram
            t = NumpyTestProgram(argv=argv, exit=False, plugins=plugins)
        return t.result
    
    def bench(self, label='fast', verbose=1, extra_argv=None):
        """
        Run benchmarks for module using nose.

        Parameters
        ----------
        label : {'fast', 'full', '', attribute identifier}, optional
            Identifies the benchmarks to run. This can be a string to pass to
            the nosetests executable with the '-A' option, or one of several
            special values.  Special values are:

            * 'fast' - the default - which corresponds to the ``nosetests -A``
              option of 'not slow'.
            * 'full' - fast (as above) and slow benchmarks as in the
              'no -A' option to nosetests - this is the same as ''.
            * None or '' - run all tests.
            * attribute_identifier - string passed directly to nosetests as '-A'.

        verbose : int, optional
            Verbosity value for benchmark outputs, in the range 1-10. Default is 1.
        extra_argv : list, optional
            List with any extra arguments to pass to nosetests.

        Returns
        -------
        success : bool
            Returns True if running the benchmarks works, False if an error
            occurred.

        Notes
        -----
        Benchmarks are like tests, but have names starting with "bench" instead
        of "test", and can be found under the "benchmarks" sub-directory of the
        module.

        Each NumPy module exposes `bench` in its namespace to run all benchmarks
        for it.

        Examples
        --------
        >>> success = np.lib.bench() #doctest: +SKIP
        Running benchmarks for numpy.lib
        ...
        using 562341 items:
        unique:
        0.11
        unique1d:
        0.11
        ratio: 1.0
        nUnique: 56230 == 56230
        ...
        OK

        >>> success #doctest: +SKIP
        True

        """
        print(f'Running benchmarks for {self.package_name}')
        self._show_system_info()
        argv = self._test_argv(label, verbose, extra_argv)
        argv += ['--match', '(?:^|[\\\\b_\\\\.%s-])[Bb]ench' % os.sep]
        nose = import_nose()
        from .noseclasses import Unplugger
        add_plugins = [Unplugger('doctest')]
        return nose.run(argv=argv, addplugins=add_plugins)


def _numpy_tester():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing._private.nosetester._numpy_tester', '_numpy_tester()', {'np': np, 'NoseTester': NoseTester}, 1)

