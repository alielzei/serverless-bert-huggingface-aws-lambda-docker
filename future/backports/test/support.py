"""Supporting definitions for the Python regression tests.

Backported for python-future from Python 3.3 test/support.py.
"""

from __future__ import absolute_import, division, print_function, unicode_literals
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
if not hasattr(unittest, 'skip'):
    import unittest2 as unittest
import importlib
import re
import subprocess
import imp
import time
try:
    import sysconfig
except ImportError:
    from distutils import sysconfig
import fnmatch
import logging.handlers
import struct
import tempfile
try:
    if utils.PY3:
        import _thread, threading
    else:
        import thread as _thread, threading
except ImportError:
    _thread = None
    threading = None
try:
    import multiprocessing.process
except ImportError:
    multiprocessing = None
try:
    import zlib
except ImportError:
    zlib = None
try:
    import gzip
except ImportError:
    gzip = None
try:
    import bz2
except ImportError:
    bz2 = None
try:
    import lzma
except ImportError:
    lzma = None
__all__ = ['Error', 'TestFailed', 'ResourceDenied', 'import_module', 'verbose', 'use_resources', 'max_memuse', 'record_original_stdout', 'get_original_stdout', 'unload', 'unlink', 'rmtree', 'forget', 'is_resource_enabled', 'requires', 'requires_freebsd_version', 'requires_linux_version', 'requires_mac_ver', 'find_unused_port', 'bind_port', 'IPV6_ENABLED', 'is_jython', 'TESTFN', 'HOST', 'SAVEDCWD', 'temp_cwd', 'findfile', 'create_empty_file', 'sortdict', 'check_syntax_error', 'open_urlresource', 'check_warnings', 'CleanImport', 'EnvironmentVarGuard', 'TransientResource', 'captured_stdout', 'captured_stdin', 'captured_stderr', 'time_out', 'socket_peer_reset', 'ioerror_peer_reset', 'run_with_locale', 'temp_umask', 'transient_internet', 'set_memlimit', 'bigmemtest', 'bigaddrspacetest', 'BasicTestRunner', 'run_unittest', 'run_doctest', 'threading_setup', 'threading_cleanup', 'reap_children', 'cpython_only', 'check_impl_detail', 'get_attribute', 'swap_item', 'swap_attr', 'requires_IEEE_754', 'TestHandler', 'Matcher', 'can_symlink', 'skip_unless_symlink', 'skip_unless_xattr', 'import_fresh_module', 'requires_zlib', 'PIPE_MAX_SIZE', 'failfast', 'anticipate_failure', 'run_with_tz', 'requires_gzip', 'requires_bz2', 'requires_lzma', 'suppress_crash_popup']


class Error(Exception):
    """Base class for regression test exceptions."""
    



class TestFailed(Error):
    """Test failed."""
    



class ResourceDenied(unittest.SkipTest):
    """Test skipped because it requested a disallowed resource.

    This is raised when a test calls requires() for a resource that
    has not be enabled.  It is used to distinguish between expected
    and unexpected skips.
    """
    


@contextlib.contextmanager
def _ignore_deprecated_imports(ignore=True):
    """Context manager to suppress package and module deprecation
    warnings when importing them.

    If ignore is False, this context manager has no effect."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support._ignore_deprecated_imports', '_ignore_deprecated_imports(ignore=True)', {'warnings': warnings, 'contextlib': contextlib, 'ignore': ignore}, 0)

def import_module(name, deprecated=False):
    """Import and return the module to be tested, raising SkipTest if
    it is not available.

    If deprecated is True, any module or package deprecation messages
    will be suppressed."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.import_module', 'import_module(name, deprecated=False)', {'_ignore_deprecated_imports': _ignore_deprecated_imports, 'importlib': importlib, 'unittest': unittest, 'name': name, 'deprecated': deprecated}, 1)

def _save_and_remove_module(name, orig_modules):
    """Helper function to save and remove a module from sys.modules

    Raise ImportError if the module can't be imported.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support._save_and_remove_module', '_save_and_remove_module(name, orig_modules)', {'sys': sys, 'name': name, 'orig_modules': orig_modules}, 0)

def _save_and_block_module(name, orig_modules):
    """Helper function to save and block a module in sys.modules

    Return True if the module was in sys.modules, False otherwise.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support._save_and_block_module', '_save_and_block_module(name, orig_modules)', {'sys': sys, 'name': name, 'orig_modules': orig_modules}, 1)

def anticipate_failure(condition):
    """Decorator to mark a test that is known to be broken in some cases

       Any use of this decorator should have a comment identifying the
       associated tracker issue.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.anticipate_failure', 'anticipate_failure(condition)', {'unittest': unittest, 'condition': condition}, 1)

def import_fresh_module(name, fresh=(), blocked=(), deprecated=False):
    """Import and return a module, deliberately bypassing sys.modules.
    This function imports and returns a fresh copy of the named Python module
    by removing the named module from sys.modules before doing the import.
    Note that unlike reload, the original module is not affected by
    this operation.

    *fresh* is an iterable of additional module names that are also removed
    from the sys.modules cache before doing the import.

    *blocked* is an iterable of module names that are replaced with None
    in the module cache during the import to ensure that attempts to import
    them raise ImportError.

    The named module and any modules named in the *fresh* and *blocked*
    parameters are saved before starting the import and then reinserted into
    sys.modules when the fresh import is complete.

    Module and package deprecation messages are suppressed during this import
    if *deprecated* is True.

    This function will raise ImportError if the named module cannot be
    imported.

    If deprecated is True, any module or package deprecation messages
    will be suppressed.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.import_fresh_module', 'import_fresh_module(name, fresh=(), blocked=(), deprecated=False)', {'_ignore_deprecated_imports': _ignore_deprecated_imports, '_save_and_remove_module': _save_and_remove_module, '_save_and_block_module': _save_and_block_module, 'importlib': importlib, 'sys': sys, 'name': name, 'fresh': fresh, 'blocked': blocked, 'deprecated': deprecated}, 1)

def get_attribute(obj, name):
    """Get an attribute, raising SkipTest if AttributeError is raised."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.get_attribute', 'get_attribute(obj, name)', {'unittest': unittest, 'obj': obj, 'name': name}, 1)
verbose = 1
use_resources = None
max_memuse = 0
real_max_memuse = 0
failfast = False
match_tests = None
_original_stdout = None

def record_original_stdout(stdout):
    global _original_stdout
    _original_stdout = stdout

def get_original_stdout():
    return (_original_stdout or sys.stdout)

def unload(name):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.unload', 'unload(name)', {'sys': sys, 'name': name}, 0)
if sys.platform.startswith('win'):
    
    def _waitfor(func, pathname, waitall=False):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('future.backports.test.support._waitfor', '_waitfor(func, pathname, waitall=False)', {'os': os, 'time': time, 'warnings': warnings, 'func': func, 'pathname': pathname, 'waitall': waitall}, 1)
    
    def _unlink(filename):
        _waitfor(os.unlink, filename)
    
    def _rmdir(dirname):
        _waitfor(os.rmdir, dirname)
    
    def _rmtree(path):
        import custom_funtemplate
        custom_funtemplate.rewrite_template('future.backports.test.support._rmtree', '_rmtree(path)', {'os': os, '_waitfor': _waitfor, 'path': path}, 0)
else:
    _unlink = os.unlink
    _rmdir = os.rmdir
    _rmtree = shutil.rmtree

def unlink(filename):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.unlink', 'unlink(filename)', {'_unlink': _unlink, 'errno': errno, 'filename': filename}, 0)

def rmdir(dirname):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.rmdir', 'rmdir(dirname)', {'_rmdir': _rmdir, 'errno': errno, 'dirname': dirname}, 0)

def rmtree(path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.rmtree', 'rmtree(path)', {'_rmtree': _rmtree, 'errno': errno, 'path': path}, 0)

def make_legacy_pyc(source):
    """Move a PEP 3147 pyc/pyo file to its legacy pyc/pyo location.

    The choice of .pyc or .pyo extension is done based on the __debug__ flag
    value.

    :param source: The file system path to the source file.  The source file
        does not need to exist, however the PEP 3147 pyc file must exist.
    :return: The file system path to the legacy pyc file.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.make_legacy_pyc', 'make_legacy_pyc(source)', {'imp': imp, 'os': os, '__debug__': __debug__, 'source': source}, 1)

def forget(modname):
    """'Forget' a module was ever imported.

    This removes the module from sys.modules and deletes any PEP 3147 or
    legacy .pyc and .pyo files.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.forget', 'forget(modname)', {'unload': unload, 'sys': sys, 'os': os, 'unlink': unlink, 'imp': imp, 'modname': modname}, 0)
if sys.platform.startswith('win'):
    import ctypes
    import ctypes.wintypes
    
    def _is_gui_available():
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('future.backports.test.support._is_gui_available', '_is_gui_available()', {'ctypes': ctypes}, 1)
else:
    
    def _is_gui_available():
        return True

def is_resource_enabled(resource):
    """Test whether a resource is enabled.  Known resources are set by
    regrtest.py."""
    return (use_resources is not None and resource in use_resources)

def requires(resource, msg=None):
    """Raise ResourceDenied if the specified resource is not available.

    If the caller's module is __main__ then automatically return True.  The
    possibility of False being returned occurs when regrtest.py is
    executing.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.requires', 'requires(resource, msg=None)', {'_is_gui_available': _is_gui_available, 'unittest': unittest, 'sys': sys, 'is_resource_enabled': is_resource_enabled, 'ResourceDenied': ResourceDenied, 'resource': resource, 'msg': msg}, 1)

def _requires_unix_version(sysname, min_version):
    """Decorator raising SkipTest if the OS is `sysname` and the version is less
    than `min_version`.

    For example, @_requires_unix_version('FreeBSD', (7, 2)) raises SkipTest if
    the FreeBSD version is less than 7.2.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support._requires_unix_version', '_requires_unix_version(sysname, min_version)', {'functools': functools, 'platform': platform, 'unittest': unittest, 'sysname': sysname, 'min_version': min_version}, 1)

def requires_freebsd_version(*min_version):
    """Decorator raising SkipTest if the OS is FreeBSD and the FreeBSD version is
    less than `min_version`.

    For example, @requires_freebsd_version(7, 2) raises SkipTest if the FreeBSD
    version is less than 7.2.
    """
    return _requires_unix_version('FreeBSD', min_version)

def requires_linux_version(*min_version):
    """Decorator raising SkipTest if the OS is Linux and the Linux version is
    less than `min_version`.

    For example, @requires_linux_version(2, 6, 32) raises SkipTest if the Linux
    version is less than 2.6.32.
    """
    return _requires_unix_version('Linux', min_version)

def requires_mac_ver(*min_version):
    """Decorator raising SkipTest if the OS is Mac OS X and the OS X
    version if less than min_version.

    For example, @requires_mac_ver(10, 5) raises SkipTest if the OS X version
    is lesser than 10.5.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.requires_mac_ver', 'requires_mac_ver(*min_version)', {'functools': functools, 'sys': sys, 'platform': platform, 'unittest': unittest, 'min_version': min_version}, 1)
HOST = '127.0.0.1'
HOSTv6 = '::1'

def find_unused_port(family=socket.AF_INET, socktype=socket.SOCK_STREAM):
    """Returns an unused port that should be suitable for binding.  This is
    achieved by creating a temporary socket with the same family and type as
    the 'sock' parameter (default is AF_INET, SOCK_STREAM), and binding it to
    the specified host address (defaults to 0.0.0.0) with the port set to 0,
    eliciting an unused ephemeral port from the OS.  The temporary socket is
    then closed and deleted, and the ephemeral port is returned.

    Either this method or bind_port() should be used for any tests where a
    server socket needs to be bound to a particular port for the duration of
    the test.  Which one to use depends on whether the calling code is creating
    a python socket, or if an unused port needs to be provided in a constructor
    or passed to an external program (i.e. the -accept argument to openssl's
    s_server mode).  Always prefer bind_port() over find_unused_port() where
    possible.  Hard coded ports should *NEVER* be used.  As soon as a server
    socket is bound to a hard coded port, the ability to run multiple instances
    of the test simultaneously on the same host is compromised, which makes the
    test a ticking time bomb in a buildbot environment. On Unix buildbots, this
    may simply manifest as a failed test, which can be recovered from without
    intervention in most cases, but on Windows, the entire python process can
    completely and utterly wedge, requiring someone to log in to the buildbot
    and manually kill the affected process.

    (This is easy to reproduce on Windows, unfortunately, and can be traced to
    the SO_REUSEADDR socket option having different semantics on Windows versus
    Unix/Linux.  On Unix, you can't have two AF_INET SOCK_STREAM sockets bind,
    listen and then accept connections on identical host/ports.  An EADDRINUSE
    socket.error will be raised at some point (depending on the platform and
    the order bind and listen were called on each socket).

    However, on Windows, if SO_REUSEADDR is set on the sockets, no EADDRINUSE
    will ever be raised when attempting to bind two identical host/ports. When
    accept() is called on each socket, the second caller's process will steal
    the port from the first caller, leaving them both in an awkwardly wedged
    state where they'll no longer respond to any signals or graceful kills, and
    must be forcibly killed via OpenProcess()/TerminateProcess().

    The solution on Windows is to use the SO_EXCLUSIVEADDRUSE socket option
    instead of SO_REUSEADDR, which effectively affords the same semantics as
    SO_REUSEADDR on Unix.  Given the propensity of Unix developers in the Open
    Source world compared to Windows ones, this is a common mistake.  A quick
    look over OpenSSL's 0.9.8g source shows that they use SO_REUSEADDR when
    openssl.exe is called with the 's_server' option, for example. See
    http://bugs.python.org/issue2550 for more info.  The following site also
    has a very thorough description about the implications of both REUSEADDR
    and EXCLUSIVEADDRUSE on Windows:
    http://msdn2.microsoft.com/en-us/library/ms740621(VS.85).aspx)

    XXX: although this approach is a vast improvement on previous attempts to
    elicit unused ports, it rests heavily on the assumption that the ephemeral
    port returned to us by the OS won't immediately be dished back out to some
    other process when we close and delete our temporary socket but before our
    calling code has a chance to bind the returned port.  We can deal with this
    issue if/when we come across it.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.find_unused_port', 'find_unused_port(family=socket.AF_INET, socktype=socket.SOCK_STREAM)', {'socket': socket, 'bind_port': bind_port, 'family': family, 'socktype': socktype}, 1)

def bind_port(sock, host=HOST):
    """Bind the socket to a free port and return the port number.  Relies on
    ephemeral ports in order to ensure we are using an unbound port.  This is
    important as many tests may be running simultaneously, especially in a
    buildbot environment.  This method raises an exception if the sock.family
    is AF_INET and sock.type is SOCK_STREAM, *and* the socket has SO_REUSEADDR
    or SO_REUSEPORT set on it.  Tests should *never* set these socket options
    for TCP/IP sockets.  The only case for setting these options is testing
    multicasting via multiple UDP sockets.

    Additionally, if the SO_EXCLUSIVEADDRUSE socket option is available (i.e.
    on Windows), it will be set on the socket.  This will prevent anyone else
    from bind()'ing to our host/port for the duration of the test.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.bind_port', 'bind_port(sock, host=HOST)', {'socket': socket, 'TestFailed': TestFailed, 'sock': sock, 'host': host, 'HOST': HOST}, 1)

def _is_ipv6_enabled():
    """Check whether IPv6 is enabled on this host."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support._is_ipv6_enabled', '_is_ipv6_enabled()', {'socket': socket}, 1)
IPV6_ENABLED = _is_ipv6_enabled()
PIPE_MAX_SIZE = 4 * 1024 * 1024 + 1
SOCK_MAX_SIZE = 16 * 1024 * 1024 + 1
requires_zlib = unittest.skipUnless(zlib, 'requires zlib')
requires_bz2 = unittest.skipUnless(bz2, 'requires bz2')
requires_lzma = unittest.skipUnless(lzma, 'requires lzma')
is_jython = sys.platform.startswith('java')
if os.name == 'java':
    TESTFN = '$test'
else:
    TESTFN = '@test'
TESTFN = '{0}_{1}_tmp'.format(TESTFN, os.getpid())
SAVEDCWD = os.getcwd()

@contextlib.contextmanager
def temp_cwd(name='tempcwd', quiet=False, path=None):
    """
    Context manager that temporarily changes the CWD.

    An existing path may be provided as *path*, in which case this
    function makes no changes to the file system.

    Otherwise, the new CWD is created in the current directory and it's
    named *name*. If *quiet* is False (default) and it's not possible to
    create or change the CWD, an error is raised.  If it's True, only a
    warning is raised and the original CWD is used.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.temp_cwd', "temp_cwd(name='tempcwd', quiet=False, path=None)", {'os': os, 'warnings': warnings, 'rmtree': rmtree, 'contextlib': contextlib, 'name': name, 'quiet': quiet, 'path': path}, 0)
if hasattr(os, 'umask'):
    
    @contextlib.contextmanager
    def temp_umask(umask):
        """Context manager that temporarily sets the process umask."""
        import custom_funtemplate
        custom_funtemplate.rewrite_template('future.backports.test.support.temp_umask', 'temp_umask(umask)', {'os': os, 'contextlib': contextlib, 'umask': umask}, 0)

def findfile(file, here=__file__, subdir=None):
    """Try to find a file on sys.path and the working directory.  If it is not
    found the argument passed to the function is returned (this does not
    necessarily signal failure; could still be the legitimate path)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.findfile', 'findfile(file, here=__file__, subdir=None)', {'os': os, 'sys': sys, 'file': file, 'here': here, 'subdir': subdir, '__file__': __file__}, 1)

def create_empty_file(filename):
    """Create an empty file. If the file already exists, truncate it."""
    fd = os.open(filename, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.close(fd)

def sortdict(dict):
    """Like repr(dict), but in sorted order."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.sortdict', 'sortdict(dict)', {'dict': dict}, 1)

def make_bad_fd():
    """
    Create an invalid file descriptor by opening and closing a file and return
    its fd.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.make_bad_fd', 'make_bad_fd()', {'TESTFN': TESTFN, 'unlink': unlink}, 1)

def check_syntax_error(testcase, statement):
    testcase.assertRaises(SyntaxError, compile, statement, '<test string>', 'exec')

def open_urlresource(url, *args, **kw):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.open_urlresource', 'open_urlresource(url, *args, **kw)', {'os': os, '__file__': __file__, 'unlink': unlink, 'requires': requires, 'get_original_stdout': get_original_stdout, 'TestFailed': TestFailed, 'url': url, 'args': args, 'kw': kw}, 1)


class WarningsRecorder(object):
    """Convenience wrapper for the warnings list returned on
       entry to the warnings.catch_warnings() context manager.
    """
    
    def __init__(self, warnings_list):
        self._warnings = warnings_list
        self._last = 0
    
    def __getattr__(self, attr):
        if len(self._warnings) > self._last:
            return getattr(self._warnings[-1], attr)
        elif attr in warnings.WarningMessage._WARNING_DETAILS:
            return None
        raise AttributeError('%r has no attribute %r' % (self, attr))
    
    @property
    def warnings(self):
        return self._warnings[self._last:]
    
    def reset(self):
        self._last = len(self._warnings)


def _filterwarnings(filters, quiet=False):
    """Catch the warnings, then check if all the expected
    warnings have been raised and re-raise unexpected warnings.
    If 'quiet' is True, only re-raise the unexpected warnings.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support._filterwarnings', '_filterwarnings(filters, quiet=False)', {'sys': sys, 'utils': utils, 'warnings': warnings, 'WarningsRecorder': WarningsRecorder, 're': re, 'filters': filters, 'quiet': quiet}, 0)

@contextlib.contextmanager
def check_warnings(*filters, **kwargs):
    """Context manager to silence warnings.

    Accept 2-tuples as positional arguments:
        ("message regexp", WarningCategory)

    Optional argument:
     - if 'quiet' is True, it does not fail if a filter catches nothing
        (default True without argument,
         default False if some filters are defined)

    Without argument, it defaults to:
        check_warnings(("", Warning), quiet=True)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.check_warnings', 'check_warnings(*filters, **kwargs)', {'_filterwarnings': _filterwarnings, 'contextlib': contextlib, 'filters': filters, 'kwargs': kwargs}, 1)


class CleanImport(object):
    """Context manager to force import to return a new module reference.

    This is useful for testing module-level behaviours, such as
    the emission of a DeprecationWarning on import.

    Use like this:

        with CleanImport("foo"):
            importlib.import_module("foo") # new reference
    """
    
    def __init__(self, *module_names):
        self.original_modules = sys.modules.copy()
        for module_name in module_names:
            if module_name in sys.modules:
                module = sys.modules[module_name]
                if module.__name__ != module_name:
                    del sys.modules[module.__name__]
                del sys.modules[module_name]
    
    def __enter__(self):
        return self
    
    def __exit__(self, *ignore_exc):
        sys.modules.update(self.original_modules)

if utils.PY3:
    import collections.abc
    mybase = collections.abc.MutableMapping
else:
    import UserDict
    mybase = UserDict.DictMixin


class EnvironmentVarGuard(mybase):
    """Class to help protect the environment variable properly.  Can be used as
    a context manager."""
    
    def __init__(self):
        self._environ = os.environ
        self._changed = {}
    
    def __getitem__(self, envvar):
        return self._environ[envvar]
    
    def __setitem__(self, envvar, value):
        if envvar not in self._changed:
            self._changed[envvar] = self._environ.get(envvar)
        self._environ[envvar] = value
    
    def __delitem__(self, envvar):
        if envvar not in self._changed:
            self._changed[envvar] = self._environ.get(envvar)
        if envvar in self._environ:
            del self._environ[envvar]
    
    def keys(self):
        return self._environ.keys()
    
    def __iter__(self):
        return iter(self._environ)
    
    def __len__(self):
        return len(self._environ)
    
    def set(self, envvar, value):
        self[envvar] = value
    
    def unset(self, envvar):
        del self[envvar]
    
    def __enter__(self):
        return self
    
    def __exit__(self, *ignore_exc):
        for (k, v) in self._changed.items():
            if v is None:
                if k in self._environ:
                    del self._environ[k]
            else:
                self._environ[k] = v
        os.environ = self._environ



class DirsOnSysPath(object):
    """Context manager to temporarily add directories to sys.path.

    This makes a copy of sys.path, appends any directories given
    as positional arguments, then reverts sys.path to the copied
    settings when the context ends.

    Note that *all* sys.path modifications in the body of the
    context manager, including replacement of the object,
    will be reverted at the end of the block.
    """
    
    def __init__(self, *paths):
        self.original_value = sys.path[:]
        self.original_object = sys.path
        sys.path.extend(paths)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *ignore_exc):
        sys.path = self.original_object
        sys.path[:] = self.original_value



class TransientResource(object):
    """Raise ResourceDenied if an exception is raised while the context manager
    is in effect that matches the specified exception and attributes."""
    
    def __init__(self, exc, **kwargs):
        self.exc = exc
        self.attrs = kwargs
    
    def __enter__(self):
        return self
    
    def __exit__(self, type_=None, value=None, traceback=None):
        """If type_ is a subclass of self.exc and value has attributes matching
        self.attrs, raise ResourceDenied.  Otherwise let the exception
        propagate (if any)."""
        if (type_ is not None and issubclass(self.exc, type_)):
            for (attr, attr_value) in self.attrs.items():
                if not hasattr(value, attr):
                    break
                if getattr(value, attr) != attr_value:
                    break
            else:
                raise ResourceDenied('an optional resource is not available')

time_out = TransientResource(IOError, errno=errno.ETIMEDOUT)
socket_peer_reset = TransientResource(socket.error, errno=errno.ECONNRESET)
ioerror_peer_reset = TransientResource(IOError, errno=errno.ECONNRESET)

@contextlib.contextmanager
def transient_internet(resource_name, timeout=30.0, errnos=()):
    """Return a context manager that raises ResourceDenied when various issues
    with the Internet connection manifest themselves as exceptions."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.transient_internet', 'transient_internet(resource_name, timeout=30.0, errnos=())', {'ResourceDenied': ResourceDenied, 'errno': errno, 'socket': socket, 'verbose': verbose, 'sys': sys, 'IOError': IOError, 'contextlib': contextlib, 'resource_name': resource_name, 'timeout': timeout, 'errnos': errnos}, 0)

@contextlib.contextmanager
def captured_output(stream_name):
    """Return a context manager used by captured_stdout/stdin/stderr
    that temporarily replaces the sys stream *stream_name* with a StringIO."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.captured_output', 'captured_output(stream_name)', {'sys': sys, 'contextlib': contextlib, 'stream_name': stream_name}, 0)

def captured_stdout():
    """Capture the output of sys.stdout:

       with captured_stdout() as s:
           print("hello")
       self.assertEqual(s.getvalue(), "hello")
    """
    return captured_output('stdout')

def captured_stderr():
    return captured_output('stderr')

def captured_stdin():
    return captured_output('stdin')

def gc_collect():
    """Force as many objects as possible to be collected.

    In non-CPython implementations of Python, this is needed because timely
    deallocation is not guaranteed by the garbage collector.  (Even in CPython
    this can be the case in case of reference cycles.)  This means that __del__
    methods may be called later than expected and weakrefs may remain alive for
    longer than expected.  This function tries its best to force all garbage
    objects to disappear.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.gc_collect', 'gc_collect()', {'gc': gc, 'is_jython': is_jython, 'time': time}, 0)

@contextlib.contextmanager
def disable_gc():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.disable_gc', 'disable_gc()', {'gc': gc, 'contextlib': contextlib}, 0)

def python_is_optimized():
    """Find if Python was built with optimizations."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.python_is_optimized', 'python_is_optimized()', {}, 1)
_header = 'nP'
_align = '0n'
if hasattr(sys, 'gettotalrefcount'):
    _header = '2P' + _header
    _align = '0P'
_vheader = _header + 'n'

def calcobjsize(fmt):
    return struct.calcsize(_header + fmt + _align)

def calcvobjsize(fmt):
    return struct.calcsize(_vheader + fmt + _align)
_TPFLAGS_HAVE_GC = 1 << 14
_TPFLAGS_HEAPTYPE = 1 << 9

def check_sizeof(test, o, size):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.check_sizeof', 'check_sizeof(test, o, size)', {'sys': sys, '_TPFLAGS_HEAPTYPE': _TPFLAGS_HEAPTYPE, '_TPFLAGS_HAVE_GC': _TPFLAGS_HAVE_GC, '_testcapi': _testcapi, 'test': test, 'o': o, 'size': size}, 0)

def run_with_locale(catstr, *locales):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.run_with_locale', 'run_with_locale(catstr, *locales)', {'catstr': catstr, 'locales': locales}, 1)

def run_with_tz(tz):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.run_with_tz', 'run_with_tz(tz)', {'time': time, 'unittest': unittest, 'os': os, 'tz': tz}, 1)
_1M = 1024 * 1024
_1G = 1024 * _1M
_2G = 2 * _1G
_4G = 4 * _1G
MAX_Py_ssize_t = sys.maxsize

def set_memlimit(limit):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.set_memlimit', 'set_memlimit(limit)', {'_1M': _1M, '_1G': _1G, 're': re, 'MAX_Py_ssize_t': MAX_Py_ssize_t, '_2G': _2G, 'limit': limit}, 0)


class _MemoryWatchdog(object):
    """An object which periodically watches the process' memory consumption
    and prints it out.
    """
    
    def __init__(self):
        self.procfile = '/proc/{pid}/statm'.format(pid=os.getpid())
        self.started = False
    
    def start(self):
        try:
            f = open(self.procfile, 'r')
        except OSError as e:
            warnings.warn('/proc not available for stats: {0}'.format(e), RuntimeWarning)
            sys.stderr.flush()
            return
        watchdog_script = findfile('memory_watchdog.py')
        self.mem_watchdog = subprocess.Popen([sys.executable, watchdog_script], stdin=f, stderr=subprocess.DEVNULL)
        f.close()
        self.started = True
    
    def stop(self):
        if self.started:
            self.mem_watchdog.terminate()
            self.mem_watchdog.wait()


def bigmemtest(size, memuse, dry_run=True):
    """Decorator for bigmem tests.

    'minsize' is the minimum useful size for the test (in arbitrary,
    test-interpreted units.) 'memuse' is the number of 'bytes per size' for
    the test, or a good estimate of it.

    if 'dry_run' is False, it means the test doesn't support dummy runs
    when -M is not specified.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.bigmemtest', 'bigmemtest(size, memuse, dry_run=True)', {'real_max_memuse': real_max_memuse, 'unittest': unittest, 'verbose': verbose, '_MemoryWatchdog': _MemoryWatchdog, 'size': size, 'memuse': memuse, 'dry_run': dry_run}, 1)

def bigaddrspacetest(f):
    """Decorator for tests that fill the address space."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.bigaddrspacetest', 'bigaddrspacetest(f)', {'max_memuse': max_memuse, 'MAX_Py_ssize_t': MAX_Py_ssize_t, 'unittest': unittest, 'f': f}, 1)


class BasicTestRunner(object):
    
    def run(self, test):
        result = unittest.TestResult()
        test(result)
        return result


def _id(obj):
    return obj

def requires_resource(resource):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.requires_resource', 'requires_resource(resource)', {'_is_gui_available': _is_gui_available, 'unittest': unittest, 'is_resource_enabled': is_resource_enabled, '_id': _id, 'resource': resource}, 1)

def cpython_only(test):
    """
    Decorator for tests only applicable on CPython.
    """
    return impl_detail(cpython=True)(test)

def impl_detail(msg=None, **guards):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.impl_detail', 'impl_detail(msg=None, **guards)', {'check_impl_detail': check_impl_detail, '_id': _id, '_parse_guards': _parse_guards, 'unittest': unittest, 'msg': msg, 'guards': guards}, 1)

def _parse_guards(guards):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support._parse_guards', '_parse_guards(guards)', {'guards': guards}, 2)

def check_impl_detail(**guards):
    """This function returns True or False depending on the host platform.
       Examples:
          if check_impl_detail():               # only on CPython (default)
          if check_impl_detail(jython=True):    # only on Jython
          if check_impl_detail(cpython=False):  # everywhere except on CPython
    """
    (guards, default) = _parse_guards(guards)
    return guards.get(platform.python_implementation().lower(), default)

def no_tracing(func):
    """Decorator to temporarily turn off tracing for the duration of a test."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.no_tracing', 'no_tracing(func)', {'sys': sys, 'functools': functools, 'func': func}, 1)

def refcount_test(test):
    """Decorator for tests which involve reference counting.

    To start, the decorator does not run the test if is not run by CPython.
    After that, any trace function is unset during the test to prevent
    unexpected refcounts caused by the trace function.

    """
    return no_tracing(cpython_only(test))

def _filter_suite(suite, pred):
    """Recursively filter test cases in a suite based on a predicate."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support._filter_suite', '_filter_suite(suite, pred)', {'unittest': unittest, '_filter_suite': _filter_suite, 'suite': suite, 'pred': pred}, 0)

def _run_suite(suite):
    """Run tests from a unittest.TestSuite-derived class."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support._run_suite', '_run_suite(suite)', {'verbose': verbose, 'unittest': unittest, 'sys': sys, 'failfast': failfast, 'BasicTestRunner': BasicTestRunner, 'TestFailed': TestFailed, 'suite': suite}, 0)

def run_unittest(*classes):
    """Run tests from unittest.TestCase-derived classes."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.run_unittest', 'run_unittest(*classes)', {'unittest': unittest, 'sys': sys, 'match_tests': match_tests, 'fnmatch': fnmatch, '_filter_suite': _filter_suite, '_run_suite': _run_suite, 'classes': classes}, 1)

def run_doctest(module, verbosity=None, optionflags=0):
    """Run doctest on the given module.  Return (#failures, #tests).

    If optional argument verbosity is not specified (or is None), pass
    support's belief about verbosity on to doctest.  Else doctest's
    usual behavior is used (it searches sys.argv for -v).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.run_doctest', 'run_doctest(module, verbosity=None, optionflags=0)', {'verbose': verbose, 'TestFailed': TestFailed, 'module': module, 'verbosity': verbosity, 'optionflags': optionflags}, 2)

def modules_setup():
    return (sys.modules.copy(), )

def modules_cleanup(oldmodules):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.modules_cleanup', 'modules_cleanup(oldmodules)', {'sys': sys, 'oldmodules': oldmodules}, 0)

def threading_setup():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.threading_setup', 'threading_setup()', {'_thread': _thread}, 1)

def threading_cleanup(nb_threads):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.threading_cleanup', 'threading_cleanup(nb_threads)', {'_thread': _thread, 'time': time, 'nb_threads': nb_threads}, 1)

def reap_threads(func):
    """Use this function when threads are being used.  This will
    ensure that the threads are cleaned up even when the test fails.
    If threading is unavailable this function does nothing.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.reap_threads', 'reap_threads(func)', {'_thread': _thread, 'functools': functools, 'threading_setup': threading_setup, 'threading_cleanup': threading_cleanup, 'func': func}, 1)

def reap_children():
    """Use this function at the end of test_main() whenever sub-processes
    are started.  This will help ensure that no extra children (zombies)
    stick around to hog resources and create problems when looking
    for refleaks.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.reap_children', 'reap_children()', {'os': os}, 0)

@contextlib.contextmanager
def swap_attr(obj, attr, new_val):
    """Temporary swap out an attribute with a new object.

    Usage:
        with swap_attr(obj, "attr", 5):
            ...

        This will set obj.attr to 5 for the duration of the with: block,
        restoring the old value at the end of the block. If `attr` doesn't
        exist on `obj`, it will be created and then deleted at the end of the
        block.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.swap_attr', 'swap_attr(obj, attr, new_val)', {'contextlib': contextlib, 'obj': obj, 'attr': attr, 'new_val': new_val}, 0)

@contextlib.contextmanager
def swap_item(obj, item, new_val):
    """Temporary swap out an item with a new object.

    Usage:
        with swap_item(obj, "item", 5):
            ...

        This will set obj["item"] to 5 for the duration of the with: block,
        restoring the old value at the end of the block. If `item` doesn't
        exist on `obj`, it will be created and then deleted at the end of the
        block.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.swap_item', 'swap_item(obj, item, new_val)', {'contextlib': contextlib, 'obj': obj, 'item': item, 'new_val': new_val}, 0)

def strip_python_stderr(stderr):
    """Strip the stderr of a Python process from potential debug output
    emitted by the interpreter.

    This will typically be run on the result of the communicate() method
    of a subprocess.Popen object.
    """
    stderr = re.sub(b'\\[\\d+ refs\\]\\r?\\n?', b'', stderr).strip()
    return stderr

def args_from_interpreter_flags():
    """Return a list of command-line arguments reproducing the current
    settings in sys.flags and sys.warnoptions."""
    return subprocess._args_from_interpreter_flags()


class TestHandler(logging.handlers.BufferingHandler):
    
    def __init__(self, matcher):
        logging.handlers.BufferingHandler.__init__(self, 0)
        self.matcher = matcher
    
    def shouldFlush(self):
        return False
    
    def emit(self, record):
        self.format(record)
        self.buffer.append(record.__dict__)
    
    def matches(self, **kwargs):
        """
        Look for a saved dict whose keys/values match the supplied arguments.
        """
        result = False
        for d in self.buffer:
            if self.matcher.matches(d, **kwargs):
                result = True
                break
        return result



class Matcher(object):
    _partial_matches = ('msg', 'message')
    
    def matches(self, d, **kwargs):
        """
        Try to match a single dict with the supplied arguments.

        Keys whose values are strings and which are in self._partial_matches
        will be checked for partial (i.e. substring) matches. You can extend
        this scheme to (for example) do regular expression matching, etc.
        """
        result = True
        for k in kwargs:
            v = kwargs[k]
            dv = d.get(k)
            if not self.match_value(k, dv, v):
                result = False
                break
        return result
    
    def match_value(self, k, dv, v):
        """
        Try to match a single stored value (dv) with a supplied value (v).
        """
        if type(v) != type(dv):
            result = False
        elif (type(dv) is not str or k not in self._partial_matches):
            result = v == dv
        else:
            result = dv.find(v) >= 0
        return result

_can_symlink = None

def can_symlink():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.can_symlink', 'can_symlink()', {'TESTFN': TESTFN, 'os': os}, 1)

def skip_unless_symlink(test):
    """Skip decorator for tests that require functional symlink"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.skip_unless_symlink', 'skip_unless_symlink(test)', {'can_symlink': can_symlink, 'unittest': unittest, 'test': test}, 1)
_can_xattr = None

def can_xattr():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.can_xattr', 'can_xattr()', {'os': os, 'tempfile': tempfile, 'TESTFN': TESTFN, 'platform': platform, 're': re, 'unlink': unlink}, 1)

def skip_unless_xattr(test):
    """Skip decorator for tests that require functional extended attributes"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.test.support.skip_unless_xattr', 'skip_unless_xattr(test)', {'can_xattr': can_xattr, 'unittest': unittest, 'test': test}, 1)
if sys.platform.startswith('win'):
    
    @contextlib.contextmanager
    def suppress_crash_popup():
        """Disable Windows Error Reporting dialogs using SetErrorMode."""
        import custom_funtemplate
        custom_funtemplate.rewrite_template('future.backports.test.support.suppress_crash_popup', 'suppress_crash_popup()', {'contextlib': contextlib}, 0)
else:
    
    @contextlib.contextmanager
    def suppress_crash_popup():
        yield

def patch(test_instance, object_to_patch, attr_name, new_value):
    """Override 'object_to_patch'.'attr_name' with 'new_value'.

    Also, add a cleanup procedure to 'test_instance' to restore
    'object_to_patch' value for 'attr_name'.
    The 'attr_name' should be a valid attribute for 'object_to_patch'.

    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.test.support.patch', 'patch(test_instance, object_to_patch, attr_name, new_value)', {'test_instance': test_instance, 'object_to_patch': object_to_patch, 'attr_name': attr_name, 'new_value': new_value}, 0)

