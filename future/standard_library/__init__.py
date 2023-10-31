"""
Python 3 reorganized the standard library (PEP 3108). This module exposes
several standard library modules to Python 2 under their new Python 3
names.

It is designed to be used as follows::

    from future import standard_library
    standard_library.install_aliases()

And then these normal Py3 imports work on both Py3 and Py2::

    import builtins
    import copyreg
    import queue
    import reprlib
    import socketserver
    import winreg    # on Windows only
    import test.support
    import html, html.parser, html.entites
    import http, http.client, http.server
    import http.cookies, http.cookiejar
    import urllib.parse, urllib.request, urllib.response, urllib.error, urllib.robotparser
    import xmlrpc.client, xmlrpc.server

    import _thread
    import _dummy_thread
    import _markupbase

    from itertools import filterfalse, zip_longest
    from sys import intern
    from collections import UserDict, UserList, UserString
    from collections import OrderedDict, Counter, ChainMap     # even on Py2.6
    from subprocess import getoutput, getstatusoutput
    from subprocess import check_output              # even on Py2.6

(The renamed modules and functions are still available under their old
names on Python 2.)

This is a cleaner alternative to this idiom (see
http://docs.pythonsprints.com/python3_porting/py-porting.html)::

    try:
        import queue
    except ImportError:
        import Queue as queue


Limitations
-----------
We don't currently support these modules, but would like to::

    import dbm
    import dbm.dumb
    import dbm.gnu
    import collections.abc  # on Py33
    import pickle     # should (optionally) bring in cPickle on Python 2

"""

from __future__ import absolute_import, division, print_function
import sys
import logging
import imp
import contextlib
import types
import copy
import os
flog = logging.getLogger('future_stdlib')
_formatter = logging.Formatter(logging.BASIC_FORMAT)
_handler = logging.StreamHandler()
_handler.setFormatter(_formatter)
flog.addHandler(_handler)
flog.setLevel(logging.WARN)
from future.utils import PY2, PY3
REPLACED_MODULES = set(['test', 'urllib', 'pickle', 'dbm'])
RENAMES = {'__builtin__': 'builtins', 'copy_reg': 'copyreg', 'Queue': 'queue', 'future.moves.socketserver': 'socketserver', 'ConfigParser': 'configparser', 'repr': 'reprlib', '_winreg': 'winreg', 'thread': '_thread', 'dummy_thread': '_dummy_thread', 'future.moves.xmlrpc': 'xmlrpc', 'future.moves.html': 'html', 'future.moves.http': 'http', 'future.moves._markupbase': '_markupbase'}
assert len(set(RENAMES.values()) & set(REPLACED_MODULES)) == 0
MOVES = [('collections', 'UserList', 'UserList', 'UserList'), ('collections', 'UserDict', 'UserDict', 'UserDict'), ('collections', 'UserString', 'UserString', 'UserString'), ('collections', 'ChainMap', 'future.backports.misc', 'ChainMap'), ('itertools', 'filterfalse', 'itertools', 'ifilterfalse'), ('itertools', 'zip_longest', 'itertools', 'izip_longest'), ('sys', 'intern', '__builtin__', 'intern'), ('re', 'ASCII', 'stat', 'ST_MODE'), ('base64', 'encodebytes', 'base64', 'encodestring'), ('base64', 'decodebytes', 'base64', 'decodestring'), ('subprocess', 'getoutput', 'commands', 'getoutput'), ('subprocess', 'getstatusoutput', 'commands', 'getstatusoutput'), ('subprocess', 'check_output', 'future.backports.misc', 'check_output'), ('math', 'ceil', 'future.backports.misc', 'ceil'), ('collections', 'OrderedDict', 'future.backports.misc', 'OrderedDict'), ('collections', 'Counter', 'future.backports.misc', 'Counter'), ('collections', 'ChainMap', 'future.backports.misc', 'ChainMap'), ('itertools', 'count', 'future.backports.misc', 'count'), ('reprlib', 'recursive_repr', 'future.backports.misc', 'recursive_repr'), ('functools', 'cmp_to_key', 'future.backports.misc', 'cmp_to_key')]


class RenameImport(object):
    """
    A class for import hooks mapping Py3 module names etc. to the Py2 equivalents.
    """
    RENAMER = True
    
    def __init__(self, old_to_new):
        """
        Pass in a dictionary-like object mapping from old names to new
        names. E.g. {'ConfigParser': 'configparser', 'cPickle': 'pickle'}
        """
        self.old_to_new = old_to_new
        both = set(old_to_new.keys()) & set(old_to_new.values())
        assert (len(both) == 0 and len(set(old_to_new.values())) == len(old_to_new.values())), 'Ambiguity in renaming (handler not implemented)'
        self.new_to_old = dict(((new, old) for (old, new) in old_to_new.items()))
    
    def find_module(self, fullname, path=None):
        new_base_names = set([s.split('.')[0] for s in self.new_to_old])
        if fullname in new_base_names:
            return self
        return None
    
    def load_module(self, name):
        path = None
        if name in sys.modules:
            return sys.modules[name]
        elif name in self.new_to_old:
            oldname = self.new_to_old[name]
            module = self._find_and_load_module(oldname)
        else:
            module = self._find_and_load_module(name)
        sys.modules[name] = module
        return module
    
    def _find_and_load_module(self, name, path=None):
        """
        Finds and loads it. But if there's a . in the name, handles it
        properly.
        """
        bits = name.split('.')
        while len(bits) > 1:
            packagename = bits.pop(0)
            package = self._find_and_load_module(packagename, path)
            try:
                path = package.__path__
            except AttributeError:
                flog.debug('Package {0} has no __path__.'.format(package))
                if name in sys.modules:
                    return sys.modules[name]
                flog.debug('What to do here?')
        name = bits[0]
        module_info = imp.find_module(name, path)
        return imp.load_module(name, *module_info)



class hooks(object):
    """
    Acts as a context manager. Saves the state of sys.modules and restores it
    after the 'with' block.

    Use like this:

    >>> from future import standard_library
    >>> with standard_library.hooks():
    ...     import http.client
    >>> import requests

    For this to work, http.client will be scrubbed from sys.modules after the
    'with' block. That way the modules imported in the 'with' block will
    continue to be accessible in the current namespace but not from any
    imported modules (like requests).
    """
    
    def __enter__(self):
        self.old_sys_modules = copy.copy(sys.modules)
        self.hooks_were_installed = detect_hooks()
        install_hooks()
        return self
    
    def __exit__(self, *args):
        if not self.hooks_were_installed:
            remove_hooks()

if PY2:
    assert len(set(RENAMES.values()) & set(sys.builtin_module_names)) == 0

def is_py2_stdlib_module(m):
    """
    Tries to infer whether the module m is from the Python 2 standard library.
    This may not be reliable on all systems.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.standard_library.__init__.is_py2_stdlib_module', 'is_py2_stdlib_module(m)', {'PY3': PY3, 'is_py2_stdlib_module': is_py2_stdlib_module, 'contextlib': contextlib, 'os': os, 'copy': copy, 'flog': flog, 'sys': sys, 'm': m}, 1)

def scrub_py2_sys_modules():
    """
    Removes any Python 2 standard library modules from ``sys.modules`` that
    would interfere with Py3-style imports using import hooks. Examples are
    modules with the same names (like urllib or email).

    (Note that currently import hooks are disabled for modules like these
    with ambiguous names anyway ...)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.standard_library.__init__.scrub_py2_sys_modules', 'scrub_py2_sys_modules()', {'PY3': PY3, 'REPLACED_MODULES': REPLACED_MODULES, 'RENAMES': RENAMES, 'sys': sys, 'is_py2_stdlib_module': is_py2_stdlib_module, 'flog': flog}, 1)

def scrub_future_sys_modules():
    """
    Deprecated.
    """
    return {}


class suspend_hooks(object):
    """
    Acts as a context manager. Use like this:

    >>> from future import standard_library
    >>> standard_library.install_hooks()
    >>> import http.client
    >>> # ...
    >>> with standard_library.suspend_hooks():
    >>>     import requests     # incompatible with ``future``'s standard library hooks

    If the hooks were disabled before the context, they are not installed when
    the context is left.
    """
    
    def __enter__(self):
        self.hooks_were_installed = detect_hooks()
        remove_hooks()
        return self
    
    def __exit__(self, *args):
        if self.hooks_were_installed:
            install_hooks()


def restore_sys_modules(scrubbed):
    """
    Add any previously scrubbed modules back to the sys.modules cache,
    but only if it's safe to do so.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.standard_library.__init__.restore_sys_modules', 'restore_sys_modules(scrubbed)', {'sys': sys, 'scrubbed': scrubbed}, 0)

def install_aliases():
    """
    Monkey-patches the standard library in Py2.6/7 to provide
    aliases for better Py3 compatibility.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.standard_library.__init__.install_aliases', 'install_aliases()', {'PY3': PY3, 'MOVES': MOVES, 'sys': sys}, 1)

def install_hooks():
    """
    This function installs the future.standard_library import hook into
    sys.meta_path.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.standard_library.__init__.install_hooks', 'install_hooks()', {'PY3': PY3, 'install_aliases': install_aliases, 'flog': flog, 'sys': sys, 'RenameImport': RenameImport, 'RENAMES': RENAMES, 'detect_hooks': detect_hooks}, 1)

def enable_hooks():
    """
    Deprecated. Use install_hooks() instead. This will be removed by
    ``future`` v1.0.
    """
    install_hooks()

def remove_hooks(scrub_sys_modules=False):
    """
    This function removes the import hook from sys.meta_path.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.standard_library.__init__.remove_hooks', 'remove_hooks(scrub_sys_modules=False)', {'PY3': PY3, 'flog': flog, 'sys': sys, 'scrub_future_sys_modules': scrub_future_sys_modules, 'scrub_sys_modules': scrub_sys_modules}, 1)

def disable_hooks():
    """
    Deprecated. Use remove_hooks() instead. This will be removed by
    ``future`` v1.0.
    """
    remove_hooks()

def detect_hooks():
    """
    Returns True if the import hooks are installed, False if not.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.standard_library.__init__.detect_hooks', 'detect_hooks()', {'flog': flog, 'sys': sys}, 1)
if not hasattr(sys, 'py2_modules'):
    sys.py2_modules = {}

def cache_py2_modules():
    """
    Currently this function is unneeded, as we are not attempting to provide import hooks
    for modules with ambiguous names: email, urllib, pickle.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.standard_library.__init__.cache_py2_modules', 'cache_py2_modules()', {'sys': sys, 'detect_hooks': detect_hooks}, 1)

def import_(module_name, backport=False):
    """
    Pass a (potentially dotted) module name of a Python 3 standard library
    module. This function imports the module compatibly on Py2 and Py3 and
    returns the top-level module.

    Example use:
        >>> http = import_('http.client')
        >>> http = import_('http.server')
        >>> urllib = import_('urllib.request')

    Then:
        >>> conn = http.client.HTTPConnection(...)
        >>> response = urllib.request.urlopen('http://mywebsite.com')
        >>> # etc.

    Use as follows:
        >>> package_name = import_(module_name)

    On Py3, equivalent to this:

        >>> import module_name

    On Py2, equivalent to this if backport=False:

        >>> from future.moves import module_name

    or to this if backport=True:

        >>> from future.backports import module_name

    except that it also handles dotted module names such as ``http.client``
    The effect then is like this:

        >>> from future.backports import module
        >>> from future.backports.module import submodule
        >>> module.submodule = submodule

    Note that this would be a SyntaxError in Python:

        >>> from future.backports import http.client

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.standard_library.__init__.import_', 'import_(module_name, backport=False)', {'PY3': PY3, 'module_name': module_name, 'backport': backport}, 1)

def from_import(module_name, *symbol_names, **kwargs):
    """
    Example use:
        >>> HTTPConnection = from_import('http.client', 'HTTPConnection')
        >>> HTTPServer = from_import('http.server', 'HTTPServer')
        >>> urlopen, urlparse = from_import('urllib.request', 'urlopen', 'urlparse')

    Equivalent to this on Py3:

        >>> from module_name import symbol_names[0], symbol_names[1], ...

    and this on Py2:

        >>> from future.moves.module_name import symbol_names[0], ...

    or:

        >>> from future.backports.module_name import symbol_names[0], ...

    except that it also handles dotted module names such as ``http.client``.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.standard_library.__init__.from_import', 'from_import(module_name, *symbol_names, **kwargs)', {'PY3': PY3, 'importlib': importlib, 'module_name': module_name, 'symbol_names': symbol_names, 'kwargs': kwargs}, 1)


class exclude_local_folder_imports(object):
    """
    A context-manager that prevents standard library modules like configparser
    from being imported from the local python-future source folder on Py3.

    (This was need prior to v0.16.0 because the presence of a configparser
    folder would otherwise have prevented setuptools from running on Py3. Maybe
    it's not needed any more?)
    """
    
    def __init__(self, *args):
        assert len(args) > 0
        self.module_names = args
        if any(['.' in m for m in self.module_names]):
            raise NotImplementedError('Dotted module names are not supported')
    
    def __enter__(self):
        self.old_sys_path = copy.copy(sys.path)
        self.old_sys_modules = copy.copy(sys.modules)
        if sys.version_info[0] < 3:
            return
        FUTURE_SOURCE_SUBFOLDERS = ['future', 'past', 'libfuturize', 'libpasteurize', 'builtins']
        for folder in self.old_sys_path:
            if all([os.path.exists(os.path.join(folder, subfolder)) for subfolder in FUTURE_SOURCE_SUBFOLDERS]):
                sys.path.remove(folder)
        for m in self.module_names:
            try:
                module = __import__(m, level=0)
            except ImportError:
                pass
    
    def __exit__(self, *args):
        sys.path = self.old_sys_path
        for m in set(self.old_sys_modules.keys()) - set(sys.modules.keys()):
            sys.modules[m] = self.old_sys_modules[m]

TOP_LEVEL_MODULES = ['builtins', 'copyreg', 'html', 'http', 'queue', 'reprlib', 'socketserver', 'test', 'tkinter', 'winreg', 'xmlrpc', '_dummy_thread', '_markupbase', '_thread']

def import_top_level_modules():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.standard_library.__init__.import_top_level_modules', 'import_top_level_modules()', {'exclude_local_folder_imports': exclude_local_folder_imports, 'TOP_LEVEL_MODULES': TOP_LEVEL_MODULES}, 0)

