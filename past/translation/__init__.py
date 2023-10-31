"""
past.translation
==================

The ``past.translation`` package provides an import hook for Python 3 which
transparently runs ``futurize`` fixers over Python 2 code on import to convert
print statements into functions, etc.

It is intended to assist users in migrating to Python 3.x even if some
dependencies still only support Python 2.x.

Usage
-----

Once your Py2 package is installed in the usual module search path, the import
hook is invoked as follows:

    >>> from past.translation import autotranslate
    >>> autotranslate('mypackagename')

Or:

    >>> autotranslate(['mypackage1', 'mypackage2'])

You can unregister the hook using::

    >>> from past.translation import remove_hooks
    >>> remove_hooks()

Author: Ed Schofield.
Inspired by and based on ``uprefix`` by Vinay M. Sajip.
"""

import imp
import logging
import marshal
import os
import sys
import copy
from lib2to3.pgen2.parse import ParseError
from lib2to3.refactor import RefactoringTool
from libfuturize import fixes
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
myfixes = list(fixes.libfuturize_fix_names_stage1) + list(fixes.lib2to3_fix_names_stage1) + list(fixes.libfuturize_fix_names_stage2) + list(fixes.lib2to3_fix_names_stage2)
py2_detect_fixers = ['lib2to3.fixes.fix_apply', 'lib2to3.fixes.fix_except', 'lib2to3.fixes.fix_execfile', 'lib2to3.fixes.fix_exitfunc', 'lib2to3.fixes.fix_funcattrs', 'lib2to3.fixes.fix_filter', 'lib2to3.fixes.fix_has_key', 'lib2to3.fixes.fix_idioms', 'lib2to3.fixes.fix_import', 'lib2to3.fixes.fix_intern', 'lib2to3.fixes.fix_isinstance', 'lib2to3.fixes.fix_methodattrs', 'lib2to3.fixes.fix_ne', 'lib2to3.fixes.fix_numliterals', 'lib2to3.fixes.fix_paren', 'lib2to3.fixes.fix_print', 'lib2to3.fixes.fix_raise', 'lib2to3.fixes.fix_renames', 'lib2to3.fixes.fix_reduce', 'lib2to3.fixes.fix_repr', 'lib2to3.fixes.fix_standarderror', 'lib2to3.fixes.fix_sys_exc', 'lib2to3.fixes.fix_throw', 'lib2to3.fixes.fix_tuple_params', 'lib2to3.fixes.fix_types', 'lib2to3.fixes.fix_ws_comma', 'lib2to3.fixes.fix_xreadlines', 'lib2to3.fixes.fix_basestring', 'lib2to3.fixes.fix_exec', 'lib2to3.fixes.fix_getcwdu', 'lib2to3.fixes.fix_long', 'lib2to3.fixes.fix_next', 'lib2to3.fixes.fix_nonzero', 'lib2to3.fixes.fix_raw_input', 'lib2to3.fixes.fix_xrange']


class RTs:
    """
    A namespace for the refactoring tools. This avoids creating these at
    the module level, which slows down the module import. (See issue #117).

    There are two possible grammars: with or without the print statement.
    Hence we have two possible refactoring tool implementations.
    """
    _rt = None
    _rtp = None
    _rt_py2_detect = None
    _rtp_py2_detect = None
    
    @staticmethod
    def setup():
        """
        Call this before using the refactoring tools to create them on demand
        if needed.
        """
        if None in [RTs._rt, RTs._rtp]:
            RTs._rt = RefactoringTool(myfixes)
            RTs._rtp = RefactoringTool(myfixes, {'print_function': True})
    
    @staticmethod
    def setup_detect_python2():
        """
        Call this before using the refactoring tools to create them on demand
        if needed.
        """
        if None in [RTs._rt_py2_detect, RTs._rtp_py2_detect]:
            RTs._rt_py2_detect = RefactoringTool(py2_detect_fixers)
            RTs._rtp_py2_detect = RefactoringTool(py2_detect_fixers, {'print_function': True})


def splitall(path):
    """
    Split a path into all components. From Python Cookbook.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('past.translation.__init__.splitall', 'splitall(path)', {'os': os, 'path': path}, 1)

def common_substring(s1, s2):
    """
    Returns the longest common substring to the two strings, starting from the
    left.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('past.translation.__init__.common_substring', 'common_substring(s1, s2)', {'splitall': splitall, 'os': os, 's1': s1, 's2': s2}, 1)

def detect_python2(source, pathname):
    """
    Returns a bool indicating whether we think the code is Py2
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('past.translation.__init__.detect_python2', 'detect_python2(source, pathname)', {'RTs': RTs, 'ParseError': ParseError, 'logger': logger, 'source': source, 'pathname': pathname}, 1)


class Py2Fixer(object):
    """
    An import hook class that uses lib2to3 for source-to-source translation of
    Py2 code to Py3.
    """
    PY2FIXER = True
    
    def __init__(self):
        self.found = None
        self.base_exclude_paths = ['future', 'past']
        self.exclude_paths = copy.copy(self.base_exclude_paths)
        self.include_paths = []
    
    def include(self, paths):
        """
        Pass in a sequence of module names such as 'plotrique.plotting' that,
        if present at the leftmost side of the full package name, would
        specify the module to be transformed from Py2 to Py3.
        """
        self.include_paths += paths
    
    def exclude(self, paths):
        """
        Pass in a sequence of strings such as 'mymodule' that, if
        present at the leftmost side of the full package name, would cause
        the module not to undergo any source transformation.
        """
        self.exclude_paths += paths
    
    def find_module(self, fullname, path=None):
        logger.debug('Running find_module: {0}...'.format(fullname))
        if '.' in fullname:
            (parent, child) = fullname.rsplit('.', 1)
            if path is None:
                loader = self.find_module(parent, path)
                mod = loader.load_module(parent)
                path = mod.__path__
            fullname = child
        try:
            self.found = imp.find_module(fullname, path)
        except Exception as e:
            logger.debug('Py2Fixer could not find {0}')
            logger.debug('Exception was: {0})'.format(fullname, e))
            return None
        self.kind = self.found[-1][-1]
        if self.kind == imp.PKG_DIRECTORY:
            self.pathname = os.path.join(self.found[1], '__init__.py')
        elif self.kind == imp.PY_SOURCE:
            self.pathname = self.found[1]
        return self
    
    def transform(self, source):
        RTs.setup()
        source += '\n'
        try:
            tree = RTs._rt.refactor_string(source, self.pathname)
        except ParseError as e:
            if (e.msg != 'bad input' or e.value != '='):
                raise
            tree = RTs._rtp.refactor_string(source, self.pathname)
        return str(tree)[:-1]
    
    def load_module(self, fullname):
        logger.debug('Running load_module for {0}...'.format(fullname))
        if fullname in sys.modules:
            mod = sys.modules[fullname]
        else:
            if self.kind in (imp.PY_COMPILED, imp.C_EXTENSION, imp.C_BUILTIN, imp.PY_FROZEN):
                convert = False
            elif any([fullname.startswith(path) for path in self.exclude_paths]):
                convert = False
            elif any([fullname.startswith(path) for path in self.include_paths]):
                convert = True
            else:
                convert = False
            if not convert:
                logger.debug('Excluded {0} from translation'.format(fullname))
                mod = imp.load_module(fullname, *self.found)
            else:
                logger.debug('Autoconverting {0} ...'.format(fullname))
                mod = imp.new_module(fullname)
                sys.modules[fullname] = mod
                mod.__file__ = self.pathname
                mod.__name__ = fullname
                mod.__loader__ = self
                if self.kind == imp.PKG_DIRECTORY:
                    mod.__path__ = [os.path.dirname(self.pathname)]
                    mod.__package__ = fullname
                else:
                    mod.__path__ = []
                    mod.__package__ = fullname.rpartition('.')[0]
                try:
                    cachename = imp.cache_from_source(self.pathname)
                    if not os.path.exists(cachename):
                        update_cache = True
                    else:
                        sourcetime = os.stat(self.pathname).st_mtime
                        cachetime = os.stat(cachename).st_mtime
                        update_cache = cachetime < sourcetime
                    if not update_cache:
                        with open(cachename, 'rb') as f:
                            data = f.read()
                            try:
                                code = marshal.loads(data)
                            except Exception:
                                update_cache = True
                    if update_cache:
                        if self.found[0]:
                            source = self.found[0].read()
                        elif self.kind == imp.PKG_DIRECTORY:
                            with open(self.pathname) as f:
                                source = f.read()
                        if detect_python2(source, self.pathname):
                            source = self.transform(source)
                        code = compile(source, self.pathname, 'exec')
                        dirname = os.path.dirname(cachename)
                        try:
                            if not os.path.exists(dirname):
                                os.makedirs(dirname)
                            with open(cachename, 'wb') as f:
                                data = marshal.dumps(code)
                                f.write(data)
                        except Exception:
                            pass
                    exec(code, mod.__dict__)
                except Exception as e:
                    del sys.modules[fullname]
                    raise
        if self.found[0]:
            self.found[0].close()
        return mod

_hook = Py2Fixer()

def install_hooks(include_paths=(), exclude_paths=()):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('past.translation.__init__.install_hooks', 'install_hooks(include_paths=(), exclude_paths=())', {'_hook': _hook, 'sys': sys, 'include_paths': include_paths, 'exclude_paths': exclude_paths}, 0)

def remove_hooks():
    if _hook in sys.meta_path:
        sys.meta_path.remove(_hook)

def detect_hooks():
    """
    Returns True if the import hooks are installed, False if not.
    """
    return _hook in sys.meta_path


class hooks(object):
    """
    Acts as a context manager. Use like this:

    >>> from past import translation
    >>> with translation.hooks():
    ...     import mypy2module
    >>> import requests        # py2/3 compatible anyway
    >>> # etc.
    """
    
    def __enter__(self):
        self.hooks_were_installed = detect_hooks()
        install_hooks()
        return self
    
    def __exit__(self, *args):
        if not self.hooks_were_installed:
            remove_hooks()



class suspend_hooks(object):
    """
    Acts as a context manager. Use like this:

    >>> from past import translation
    >>> translation.install_hooks()
    >>> import http.client
    >>> # ...
    >>> with translation.suspend_hooks():
    >>>     import requests     # or others that support Py2/3

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

autotranslate = install_hooks

