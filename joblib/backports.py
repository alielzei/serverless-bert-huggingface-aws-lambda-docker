"""
Backports of fixes for joblib dependencies
"""

import os
import re
import time
from os.path import basename
from multiprocessing import util


class Version:
    """Backport from deprecated distutils

    We maintain this backport to avoid introducing a new dependency on
    `packaging`.

    We might rexplore this choice in the future if all major Python projects
    introduce a dependency on packaging anyway.
    """
    
    def __init__(self, vstring=None):
        if vstring:
            self.parse(vstring)
    
    def __repr__(self):
        return "%s ('%s')" % (self.__class__.__name__, str(self))
    
    def __eq__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c == 0
    
    def __lt__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c < 0
    
    def __le__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c <= 0
    
    def __gt__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c > 0
    
    def __ge__(self, other):
        c = self._cmp(other)
        if c is NotImplemented:
            return c
        return c >= 0



class LooseVersion(Version):
    """Backport from deprecated distutils

    We maintain this backport to avoid introducing a new dependency on
    `packaging`.

    We might rexplore this choice in the future if all major Python projects
    introduce a dependency on packaging anyway.
    """
    component_re = re.compile('(\\d+ | [a-z]+ | \\.)', re.VERBOSE)
    
    def __init__(self, vstring=None):
        if vstring:
            self.parse(vstring)
    
    def parse(self, vstring):
        self.vstring = vstring
        components = [x for x in self.component_re.split(vstring) if (x and x != '.')]
        for (i, obj) in enumerate(components):
            try:
                components[i] = int(obj)
            except ValueError:
                pass
        self.version = components
    
    def __str__(self):
        return self.vstring
    
    def __repr__(self):
        return "LooseVersion ('%s')" % str(self)
    
    def _cmp(self, other):
        if isinstance(other, str):
            other = LooseVersion(other)
        elif not isinstance(other, LooseVersion):
            return NotImplemented
        if self.version == other.version:
            return 0
        if self.version < other.version:
            return -1
        if self.version > other.version:
            return 1

try:
    import numpy as np
    
    def make_memmap(filename, dtype='uint8', mode='r+', offset=0, shape=None, order='C', unlink_on_gc_collect=False):
        """Custom memmap constructor compatible with numpy.memmap.

        This function:
        - is a backport the numpy memmap offset fix (See
          https://github.com/numpy/numpy/pull/8443 for more details.
          The numpy fix is available starting numpy 1.13)
        - adds ``unlink_on_gc_collect``, which specifies  explicitly whether
          the process re-constructing the memmap owns a reference to the
          underlying file. If set to True, it adds a finalizer to the
          newly-created memmap that sends a maybe_unlink request for the
          memmaped file to resource_tracker.
        """
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('joblib.backports.make_memmap', "make_memmap(filename, dtype='uint8', mode='r+', offset=0, shape=None, order='C', unlink_on_gc_collect=False)", {'util': util, 'basename': basename, 'os': os, 'np': np, 'LooseVersion': LooseVersion, 'filename': filename, 'dtype': dtype, 'mode': mode, 'offset': offset, 'shape': shape, 'order': order, 'unlink_on_gc_collect': unlink_on_gc_collect}, 1)
except ImportError:
    
    def make_memmap(filename, dtype='uint8', mode='r+', offset=0, shape=None, order='C', unlink_on_gc_collect=False):
        raise NotImplementedError("'joblib.backports.make_memmap' should not be used if numpy is not installed.")
if os.name == 'nt':
    access_denied_errors = (5, 13)
    from os import replace
    
    def concurrency_safe_rename(src, dst):
        """Renames ``src`` into ``dst`` overwriting ``dst`` if it exists.

        On Windows os.replace can yield permission errors if executed by two
        different processes.
        """
        import custom_funtemplate
        custom_funtemplate.rewrite_template('joblib.backports.concurrency_safe_rename', 'concurrency_safe_rename(src, dst)', {'access_denied_errors': access_denied_errors, 'time': time, 'src': src, 'dst': dst}, 0)
else:
    from os import replace as concurrency_safe_rename

