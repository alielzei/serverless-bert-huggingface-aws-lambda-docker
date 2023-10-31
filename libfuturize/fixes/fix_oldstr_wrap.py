"""
For the ``future`` package.

Adds this import line:

    from past.builtins import str as oldstr

at the top and wraps any unadorned string literals 'abc' or explicit byte-string
literals b'abc' in oldstr() calls so the code has the same behaviour on Py3 as
on Py2.6/2.7.
"""

from __future__ import unicode_literals
import re
from lib2to3 import fixer_base
from lib2to3.pgen2 import token
from lib2to3.fixer_util import syms
from libfuturize.fixer_util import future_import, touch_import_top, wrap_in_fn_call
_literal_re = re.compile('[^uUrR]?[\\\'\\"]')


class FixOldstrWrap(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = 'STRING'
    
    def transform(self, node, results):
        if node.type == token.STRING:
            touch_import_top('past.types', 'oldstr', node)
            if _literal_re.match(node.value):
                new = node.clone()
                new.prefix = ''
                new.value = 'b' + new.value
                wrapped = wrap_in_fn_call('oldstr', [new], prefix=node.prefix)
                return wrapped


