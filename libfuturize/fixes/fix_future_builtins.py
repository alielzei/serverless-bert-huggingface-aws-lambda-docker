"""
For the ``future`` package.

Adds this import line::

    from builtins import XYZ

for each of the functions XYZ that is used in the module.

Adds these imports after any other imports (in an initial block of them).
"""

from __future__ import unicode_literals
from lib2to3 import fixer_base
from lib2to3.pygram import python_symbols as syms
from lib2to3.fixer_util import Name, Call, in_special_context
from libfuturize.fixer_util import touch_import_top
replaced_builtin_fns = 'filter map zip\n                       ascii chr hex input next oct\n                       bytes range str raw_input'.split()
expression = '|'.join(["name='{0}'".format(name) for name in replaced_builtin_fns])


class FixFutureBuiltins(fixer_base.BaseFix):
    BM_compatible = True
    run_order = 7
    PATTERN = "\n              power<\n                 ({0}) trailer< '(' [arglist=any] ')' >\n              rest=any* >\n              |\n              power<\n                  'map' trailer< '(' [arglist=any] ')' >\n              >\n              ".format(expression)
    
    def transform(self, node, results):
        name = results['name']
        touch_import_top('builtins', name.value, node)


