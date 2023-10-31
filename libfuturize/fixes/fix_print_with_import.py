"""
For the ``future`` package.

Turns any print statements into functions and adds this import line:

    from __future__ import print_function

at the top to retain compatibility with Python 2.6+.
"""

from libfuturize.fixes.fix_print import FixPrint
from libfuturize.fixer_util import future_import


class FixPrintWithImport(FixPrint):
    run_order = 7
    
    def transform(self, node, results):
        future_import('print_function', node)
        n_stmt = super(FixPrintWithImport, self).transform(node, results)
        return n_stmt


