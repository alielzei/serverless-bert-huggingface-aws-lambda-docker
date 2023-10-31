"""
Fixer for "class Foo: ..." -> "class Foo(object): ..."
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import LParen, RParen, Name
from libfuturize.fixer_util import touch_import_top

def insert_object(node, idx):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('libpasteurize.fixes.fix_newstyle.insert_object', 'insert_object(node, idx)', {'RParen': RParen, 'Name': Name, 'LParen': LParen, 'node': node, 'idx': idx}, 0)


class FixNewstyle(fixer_base.BaseFix):
    PATTERN = "classdef< 'class' NAME ['(' ')'] colon=':' any >"
    
    def transform(self, node, results):
        colon = results['colon']
        idx = node.children.index(colon)
        if (node.children[idx - 2].value == '(' and node.children[idx - 1].value == ')'):
            del node.children[idx - 2:idx]
            idx -= 2
        insert_object(node, idx)
        touch_import_top('builtins', 'object', node)


