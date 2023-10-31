"""
Fixer for (metaclass=X) -> __metaclass__ = X
Some semantics (see PEP 3115) may be altered in the translation."""

from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, syms, Node, Leaf, Newline, find_root
from lib2to3.pygram import token
from libfuturize.fixer_util import indentation, suitify

def has_metaclass(parent):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libpasteurize.fixes.fix_metaclass.has_metaclass', 'has_metaclass(parent)', {'syms': syms, 'Leaf': Leaf, 'token': token, 'Node': Node, 'parent': parent}, 1)


class FixMetaclass(fixer_base.BaseFix):
    PATTERN = '\n    classdef<any*>\n    '
    
    def transform(self, node, results):
        meta_results = has_metaclass(node)
        if not meta_results:
            return
        for meta in meta_results:
            meta.remove()
        target = Leaf(token.NAME, '__metaclass__')
        equal = Leaf(token.EQUAL, '=', prefix=' ')
        name = meta
        name.prefix = ' '
        stmt_node = Node(syms.atom, [target, equal, name])
        suitify(node)
        for item in node.children:
            if item.type == syms.suite:
                for stmt in item.children:
                    if stmt.type == token.INDENT:
                        loc = item.children.index(stmt) + 1
                        ident = Leaf(token.INDENT, stmt.value)
                        item.insert_child(loc, ident)
                        item.insert_child(loc, Newline())
                        item.insert_child(loc, stmt_node)
                        break


