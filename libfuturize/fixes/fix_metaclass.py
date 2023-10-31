"""Fixer for __metaclass__ = X -> (future.utils.with_metaclass(X)) methods.

   The various forms of classef (inherits nothing, inherits once, inherints
   many) don't parse the same in the CST so we look at ALL classes for
   a __metaclass__ and if we find one normalize the inherits to all be
   an arglist.

   For one-liner classes ('class X: pass') there is no indent/dedent so
   we normalize those into having a suite.

   Moving the __metaclass__ into the classdef can also cause the class
   body to be empty so there is some special casing for that as well.

   This fixer also tries very hard to keep original indenting and spacing
   in all those corner cases.
"""

from lib2to3 import fixer_base
from lib2to3.pygram import token
from lib2to3.fixer_util import Name, syms, Node, Leaf, touch_import, Call, String, Comma, parenthesize

def has_metaclass(parent):
    """ we have to check the cls_node without changing it.
        There are two possiblities:
          1)  clsdef => suite => simple_stmt => expr_stmt => Leaf('__meta')
          2)  clsdef => simple_stmt => expr_stmt => Leaf('__meta')
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixes.fix_metaclass.has_metaclass', 'has_metaclass(parent)', {'syms': syms, 'has_metaclass': has_metaclass, 'Leaf': Leaf, 'parent': parent}, 1)

def fixup_parse_tree(cls_node):
    """ one-line classes don't get a suite in the parse tree so we add
        one to normalize the tree
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixes.fix_metaclass.fixup_parse_tree', 'fixup_parse_tree(cls_node)', {'syms': syms, 'token': token, 'Node': Node, 'cls_node': cls_node}, 1)

def fixup_simple_stmt(parent, i, stmt_node):
    """ if there is a semi-colon all the parts count as part of the same
        simple_stmt.  We just want the __metaclass__ part so we move
        everything efter the semi-colon into its own simple_stmt node
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixes.fix_metaclass.fixup_simple_stmt', 'fixup_simple_stmt(parent, i, stmt_node)', {'token': token, 'Node': Node, 'syms': syms, 'parent': parent, 'i': i, 'stmt_node': stmt_node}, 1)

def remove_trailing_newline(node):
    if (node.children and node.children[-1].type == token.NEWLINE):
        node.children[-1].remove()

def find_metas(cls_node):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('libfuturize.fixes.fix_metaclass.find_metas', 'find_metas(cls_node)', {'syms': syms, 'Leaf': Leaf, 'fixup_simple_stmt': fixup_simple_stmt, 'remove_trailing_newline': remove_trailing_newline, 'cls_node': cls_node}, 0)

def fixup_indent(suite):
    """ If an INDENT is followed by a thing with a prefix then nuke the prefix
        Otherwise we get in trouble when removing __metaclass__ at suite start
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixes.fix_metaclass.fixup_indent', 'fixup_indent(suite)', {'token': token, 'Leaf': Leaf, 'suite': suite}, 1)


class FixMetaclass(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = '\n    classdef<any*>\n    '
    
    def transform(self, node, results):
        if not has_metaclass(node):
            return
        fixup_parse_tree(node)
        last_metaclass = None
        for (suite, i, stmt) in find_metas(node):
            last_metaclass = stmt
            stmt.remove()
        text_type = node.children[0].type
        if len(node.children) == 7:
            if node.children[3].type == syms.arglist:
                arglist = node.children[3]
            else:
                parent = node.children[3].clone()
                arglist = Node(syms.arglist, [parent])
                node.set_child(3, arglist)
        elif len(node.children) == 6:
            arglist = Node(syms.arglist, [])
            node.insert_child(3, arglist)
        elif len(node.children) == 4:
            arglist = Node(syms.arglist, [])
            node.insert_child(2, Leaf(token.RPAR, ')'))
            node.insert_child(2, arglist)
            node.insert_child(2, Leaf(token.LPAR, '('))
        else:
            raise ValueError('Unexpected class definition')
        meta_txt = last_metaclass.children[0].children[0]
        meta_txt.value = 'metaclass'
        orig_meta_prefix = meta_txt.prefix
        touch_import('future.utils', 'with_metaclass', node)
        metaclass = last_metaclass.children[0].children[2].clone()
        metaclass.prefix = ''
        arguments = [metaclass]
        if arglist.children:
            if len(arglist.children) == 1:
                base = arglist.children[0].clone()
                base.prefix = ' '
            else:
                bases = parenthesize(arglist.clone())
                bases.prefix = ' '
                base = Call(Name('type'), [String("'NewBase'"), Comma(), bases, Comma(), Node(syms.atom, [Leaf(token.LBRACE, '{'), Leaf(token.RBRACE, '}')], prefix=' ')], prefix=' ')
            arguments.extend([Comma(), base])
        arglist.replace(Call(Name('with_metaclass', prefix=arglist.prefix), arguments))
        fixup_indent(suite)
        if not suite.children:
            suite.remove()
            pass_leaf = Leaf(text_type, 'pass')
            pass_leaf.prefix = orig_meta_prefix
            node.append_child(pass_leaf)
            node.append_child(Leaf(token.NEWLINE, '\n'))
        elif (len(suite.children) > 1 and (suite.children[-2].type == token.INDENT and suite.children[-1].type == token.DEDENT)):
            pass_leaf = Leaf(text_type, 'pass')
            suite.insert_child(-1, pass_leaf)
            suite.insert_child(-1, Leaf(token.NEWLINE, '\n'))


