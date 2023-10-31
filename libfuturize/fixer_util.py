"""
Utility functions from 2to3, 3to2 and python-modernize (and some home-grown
ones).

Licences:
2to3: PSF License v2
3to2: Apache Software License (from 3to2/setup.py)
python-modernize licence: BSD (from python-modernize/LICENSE)
"""

from lib2to3.fixer_util import FromImport, Newline, is_import, find_root, does_tree_import, Comma
from lib2to3.pytree import Leaf, Node
from lib2to3.pygram import python_symbols as syms, python_grammar
from lib2to3.pygram import token
from lib2to3.fixer_util import Node, Call, Name, syms, Comma, Number
import re

def canonical_fix_name(fix, avail_fixes):
    """
    Examples:
    >>> canonical_fix_name('fix_wrap_text_literals')
    'libfuturize.fixes.fix_wrap_text_literals'
    >>> canonical_fix_name('wrap_text_literals')
    'libfuturize.fixes.fix_wrap_text_literals'
    >>> canonical_fix_name('wrap_te')
    ValueError("unknown fixer name")
    >>> canonical_fix_name('wrap')
    ValueError("ambiguous fixer name")
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.canonical_fix_name', 'canonical_fix_name(fix, avail_fixes)', {'fix': fix, 'avail_fixes': avail_fixes}, 1)

def Star(prefix=None):
    return Leaf(token.STAR, '*', prefix=prefix)

def DoubleStar(prefix=None):
    return Leaf(token.DOUBLESTAR, '**', prefix=prefix)

def Minus(prefix=None):
    return Leaf(token.MINUS, '-', prefix=prefix)

def commatize(leafs):
    """
    Accepts/turns: (Name, Name, ..., Name, Name)
    Returns/into: (Name, Comma, Name, Comma, ..., Name, Comma, Name)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.commatize', 'commatize(leafs)', {'Comma': Comma, 'leafs': leafs}, 1)

def indentation(node):
    """
    Returns the indentation for this node
    Iff a node is in a suite, then it has indentation.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.indentation', 'indentation(node)', {'syms': syms, 'token': token, 'node': node}, 1)

def indentation_step(node):
    """
    Dirty little trick to get the difference between each indentation level
    Implemented by finding the shortest indentation string
    (technically, the "least" of all of the indentation strings, but
    tabs and spaces mixed won't get this far, so those are synonymous.)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.indentation_step', 'indentation_step(node)', {'find_root': find_root, 'token': token, 'node': node}, 1)

def suitify(parent):
    """
    Turn the stuff after the first colon in parent's children
    into a suite, if it wasn't already
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.suitify', 'suitify(parent)', {'syms': syms, 'token': token, 'Node': Node, 'Newline': Newline, 'Leaf': Leaf, 'indentation': indentation, 'indentation_step': indentation_step, 'parent': parent}, 1)

def NameImport(package, as_name=None, prefix=None):
    """
    Accepts a package (Name node), name to import it as (string), and
    optional prefix and returns a node:
    import <package> [as <as_name>]
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.NameImport', 'NameImport(package, as_name=None, prefix=None)', {'Name': Name, 'Node': Node, 'syms': syms, 'package': package, 'as_name': as_name, 'prefix': prefix}, 1)
_compound_stmts = (syms.if_stmt, syms.while_stmt, syms.for_stmt, syms.try_stmt, syms.with_stmt)
_import_stmts = (syms.import_name, syms.import_from)

def import_binding_scope(node):
    """
    Generator yields all nodes for which a node (an import_stmt) has scope
    The purpose of this is for a call to _find() on each of them
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.import_binding_scope', 'import_binding_scope(node)', {'_import_stmts': _import_stmts, 'token': token, 'syms': syms, '_compound_stmts': _compound_stmts, 'node': node}, 1)

def ImportAsName(name, as_name, prefix=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.ImportAsName', 'ImportAsName(name, as_name, prefix=None)', {'Name': Name, 'Node': Node, 'syms': syms, 'name': name, 'as_name': as_name, 'prefix': prefix}, 1)

def is_docstring(node):
    """
    Returns True if the node appears to be a docstring
    """
    return (node.type == syms.simple_stmt and len(node.children) > 0 and node.children[0].type == token.STRING)

def future_import(feature, node):
    """
    This seems to work
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.future_import', 'future_import(feature, node)', {'find_root': find_root, 'does_tree_import': does_tree_import, 'is_shebang_comment': is_shebang_comment, 'is_encoding_comment': is_encoding_comment, 'is_docstring': is_docstring, 'check_future_import': check_future_import, 'FromImport': FromImport, 'Leaf': Leaf, 'token': token, 'Newline': Newline, 'Node': Node, 'syms': syms, 'feature': feature, 'node': node}, 1)

def future_import2(feature, node):
    """
    An alternative to future_import() which might not work ...
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.future_import2', 'future_import2(feature, node)', {'find_root': find_root, 'does_tree_import': does_tree_import, 'syms': syms, 'token': token, 'FromImport': FromImport, 'Leaf': Leaf, 'Newline': Newline, 'Node': Node, 'feature': feature, 'node': node}, 1)

def parse_args(arglist, scheme):
    """
    Parse a list of arguments into a dict
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.parse_args', 'parse_args(arglist, scheme)', {'token': token, 'syms': syms, 'arglist': arglist, 'scheme': scheme}, 1)

def is_import_stmt(node):
    return (node.type == syms.simple_stmt and node.children and is_import(node.children[0]))

def touch_import_top(package, name_to_import, node):
    """Works like `does_tree_import` but adds an import statement at the
    top if it was not imported (but below any __future__ imports) and below any
    comments such as shebang lines).

    Based on lib2to3.fixer_util.touch_import()

    Calling this multiple times adds the imports in reverse order.

    Also adds "standard_library.install_aliases()" after "from future import
    standard_library".  This should probably be factored into another function.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.touch_import_top', 'touch_import_top(package, name_to_import, node)', {'find_root': find_root, 'does_tree_import': does_tree_import, 'check_future_import': check_future_import, 'syms': syms, 'is_docstring': is_docstring, 'Node': Node, 'Leaf': Leaf, 'token': token, 'FromImport': FromImport, 'Newline': Newline, 'package': package, 'name_to_import': name_to_import, 'node': node}, 1)

def check_future_import(node):
    """If this is a future import, return set of symbols that are imported,
    else return None."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.check_future_import', 'check_future_import(node)', {'syms': syms, 'token': token, 'node': node}, 1)
SHEBANG_REGEX = '^#!.*python'
ENCODING_REGEX = '^#.*coding[:=]\\s*([-\\w.]+)'

def is_shebang_comment(node):
    """
    Comments are prefixes for Leaf nodes. Returns whether the given node has a
    prefix that looks like a shebang line or an encoding line:

        #!/usr/bin/env python
        #!/usr/bin/python3
    """
    return bool(re.match(SHEBANG_REGEX, node.prefix))

def is_encoding_comment(node):
    """
    Comments are prefixes for Leaf nodes. Returns whether the given node has a
    prefix that looks like an encoding line:

        # coding: utf-8
        # encoding: utf-8
        # -*- coding: <encoding name> -*-
        # vim: set fileencoding=<encoding name> :
    """
    return bool(re.match(ENCODING_REGEX, node.prefix))

def wrap_in_fn_call(fn_name, args, prefix=None):
    """
    Example:
    >>> wrap_in_fn_call("oldstr", (arg,))
    oldstr(arg)

    >>> wrap_in_fn_call("olddiv", (arg1, arg2))
    olddiv(arg1, arg2)

    >>> wrap_in_fn_call("olddiv", [arg1, comma, arg2, comma, arg3])
    olddiv(arg1, arg2, arg3)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixer_util.wrap_in_fn_call', 'wrap_in_fn_call(fn_name, args, prefix=None)', {'Comma': Comma, 'Call': Call, 'Name': Name, 'fn_name': fn_name, 'args': args, 'prefix': prefix}, 1)

