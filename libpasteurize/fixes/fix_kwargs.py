"""
Fixer for Python 3 function parameter syntax
This fixer is rather sensitive to incorrect py3k syntax.
"""

from lib2to3 import fixer_base
from lib2to3.fixer_util import token, String, Newline, Comma, Name
from libfuturize.fixer_util import indentation, suitify, DoubleStar
_assign_template = "%(name)s = %(kwargs)s['%(name)s']; del %(kwargs)s['%(name)s']"
_if_template = "if '%(name)s' in %(kwargs)s: %(assign)s"
_else_template = 'else: %(name)s = %(default)s'
_kwargs_default_name = '_3to2kwargs'

def gen_params(raw_params):
    """
    Generator that yields tuples of (name, default_value) for each parameter in the list
    If no default is given, then it is default_value is None (not Leaf(token.NAME, 'None'))
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('libpasteurize.fixes.fix_kwargs.gen_params', 'gen_params(raw_params)', {'token': token, 'raw_params': raw_params}, 0)

def remove_params(raw_params, kwargs_default=_kwargs_default_name):
    """
    Removes all keyword-only args from the params list and a bare star, if any.
    Does not add the kwargs dict if needed.
    Returns True if more action is needed, False if not
    (more action is needed if no kwargs dict exists)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libpasteurize.fixes.fix_kwargs.remove_params', 'remove_params(raw_params, kwargs_default=_kwargs_default_name)', {'token': token, 'raw_params': raw_params, 'kwargs_default': kwargs_default, '_kwargs_default_name': _kwargs_default_name}, 1)

def needs_fixing(raw_params, kwargs_default=_kwargs_default_name):
    """
    Returns string with the name of the kwargs dict if the params after the first star need fixing
    Otherwise returns empty string
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libpasteurize.fixes.fix_kwargs.needs_fixing', 'needs_fixing(raw_params, kwargs_default=_kwargs_default_name)', {'token': token, 'raw_params': raw_params, 'kwargs_default': kwargs_default, '_kwargs_default_name': _kwargs_default_name}, 1)


class FixKwargs(fixer_base.BaseFix):
    run_order = 7
    PATTERN = "funcdef< 'def' NAME parameters< '(' arglist=typedargslist< params=any* > ')' > ':' suite=any >"
    
    def transform(self, node, results):
        params_rawlist = results['params']
        for (i, item) in enumerate(params_rawlist):
            if item.type == token.STAR:
                params_rawlist = params_rawlist[i:]
                break
        else:
            return
        new_kwargs = needs_fixing(params_rawlist)
        if not new_kwargs:
            return
        suitify(node)
        suite = node.children[4]
        first_stmt = suite.children[2]
        ident = indentation(first_stmt)
        for (name, default_value) in gen_params(params_rawlist):
            if default_value is None:
                suite.insert_child(2, Newline())
                suite.insert_child(2, String(_assign_template % {'name': name, 'kwargs': new_kwargs}, prefix=ident))
            else:
                suite.insert_child(2, Newline())
                suite.insert_child(2, String(_else_template % {'name': name, 'default': default_value}, prefix=ident))
                suite.insert_child(2, Newline())
                suite.insert_child(2, String(_if_template % {'assign': _assign_template % {'name': name, 'kwargs': new_kwargs}, 'name': name, 'kwargs': new_kwargs}, prefix=ident))
        first_stmt.prefix = ident
        suite.children[2].prefix = ''
        must_add_kwargs = remove_params(params_rawlist)
        if must_add_kwargs:
            arglist = results['arglist']
            if (len(arglist.children) > 0 and arglist.children[-1].type != token.COMMA):
                arglist.append_child(Comma())
            arglist.append_child(DoubleStar(prefix=' '))
            arglist.append_child(Name(new_kwargs))


