"""Fixer for 'raise E, V'

From Armin Ronacher's ``python-modernize``.

raise         -> raise
raise E       -> raise E
raise E, 5    -> raise E(5)
raise E, 5, T -> raise E(5).with_traceback(T)
raise E, None, T -> raise E.with_traceback(T)

raise (((E, E'), E''), E'''), 5 -> raise E(5)
raise "foo", V, T               -> warns about string exceptions

raise E, (V1, V2) -> raise E(V1, V2)
raise E, (V1, V2), T -> raise E(V1, V2).with_traceback(T)


CAVEATS:
1) "raise E, V, T" cannot be translated safely in general. If V
   is not a tuple or a (number, string, None) literal, then:

   raise E, V, T -> from future.utils import raise_
                    raise_(E, V, T)
"""

from lib2to3 import pytree, fixer_base
from lib2to3.pgen2 import token
from lib2to3.fixer_util import Name, Call, is_tuple, Comma, Attr, ArgList
from libfuturize.fixer_util import touch_import_top


class FixRaise(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "\n    raise_stmt< 'raise' exc=any [',' val=any [',' tb=any]] >\n    "
    
    def transform(self, node, results):
        syms = self.syms
        exc = results['exc'].clone()
        if exc.type == token.STRING:
            msg = 'Python 3 does not support string exceptions'
            self.cannot_convert(node, msg)
            return
        if is_tuple(exc):
            while is_tuple(exc):
                exc = exc.children[1].children[0].clone()
            exc.prefix = ' '
        if 'tb' in results:
            tb = results['tb'].clone()
        else:
            tb = None
        if 'val' in results:
            val = results['val'].clone()
            if is_tuple(val):
                args = [c.clone() for c in val.children[1:-1]]
                exc = Call(exc, args)
            elif val.type in (token.NUMBER, token.STRING):
                val.prefix = ''
                exc = Call(exc, [val])
            elif (val.type == token.NAME and val.value == 'None'):
                pass
            else:
                touch_import_top('future.utils', 'raise_', node)
                exc.prefix = ''
                args = [exc, Comma(), val]
                if tb is not None:
                    args += [Comma(), tb]
                return Call(Name('raise_'), args, prefix=node.prefix)
        if tb is not None:
            tb.prefix = ''
            exc_list = Attr(exc, Name('with_traceback')) + [ArgList([tb])]
        else:
            exc_list = [exc]
        return pytree.Node(syms.raise_stmt, [Name('raise')] + exc_list, prefix=node.prefix)


