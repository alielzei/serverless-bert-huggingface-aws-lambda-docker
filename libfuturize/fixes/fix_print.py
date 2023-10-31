"""Fixer for print.

Change:
    "print"          into "print()"
    "print ..."      into "print(...)"
    "print(...)"     not changed
    "print ... ,"    into "print(..., end=' ')"
    "print >>x, ..." into "print(..., file=x)"

No changes are applied if print_function is imported from __future__

"""

from lib2to3 import patcomp, pytree, fixer_base
from lib2to3.pgen2 import token
from lib2to3.fixer_util import Name, Call, Comma, String
parend_expr = patcomp.compile_pattern("atom< '(' [arith_expr|atom|power|term|STRING|NAME] ')' >")


class FixPrint(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "\n              simple_stmt< any* bare='print' any* > | print_stmt\n              "
    
    def transform(self, node, results):
        assert results
        bare_print = results.get('bare')
        if bare_print:
            bare_print.replace(Call(Name('print'), [], prefix=bare_print.prefix))
            return
        assert node.children[0] == Name('print')
        args = node.children[1:]
        if (len(args) == 1 and parend_expr.match(args[0])):
            return
        sep = end = file = None
        if (args and args[-1] == Comma()):
            args = args[:-1]
            end = ' '
            string_leaves = [leaf for leaf in args[-1].leaves() if leaf.type == token.STRING]
            if (string_leaves and string_leaves[-1].value[0] != 'r' and string_leaves[-1].value[-3:-1] in ('\\t', '\\n', '\\r')):
                end = ''
        if (args and args[0] == pytree.Leaf(token.RIGHTSHIFT, '>>')):
            assert len(args) >= 2
            file = args[1].clone()
            args = args[3:]
        l_args = [arg.clone() for arg in args]
        if l_args:
            l_args[0].prefix = ''
        if (sep is not None or end is not None or file is not None):
            if sep is not None:
                self.add_kwarg(l_args, 'sep', String(repr(sep)))
            if end is not None:
                self.add_kwarg(l_args, 'end', String(repr(end)))
            if file is not None:
                self.add_kwarg(l_args, 'file', file)
        n_stmt = Call(Name('print'), l_args)
        n_stmt.prefix = node.prefix
        return n_stmt
    
    def add_kwarg(self, l_nodes, s_kwd, n_expr):
        n_expr.prefix = ''
        n_argument = pytree.Node(self.syms.argument, (Name(s_kwd), pytree.Leaf(token.EQUAL, '='), n_expr))
        if l_nodes:
            l_nodes.append(Comma())
            n_argument.prefix = ' '
        l_nodes.append(n_argument)


