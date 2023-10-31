"""
Fixer for:
(a,)* *b (,c)* [,] = s
for (a,)* *b (,c)* [,] in d: ...
"""

from lib2to3 import fixer_base
from itertools import count
from lib2to3.fixer_util import Assign, Comma, Call, Newline, Name, Number, token, syms, Node, Leaf
from libfuturize.fixer_util import indentation, suitify, commatize

def assignment_source(num_pre, num_post, LISTNAME, ITERNAME):
    """
    Accepts num_pre and num_post, which are counts of values
    before and after the starg (not including the starg)
    Returns a source fit for Assign() from fixer_util
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libpasteurize.fixes.fix_unpacking.assignment_source', 'assignment_source(num_pre, num_post, LISTNAME, ITERNAME)', {'unicode': unicode, 'Node': Node, 'syms': syms, 'Name': Name, 'Leaf': Leaf, 'token': token, 'Number': Number, 'num_pre': num_pre, 'num_post': num_post, 'LISTNAME': LISTNAME, 'ITERNAME': ITERNAME}, 1)


class FixUnpacking(fixer_base.BaseFix):
    PATTERN = "\n    expl=expr_stmt< testlist_star_expr<\n        pre=(any ',')*\n            star_expr< '*' name=NAME >\n        post=(',' any)* [','] > '=' source=any > |\n    impl=for_stmt< 'for' lst=exprlist<\n        pre=(any ',')*\n            star_expr< '*' name=NAME >\n        post=(',' any)* [','] > 'in' it=any ':' suite=any>"
    
    def fix_explicit_context(self, node, results):
        (pre, name, post, source) = (results.get(n) for n in ('pre', 'name', 'post', 'source'))
        pre = [n.clone() for n in pre if n.type == token.NAME]
        name.prefix = ' '
        post = [n.clone() for n in post if n.type == token.NAME]
        target = [n.clone() for n in commatize(pre + [name.clone()] + post)]
        target.append(Comma())
        source.prefix = ''
        setup_line = Assign(Name(self.LISTNAME), Call(Name('list'), [source.clone()]))
        power_line = Assign(target, assignment_source(len(pre), len(post), self.LISTNAME, self.ITERNAME))
        return (setup_line, power_line)
    
    def fix_implicit_context(self, node, results):
        """
        Only example of the implicit context is
        a for loop, so only fix that.
        """
        (pre, name, post, it) = (results.get(n) for n in ('pre', 'name', 'post', 'it'))
        pre = [n.clone() for n in pre if n.type == token.NAME]
        name.prefix = ' '
        post = [n.clone() for n in post if n.type == token.NAME]
        target = [n.clone() for n in commatize(pre + [name.clone()] + post)]
        target.append(Comma())
        source = it.clone()
        source.prefix = ''
        setup_line = Assign(Name(self.LISTNAME), Call(Name('list'), [Name(self.ITERNAME)]))
        power_line = Assign(target, assignment_source(len(pre), len(post), self.LISTNAME, self.ITERNAME))
        return (setup_line, power_line)
    
    def transform(self, node, results):
        """
        a,b,c,d,e,f,*g,h,i = range(100) changes to
        _3to2list = list(range(100))
        a,b,c,d,e,f,g,h,i, = _3to2list[:6] + [_3to2list[6:-2]] + _3to2list[-2:]

        and

        for a,b,*c,d,e in iter_of_iters: do_stuff changes to
        for _3to2iter in iter_of_iters:
            _3to2list = list(_3to2iter)
            a,b,c,d,e, = _3to2list[:2] + [_3to2list[2:-2]] + _3to2list[-2:]
            do_stuff
        """
        self.LISTNAME = self.new_name('_3to2list')
        self.ITERNAME = self.new_name('_3to2iter')
        (expl, impl) = (results.get('expl'), results.get('impl'))
        if expl is not None:
            (setup_line, power_line) = self.fix_explicit_context(node, results)
            setup_line.prefix = expl.prefix
            power_line.prefix = indentation(expl.parent)
            setup_line.append_child(Newline())
            parent = node.parent
            i = node.remove()
            parent.insert_child(i, power_line)
            parent.insert_child(i, setup_line)
        elif impl is not None:
            (setup_line, power_line) = self.fix_implicit_context(node, results)
            suitify(node)
            suite = [k for k in node.children if k.type == syms.suite][0]
            setup_line.prefix = ''
            power_line.prefix = suite.children[1].value
            suite.children[2].prefix = indentation(suite.children[2])
            suite.insert_child(2, Newline())
            suite.insert_child(2, power_line)
            suite.insert_child(2, Newline())
            suite.insert_child(2, setup_line)
            results.get('lst').replace(Name(self.ITERNAME, prefix=' '))


