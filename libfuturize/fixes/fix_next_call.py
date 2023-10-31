"""
Based on fix_next.py by Collin Winter.

Replaces it.next() -> next(it), per PEP 3114.

Unlike fix_next.py, this fixer doesn't replace the name of a next method with __next__,
which would break Python 2 compatibility without further help from fixers in
stage 2.
"""

from lib2to3.pgen2 import token
from lib2to3.pygram import python_symbols as syms
from lib2to3 import fixer_base
from lib2to3.fixer_util import Name, Call, find_binding
bind_warning = 'Calls to builtin next() possibly shadowed by global binding'


class FixNextCall(fixer_base.BaseFix):
    BM_compatible = True
    PATTERN = "\n    power< base=any+ trailer< '.' attr='next' > trailer< '(' ')' > >\n    |\n    power< head=any+ trailer< '.' attr='next' > not trailer< '(' ')' > >\n    |\n    global=global_stmt< 'global' any* 'next' any* >\n    "
    order = 'pre'
    
    def start_tree(self, tree, filename):
        super(FixNextCall, self).start_tree(tree, filename)
        n = find_binding('next', tree)
        if n:
            self.warning(n, bind_warning)
            self.shadowed_next = True
        else:
            self.shadowed_next = False
    
    def transform(self, node, results):
        assert results
        base = results.get('base')
        attr = results.get('attr')
        name = results.get('name')
        if base:
            if self.shadowed_next:
                pass
            else:
                base = [n.clone() for n in base]
                base[0].prefix = ''
                node.replace(Call(Name('next', prefix=node.prefix), base))
        elif name:
            pass
        elif attr:
            if is_assign_target(node):
                head = results['head']
                if ''.join([str(n) for n in head]).strip() == '__builtin__':
                    self.warning(node, bind_warning)
                return
        elif 'global' in results:
            self.warning(node, bind_warning)
            self.shadowed_next = True


def is_assign_target(node):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixes.fix_next_call.is_assign_target', 'is_assign_target(node)', {'find_assign': find_assign, 'token': token, 'is_subtree': is_subtree, 'node': node}, 1)

def find_assign(node):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixes.fix_next_call.find_assign', 'find_assign(node)', {'syms': syms, 'find_assign': find_assign, 'node': node}, 1)

def is_subtree(root, node):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.fixes.fix_next_call.is_subtree', 'is_subtree(root, node)', {'is_subtree': is_subtree, 'root': root, 'node': node}, 1)

