import ast
from dataclasses import dataclass
import operator as op
operators = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv, ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg}

def eval_expr(expr):
    """
    >>> eval_expr('2*6')
    12
    >>> eval_expr('2**6')
    64
    >>> eval_expr('1 + 2*3**(4) / (6 + -7)')
    -161.0
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib._utils.eval_expr', 'eval_expr(expr)', {'eval_': eval_, 'ast': ast, 'expr': expr}, 1)

def eval_(node):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib._utils.eval_', 'eval_(node)', {'ast': ast, 'operators': operators, 'eval_': eval_, 'node': node}, 1)


@dataclass(frozen=True)
class _Sentinel:
    """A sentinel to mark a parameter as not explicitly set"""
    default_value: object
    
    def __repr__(self):
        return f'default({self.default_value!r})'


