"""Prints type-coercion tables for the built-in NumPy types

"""

import numpy as np
from collections import namedtuple


class GenericObject:
    
    def __init__(self, v):
        self.v = v
    
    def __add__(self, other):
        return self
    
    def __radd__(self, other):
        return self
    dtype = np.dtype('O')


def print_cancast_table(ntypes):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.testing.print_coercion_tables.print_cancast_table', 'print_cancast_table(ntypes)', {'np': np, 'ntypes': ntypes}, 0)

def print_coercion_table(ntypes, inputfirstvalue, inputsecondvalue, firstarray, use_promote_types=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.testing.print_coercion_tables.print_coercion_table', 'print_coercion_table(ntypes, inputfirstvalue, inputsecondvalue, firstarray, use_promote_types=False)', {'GenericObject': GenericObject, 'np': np, 'ntypes': ntypes, 'inputfirstvalue': inputfirstvalue, 'inputsecondvalue': inputsecondvalue, 'firstarray': firstarray, 'use_promote_types': use_promote_types}, 0)

def print_new_cast_table(*, can_cast=True, legacy=False, flags=False):
    """Prints new casts, the values given are default "can-cast" values, not
    actual ones.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.testing.print_coercion_tables.print_new_cast_table', 'print_new_cast_table(*, can_cast=True, legacy=False, flags=False)', {'namedtuple': namedtuple, 'np': np, 'can_cast': can_cast, 'legacy': legacy, 'flags': flags}, 2)
if __name__ == '__main__':
    print('can cast')
    print_cancast_table(np.typecodes['All'])
    print()
    print("In these tables, ValueError is '!', OverflowError is '@', TypeError is '#'")
    print()
    print('scalar + scalar')
    print_coercion_table(np.typecodes['All'], 0, 0, False)
    print()
    print('scalar + neg scalar')
    print_coercion_table(np.typecodes['All'], 0, -1, False)
    print()
    print('array + scalar')
    print_coercion_table(np.typecodes['All'], 0, 0, True)
    print()
    print('array + neg scalar')
    print_coercion_table(np.typecodes['All'], 0, -1, True)
    print()
    print('promote_types')
    print_coercion_table(np.typecodes['All'], 0, 0, False, True)
    print('New casting type promotion:')
    print_new_cast_table(can_cast=True, legacy=True, flags=True)

