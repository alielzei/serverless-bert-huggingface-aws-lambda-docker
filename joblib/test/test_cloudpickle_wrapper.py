"""
Test that our implementation of wrap_non_picklable_objects mimics
properly the loky implementation.
"""

from .._cloudpickle_wrapper import wrap_non_picklable_objects
from .._cloudpickle_wrapper import _my_wrap_non_picklable_objects

def a_function(x):
    return x


class AClass(object):
    
    def __call__(self, x):
        return x


def test_wrap_non_picklable_objects():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_cloudpickle_wrapper.test_wrap_non_picklable_objects', 'test_wrap_non_picklable_objects()', {'a_function': a_function, 'AClass': AClass, 'wrap_non_picklable_objects': wrap_non_picklable_objects, '_my_wrap_non_picklable_objects': _my_wrap_non_picklable_objects}, 0)

