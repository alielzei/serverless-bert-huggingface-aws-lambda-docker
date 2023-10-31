"""
Test the func_inspect module.
"""

import functools
from joblib.func_inspect import filter_args, get_func_name, get_func_code
from joblib.func_inspect import _clean_win_chars, format_signature
from joblib.memory import Memory
from joblib.test.common import with_numpy
from joblib.testing import fixture, parametrize, raises

def f(x, y=0):
    pass

def g(x):
    pass

def h(x, y=0, *args, **kwargs):
    pass

def i(x=1):
    pass

def j(x, y, **kwargs):
    pass

def k(*args, **kwargs):
    pass

def m1(x, *, y):
    pass

def m2(x, *, y, z=3):
    pass

@fixture(scope='module')
def cached_func(tmpdir_factory):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.cached_func', 'cached_func(tmpdir_factory)', {'Memory': Memory, 'fixture': fixture, 'tmpdir_factory': tmpdir_factory}, 1)


class Klass(object):
    
    def f(self, x):
        return x


@parametrize('func,args,filtered_args', [(f, [[], (1, )], {'x': 1, 'y': 0}), (f, [['x'], (1, )], {'y': 0}), (f, [['y'], (0, )], {'x': 0}), (f, [['y'], (0, ), {'y': 1}], {'x': 0}), (f, [['x', 'y'], (0, )], {}), (f, [[], (0, ), {'y': 1}], {'x': 0, 'y': 1}), (f, [['y'], (), {'x': 2, 'y': 1}], {'x': 2}), (g, [[], (), {'x': 1}], {'x': 1}), (i, [[], (2, )], {'x': 2})])
def test_filter_args(func, args, filtered_args):
    assert filter_args(func, *args) == filtered_args

def test_filter_args_method():
    obj = Klass()
    assert filter_args(obj.f, [], (1, )) == {'x': 1, 'self': obj}

@parametrize('func,args,filtered_args', [(h, [[], (1, )], {'x': 1, 'y': 0, '*': [], '**': {}}), (h, [[], (1, 2, 3, 4)], {'x': 1, 'y': 2, '*': [3, 4], '**': {}}), (h, [[], (1, 25), {'ee': 2}], {'x': 1, 'y': 25, '*': [], '**': {'ee': 2}}), (h, [['*'], (1, 2, 25), {'ee': 2}], {'x': 1, 'y': 2, '**': {'ee': 2}})])
def test_filter_varargs(func, args, filtered_args):
    assert filter_args(func, *args) == filtered_args
test_filter_kwargs_extra_params = [(m1, [[], (1, ), {'y': 2}], {'x': 1, 'y': 2}), (m2, [[], (1, ), {'y': 2}], {'x': 1, 'y': 2, 'z': 3})]

@parametrize('func,args,filtered_args', [(k, [[], (1, 2), {'ee': 2}], {'*': [1, 2], '**': {'ee': 2}}), (k, [[], (3, 4)], {'*': [3, 4], '**': {}})] + test_filter_kwargs_extra_params)
def test_filter_kwargs(func, args, filtered_args):
    assert filter_args(func, *args) == filtered_args

def test_filter_args_2():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.test_filter_args_2', 'test_filter_args_2()', {'filter_args': filter_args, 'j': j, 'functools': functools, 'f': f}, 0)

@parametrize('func,funcname', [(f, 'f'), (g, 'g'), (cached_func, 'cached_func')])
def test_func_name(func, funcname):
    assert get_func_name(func)[1] == funcname

def test_func_name_on_inner_func(cached_func):
    assert get_func_name(cached_func)[1] == 'cached_func_inner'

def test_func_name_collision_on_inner_func():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.test_func_name_collision_on_inner_func', 'test_func_name_collision_on_inner_func()', {'get_func_name': get_func_name, 'f': f}, 1)

def test_func_inspect_errors():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.test_func_inspect_errors', 'test_func_inspect_errors()', {'get_func_name': get_func_name, 'get_func_code': get_func_code, '__file__': __file__}, 0)

def func_with_kwonly_args(a, b, *, kw1='kw1', kw2='kw2'):
    pass

def func_with_signature(a: int, b: int) -> None:
    pass

def test_filter_args_edge_cases():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.test_filter_args_edge_cases', 'test_filter_args_edge_cases()', {'filter_args': filter_args, 'func_with_kwonly_args': func_with_kwonly_args, 'raises': raises, 'func_with_signature': func_with_signature}, 0)

def test_bound_methods():
    """ Make sure that calling the same method on two different instances
        of the same class does resolv to different signatures.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.test_bound_methods', 'test_bound_methods()', {'Klass': Klass, 'filter_args': filter_args}, 0)

@parametrize('exception,regex,func,args', [(ValueError, 'ignore_lst must be a list of parameters to ignore', f, ['bar', (None, )]), (ValueError, "Ignore list: argument \\'(.*)\\' is not defined", g, [['bar'], (None, )]), (ValueError, 'Wrong number of arguments', h, [[]])])
def test_filter_args_error_msg(exception, regex, func, args):
    """ Make sure that filter_args returns decent error messages, for the
        sake of the user.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.test_filter_args_error_msg', 'test_filter_args_error_msg(exception, regex, func, args)', {'raises': raises, 'filter_args': filter_args, 'parametrize': parametrize, 'ValueError': ValueError, 'f': f, 'g': g, 'h': h, 'exception': exception, 'regex': regex, 'func': func, 'args': args}, 0)

def test_filter_args_no_kwargs_mutation():
    """None-regression test against 0.12.0 changes.

    https://github.com/joblib/joblib/pull/75

    Make sure filter args doesn't mutate the kwargs dict that gets passed in.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.test_filter_args_no_kwargs_mutation', 'test_filter_args_no_kwargs_mutation()', {'filter_args': filter_args, 'g': g}, 0)

def test_clean_win_chars():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.test_clean_win_chars', 'test_clean_win_chars()', {'_clean_win_chars': _clean_win_chars}, 0)

@parametrize('func,args,kwargs,sgn_expected', [(g, [list(range(5))], {}, 'g([0, 1, 2, 3, 4])'), (k, [1, 2, (3, 4)], {'y': True}, 'k(1, 2, (3, 4), y=True)')])
def test_format_signature(func, args, kwargs, sgn_expected):
    (path, sgn_result) = format_signature(func, *args, **kwargs)
    assert sgn_result == sgn_expected

def test_format_signature_long_arguments():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.test_format_signature_long_arguments', 'test_format_signature_long_arguments()', {'format_signature': format_signature, 'h': h}, 0)

@with_numpy
def test_format_signature_numpy():
    """ Test the format signature formatting with numpy.
    """
    

def test_special_source_encoding():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.test_special_source_encoding', 'test_special_source_encoding()', {'get_func_code': get_func_code}, 0)

def _get_code():
    from joblib.test.test_func_inspect_special_encoding import big5_f
    return get_func_code(big5_f)[0]

def test_func_code_consistency():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_func_inspect.test_func_code_consistency', 'test_func_code_consistency()', {'_get_code': _get_code}, 0)

