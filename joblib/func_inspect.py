"""
My own variation on function-specific inspect-like features.
"""

import inspect
import warnings
import re
import os
import collections
from itertools import islice
from tokenize import open as open_py_source
from .logger import pformat
full_argspec_fields = 'args varargs varkw defaults kwonlyargs kwonlydefaults annotations'
full_argspec_type = collections.namedtuple('FullArgSpec', full_argspec_fields)

def get_func_code(func):
    """ Attempts to retrieve a reliable function code hash.

        The reason we don't use inspect.getsource is that it caches the
        source, whereas we want this to be modified on the fly when the
        function is modified.

        Returns
        -------
        func_code: string
            The function code
        source_file: string
            The path to the file in which the function is defined.
        first_line: int
            The first line of the code in the source file.

        Notes
        ------
        This function does a bit more magic than inspect, and is thus
        more robust.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.func_inspect.get_func_code', 'get_func_code(func)', {'os': os, 'inspect': inspect, 're': re, 'open_py_source': open_py_source, 'islice': islice, 'func': func}, 3)

def _clean_win_chars(string):
    """Windows cannot encode some characters in filename."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.func_inspect._clean_win_chars', '_clean_win_chars(string)', {'string': string}, 1)

def get_func_name(func, resolv_alias=True, win_characters=True):
    """ Return the function import path (as a list of module names), and
        a name for the function.

        Parameters
        ----------
        func: callable
            The func to inspect
        resolv_alias: boolean, optional
            If true, possible local aliases are indicated.
        win_characters: boolean, optional
            If true, substitute special characters using urllib.quote
            This is useful in Windows, as it cannot encode some filenames
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.func_inspect.get_func_name', 'get_func_name(func, resolv_alias=True, win_characters=True)', {'inspect': inspect, 'os': os, '_clean_win_chars': _clean_win_chars, 'func': func, 'resolv_alias': resolv_alias, 'win_characters': win_characters}, 2)

def _signature_str(function_name, arg_sig):
    """Helper function to output a function signature"""
    return '{}{}'.format(function_name, arg_sig)

def _function_called_str(function_name, args, kwargs):
    """Helper function to output a function call"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.func_inspect._function_called_str', '_function_called_str(function_name, args, kwargs)', {'function_name': function_name, 'args': args, 'kwargs': kwargs}, 1)

def filter_args(func, ignore_lst, args=(), kwargs=dict()):
    """ Filters the given args and kwargs using a list of arguments to
        ignore, and a function specification.

        Parameters
        ----------
        func: callable
            Function giving the argument specification
        ignore_lst: list of strings
            List of arguments to ignore (either a name of an argument
            in the function spec, or '*', or '**')
        *args: list
            Positional arguments passed to the function.
        **kwargs: dict
            Keyword arguments passed to the function

        Returns
        -------
        filtered_args: list
            List of filtered positional and keyword arguments.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.func_inspect.filter_args', 'filter_args(func, ignore_lst, args=(), kwargs=dict())', {'inspect': inspect, 'warnings': warnings, 'get_func_name': get_func_name, '_signature_str': _signature_str, '_function_called_str': _function_called_str, 'func': func, 'ignore_lst': ignore_lst, 'args': args, 'kwargs': kwargs}, 1)

def _format_arg(arg):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.func_inspect._format_arg', '_format_arg(arg)', {'pformat': pformat, 'arg': arg}, 1)

def format_signature(func, *args, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.func_inspect.format_signature', 'format_signature(func, *args, **kwargs)', {'get_func_name': get_func_name, '_format_arg': _format_arg, 'func': func, 'args': args, 'kwargs': kwargs}, 2)

def format_call(func, args, kwargs, object_name='Memory'):
    """ Returns a nicely formatted statement displaying the function
        call with the given arguments.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.func_inspect.format_call', "format_call(func, args, kwargs, object_name='Memory')", {'format_signature': format_signature, 'func': func, 'args': args, 'kwargs': kwargs, 'object_name': object_name}, 1)

