import sys
import ast
import inspect
import re
import torch
from .._jit_internal import List, BroadcastingList1, BroadcastingList2, BroadcastingList3, Tuple, is_tuple, is_list, Dict, is_dict, Optional, is_optional, _qualified_name, Any, RRef, is_rref
from torch._C import TensorType, TupleType, FloatType, IntType, ListType, StringType, DictType, BoolType, OptionalType, ClassType, InterfaceType, AnyType, NoneType, DeviceObjType, RRefType
from textwrap import dedent
from torch._six import builtins, PY2
from torch._utils_internal import get_source_lines_and_file
PY35 = sys.version_info >= (3, 5)


class Module(object):
    
    def __init__(self, name, members):
        self.name = name
        self.members = members
    
    def __getattr__(self, name):
        try:
            return self.members[name]
        except KeyError:
            raise RuntimeError('Module {} has no member called {}'.format(self.name, name))



class EvalEnv(object):
    env = {'torch': Module('torch', {'Tensor': torch.Tensor}), 'Tensor': torch.Tensor, 'typing': Module('typing', {'Tuple': Tuple}), 'Tuple': Tuple, 'List': List, 'Dict': Dict, 'Optional': Optional, 'RRef': RRef}
    
    def __init__(self, rcb):
        self.rcb = rcb
    
    def __getitem__(self, name):
        if name in self.env:
            return self.env[name]
        if self.rcb is not None:
            return self.rcb(name)
        return getattr(builtins, name, None)


def get_signature(fn, rcb, loc, is_method):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.annotations.get_signature', 'get_signature(fn, rcb, loc, is_method)', {'PY35': PY35, 'try_real_annotations': try_real_annotations, 'dedent': dedent, 'get_source_lines_and_file': get_source_lines_and_file, 'get_type_line': get_type_line, 'parse_type_line': parse_type_line, 'fn': fn, 'rcb': rcb, 'loc': loc, 'is_method': is_method}, 1)

def is_function_or_method(the_callable):
    return (inspect.isfunction(the_callable) or inspect.ismethod(the_callable))

def is_vararg(the_callable):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.annotations.is_vararg', 'is_vararg(the_callable)', {'is_function_or_method': is_function_or_method, 'PY2': PY2, 'inspect': inspect, 'the_callable': the_callable}, 1)

def get_param_names(fn, n_args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.annotations.get_param_names', 'get_param_names(fn, n_args)', {'is_function_or_method': is_function_or_method, 'PY2': PY2, 'inspect': inspect, 'fn': fn, 'n_args': n_args}, 1)

def check_fn(fn, loc):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.annotations.check_fn', 'check_fn(fn, loc)', {'dedent': dedent, 'get_source_lines_and_file': get_source_lines_and_file, 'IOError': IOError, 'ast': ast, 'torch': torch, 'fn': fn, 'loc': loc}, 1)

def parse_type_line(type_line, rcb, loc):
    """Parses a type annotation specified as a comment.

    Example inputs:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor]
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tensor
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.annotations.parse_type_line', 'parse_type_line(type_line, rcb, loc)', {'split_type_line': split_type_line, 'EvalEnv': EvalEnv, 'ann_to_type': ann_to_type, 'type_line': type_line, 'rcb': rcb, 'loc': loc}, 2)

def get_type_line(source):
    """Tries to find the line containing a comment with the type annotation."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.annotations.get_type_line', 'get_type_line(source)', {'re': re, 'source': source}, 1)

def split_type_line(type_line):
    """Splits the comment with the type annotation into parts for argument and return types.

    For example, for an input of:
        # type: (Tensor, torch.Tensor) -> Tuple[Tensor, Tensor]

    This function will return:
        ("(Tensor, torch.Tensor)", "Tuple[Tensor, Tensor]")

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.annotations.split_type_line', 'split_type_line(type_line)', {'type_line': type_line}, 2)

def try_real_annotations(fn, loc):
    """Tries to use the Py3.5+ annotation syntax to get the type."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.annotations.try_real_annotations', 'try_real_annotations(fn, loc)', {'inspect': inspect, 'ann_to_type': ann_to_type, 'fn': fn, 'loc': loc}, 1)

def try_ann_to_type(ann, loc):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.annotations.try_ann_to_type', 'try_ann_to_type(ann, loc)', {'TensorType': TensorType, 'torch': torch, 'is_tuple': is_tuple, 'TupleType': TupleType, 'try_ann_to_type': try_ann_to_type, 'is_list': is_list, 'ListType': ListType, 'is_dict': is_dict, 'DictType': DictType, 'is_optional': is_optional, 'OptionalType': OptionalType, 'is_rref': is_rref, 'RRefType': RRefType, 'FloatType': FloatType, 'IntType': IntType, 'StringType': StringType, 'BoolType': BoolType, 'Any': Any, 'AnyType': AnyType, 'NoneType': NoneType, 'inspect': inspect, 'ClassType': ClassType, '_qualified_name': _qualified_name, 'InterfaceType': InterfaceType, 'DeviceObjType': DeviceObjType, 'ann': ann, 'loc': loc}, 1)

def ann_to_type(ann, loc):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.annotations.ann_to_type', 'ann_to_type(ann, loc)', {'try_ann_to_type': try_ann_to_type, 'ann': ann, 'loc': loc}, 1)
__all__ = ['Any', 'List', 'BroadcastingList1', 'BroadcastingList2', 'BroadcastingList3', 'Tuple', 'is_tuple', 'is_list', 'Dict', 'is_dict', 'TensorType', 'TupleType', 'FloatType', 'IntType', 'ListType', 'StringType', 'DictType', 'AnyType', 'Module', 'get_signature', 'check_fn', 'get_param_names', 'parse_type_line', 'get_type_line', 'split_type_line', 'try_real_annotations', 'try_ann_to_type', 'ann_to_type']

