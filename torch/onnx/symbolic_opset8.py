from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_opset9 as sym_opset9
from torch.onnx.symbolic_helper import parse_args, _unimplemented, _black_list_in_opset, _try_get_scalar_type
from torch.onnx.symbolic_opset9 import _cast_Float
import warnings
black_listed_operators = ['nonzero', 'where', 'scatter', 'scatter_add', 'erf', 'sign', 'isnan', 'gather', 'arange', 'masked_fill', 'index_fill', 'index_copy']
for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)

def _interpolate(name, dim, interpolate_mode):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8._interpolate', '_interpolate(name, dim, interpolate_mode)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'name': name, 'dim': dim, 'interpolate_mode': interpolate_mode}, 1)
upsample_nearest1d = _interpolate('upsample_nearest1d', 3, 'nearest')
upsample_nearest2d = _interpolate('upsample_nearest2d', 4, 'nearest')
upsample_nearest3d = _interpolate('upsample_nearest3d', 5, 'nearest')
upsample_linear1d = _interpolate('upsample_linear1d', 3, 'linear')
upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, 'linear')
upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, 'linear')

def __interpolate(g, input, size, scale_factor, mode, align_corners, recompute_scale_factor):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8.__interpolate', '__interpolate(g, input, size, scale_factor, mode, align_corners, recompute_scale_factor)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'g': g, 'input': input, 'size': size, 'scale_factor': scale_factor, 'mode': mode, 'align_corners': align_corners, 'recompute_scale_factor': recompute_scale_factor}, 1)

def _try_cast_integer_to_float(g, *args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8._try_cast_integer_to_float', '_try_cast_integer_to_float(g, *args)', {'_cast_Float': _cast_Float, 'warnings': warnings, 'g': g, 'args': args}, 1)

def _cast_to_type(g, input, to_type):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8._cast_to_type', '_cast_to_type(g, input, to_type)', {'sym_opset9': sym_opset9, 'g': g, 'input': input, 'to_type': to_type}, 1)

def _comparison_operator(g, input, other, op_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8._comparison_operator', '_comparison_operator(g, input, other, op_name)', {'sym_help': sym_help, '_try_cast_integer_to_float': _try_cast_integer_to_float, 'g': g, 'input': input, 'other': other, 'op_name': op_name}, 1)

def gt(g, input, other):
    return _comparison_operator(g, input, other, 'Greater')

def lt(g, input, other):
    return _comparison_operator(g, input, other, 'Less')

def bmm(g, self, other):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8.bmm', 'bmm(g, self, other)', {'_try_get_scalar_type': _try_get_scalar_type, '_try_cast_integer_to_float': _try_cast_integer_to_float, '_cast_to_type': _cast_to_type, 'g': g, 'self': self, 'other': other}, 1)

def matmul(g, self, other):
    return bmm(g, self, other)

def prelu(g, self, weight):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8.prelu', 'prelu(g, self, weight)', {'_try_get_scalar_type': _try_get_scalar_type, '_try_cast_integer_to_float': _try_cast_integer_to_float, '_cast_to_type': _cast_to_type, 'g': g, 'self': self, 'weight': weight}, 1)

def mm(g, self, other):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8.mm', 'mm(g, self, other)', {'sym_help': sym_help, '_try_get_scalar_type': _try_get_scalar_type, '_try_cast_integer_to_float': _try_cast_integer_to_float, '_cast_to_type': _cast_to_type, 'g': g, 'self': self, 'other': other}, 1)

@parse_args('v', 'v', 'v', 't', 't')
def addmm(g, self, mat1, mat2, beta, alpha):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8.addmm', 'addmm(g, self, mat1, mat2, beta, alpha)', {'_try_get_scalar_type': _try_get_scalar_type, '_try_cast_integer_to_float': _try_cast_integer_to_float, '_cast_to_type': _cast_to_type, 'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'self': self, 'mat1': mat1, 'mat2': mat2, 'beta': beta, 'alpha': alpha}, 1)

def view(g, self, size):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8.view', 'view(g, self, size)', {'sym_help': sym_help, '_try_cast_integer_to_float': _try_cast_integer_to_float, '_cast_to_type': _cast_to_type, 'torch': torch, 'g': g, 'self': self, 'size': size}, 1)

def flatten(g, input, start_dim, end_dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8.flatten', 'flatten(g, input, start_dim, end_dim)', {'sym_help': sym_help, '_try_get_scalar_type': _try_get_scalar_type, '_try_cast_integer_to_float': _try_cast_integer_to_float, '_cast_to_type': _cast_to_type, 'sym_opset9': sym_opset9, 'g': g, 'input': input, 'start_dim': start_dim, 'end_dim': end_dim}, 1)

def _constant_fill(g, sizes, dtype, const_value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8._constant_fill', '_constant_fill(g, sizes, dtype, const_value)', {'sym_help': sym_help, 'g': g, 'sizes': sizes, 'dtype': dtype, 'const_value': const_value}, 1)

@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def empty(g, sizes, dtype, layout, device, pin_memory=False, memory_format=None):
    return zeros(g, sizes, dtype, layout, device, pin_memory)

@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def empty_like(g, input, dtype, layout, device, pin_memory=False, memory_format=None):
    return zeros_like(g, input, dtype, layout, device, pin_memory)

@parse_args('v', 'i', 'v', 'v', 'v')
def zeros(g, sizes, dtype, layout, device, pin_memory=False):
    return _constant_fill(g, sizes, dtype, 0)

@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def zeros_like(g, input, dtype, layout, device, pin_memory=False, memory_format=None):
    shape = g.op('Shape', input)
    return _constant_fill(g, shape, dtype, 0)

@parse_args('v', 'i', 'v', 'v', 'v')
def ones(g, sizes, dtype, layout, device, pin_memory=False):
    return _constant_fill(g, sizes, dtype, 1)

@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def ones_like(g, input, dtype, layout, device, pin_memory=False, memory_format=None):
    shape = g.op('Shape', input)
    return _constant_fill(g, shape, dtype, 1)

def full(g, sizes, value, dtype, layout, device, pin_memory=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset8.full', 'full(g, sizes, value, dtype, layout, device, pin_memory=False)', {'sym_help': sym_help, 'zeros': zeros, 'sym_opset9': sym_opset9, 'torch': torch, '_constant_fill': _constant_fill, 'g': g, 'sizes': sizes, 'value': value, 'dtype': dtype, 'layout': layout, 'device': device, 'pin_memory': pin_memory}, 1)

@parse_args('v', 'f', 'i', 'v', 'v', 'v', 'v')
def full_like(g, input, fill_value, dtype, layout, device, pin_memory=False, memory_format=None):
    shape = g.op('Shape', input)
    return _constant_fill(g, shape, dtype, fill_value)

