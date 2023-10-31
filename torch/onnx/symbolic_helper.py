from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch._C import ListType
import warnings
from sys import maxsize as maxsize
import torch.onnx
import torch.onnx.utils
from functools import wraps
_sum = sum

def _parse_arg(value, desc):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._parse_arg', '_parse_arg(value, desc)', {'_is_value': _is_value, 'value': value, 'desc': desc}, 1)

def _maybe_get_const(value, desc):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._maybe_get_const', '_maybe_get_const(value, desc)', {'_is_value': _is_value, '_parse_arg': _parse_arg, 'value': value, 'desc': desc}, 1)

def _maybe_get_scalar(value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._maybe_get_scalar', '_maybe_get_scalar(value)', {'_maybe_get_const': _maybe_get_const, 'torch': torch, 'value': value}, 1)

def _get_const(value, desc, arg_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._get_const', '_get_const(value, desc, arg_name)', {'_is_value': _is_value, '_parse_arg': _parse_arg, 'value': value, 'desc': desc, 'arg_name': arg_name}, 1)

def _unpack_list(list_value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._unpack_list', '_unpack_list(list_value)', {'list_value': list_value}, 1)

def _is_packed_list(list_value):
    return (_is_value(list_value) and list_value.node().kind() == 'prim::ListConstruct')

def parse_args(*arg_descriptors):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper.parse_args', 'parse_args(*arg_descriptors)', {'_parse_arg': _parse_arg, 'wraps': wraps, 'arg_descriptors': arg_descriptors}, 1)

def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x.item()

def _if_scalar_type_as(g, self, tensor):
    """
    Convert self into the same type of tensor, as necessary.

    We only support implicit casting for scalars, so we never
    actually need to insert an ONNX cast operator here; just
    fix up the scalar.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._if_scalar_type_as', '_if_scalar_type_as(g, self, tensor)', {'torch': torch, 'g': g, 'self': self, 'tensor': tensor}, 1)

def _is_none(x):
    return x.node().mustBeNone()

def _is_value(x):
    return isinstance(x, torch._C.Value)

def _is_tensor_list(x):
    return x.type().isSubtypeOf(ListType.ofTensors())

def _unimplemented(op, msg):
    warnings.warn('ONNX export failed on ' + op + ' because ' + msg + ' not supported')

def _black_list_in_opset(name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._black_list_in_opset', '_black_list_in_opset(name)', {'_export_onnx_opset_version': _export_onnx_opset_version, 'name': name}, 1)

def _try_get_scalar_type(*args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._try_get_scalar_type', '_try_get_scalar_type(*args)', {'args': args}, 1)

def _slice_helper(g, input, axes, starts, ends, steps=None, dynamic_slice=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._slice_helper', '_slice_helper(g, input, axes, starts, ends, steps=None, dynamic_slice=False)', {'_export_onnx_opset_version': _export_onnx_opset_version, 'g': g, 'input': input, 'axes': axes, 'starts': starts, 'ends': ends, 'steps': steps, 'dynamic_slice': dynamic_slice}, 1)

def _is_fp(value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._is_fp', '_is_fp(value)', {'value': value}, 1)

def _sort_helper(g, input, dim, decending=True, out=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._sort_helper', '_sort_helper(g, input, dim, decending=True, out=None)', {'_unimplemented': _unimplemented, 'torch': torch, '_export_onnx_opset_version': _export_onnx_opset_version, 'g': g, 'input': input, 'dim': dim, 'decending': decending, 'out': out}, 1)

def _topk_helper(g, input, k, dim, largest=True, sorted=False, out=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._topk_helper', '_topk_helper(g, input, k, dim, largest=True, sorted=False, out=None)', {'_unimplemented': _unimplemented, '_is_value': _is_value, 'torch': torch, '_export_onnx_opset_version': _export_onnx_opset_version, 'g': g, 'input': input, 'k': k, 'dim': dim, 'largest': largest, 'sorted': sorted, 'out': out}, 1)

def _interpolate_warning(interpolate_mode):
    onnx_op = ('onnx:Resize' if _export_onnx_opset_version >= 10 else 'onnx:Upsample')
    warnings.warn('You are trying to export the model with ' + onnx_op + ' for ONNX opset version ' + str(_export_onnx_opset_version) + ". This operator might cause results to not match the expected results by PyTorch.\nONNX's Upsample/Resize operator did not match Pytorch's Interpolation until opset 11. Attributes to determine how to transform the input were added in onnx:Resize in opset 11 to support Pytorch's behavior (like coordinate_transformation_mode and nearest_mode).\nWe recommend using opset 11 and above for models using this operator. ")

def _unsqueeze_helper(g, input, dim):
    from torch.onnx.symbolic_opset9 import unsqueeze
    return unsqueeze(g, input, dim)

def _interpolate_size_to_scales(g, input, output_size, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._interpolate_size_to_scales', '_interpolate_size_to_scales(g, input, output_size, dim)', {'_maybe_get_const': _maybe_get_const, '_is_value': _is_value, 'torch': torch, 'cast_pytorch_to_onnx': cast_pytorch_to_onnx, '_slice_helper': _slice_helper, 'maxsize': maxsize, 'g': g, 'input': input, 'output_size': output_size, 'dim': dim}, 1)

def _interpolate_get_scales_if_available(g, scales):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._interpolate_get_scales_if_available', '_interpolate_get_scales_if_available(g, scales)', {'_maybe_get_const': _maybe_get_const, '_is_none': _is_none, '_unsqueeze_helper': _unsqueeze_helper, 'cast_pytorch_to_onnx': cast_pytorch_to_onnx, 'torch': torch, 'g': g, 'scales': scales}, 1)

def _get_interpolate_attributes(g, mode, args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._get_interpolate_attributes', '_get_interpolate_attributes(g, mode, args)', {'_interpolate_get_scales_if_available': _interpolate_get_scales_if_available, 'g': g, 'mode': mode, 'args': args}, 2)

def _interpolate_get_scales(g, scale_factor, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._interpolate_get_scales', '_interpolate_get_scales(g, scale_factor, dim)', {'torch': torch, '_unsqueeze_helper': _unsqueeze_helper, 'cast_pytorch_to_onnx': cast_pytorch_to_onnx, 'g': g, 'scale_factor': scale_factor, 'dim': dim}, 1)

def _interpolate_get_scales_and_mode(g, input, size, scale_factor, mode, align_corners):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._interpolate_get_scales_and_mode', '_interpolate_get_scales_and_mode(g, input, size, scale_factor, mode, align_corners)', {'_maybe_get_const': _maybe_get_const, '_interpolate_warning': _interpolate_warning, '_unimplemented': _unimplemented, '_is_none': _is_none, '_interpolate_get_scales': _interpolate_get_scales, '_is_packed_list': _is_packed_list, '_unsqueeze_helper': _unsqueeze_helper, '_interpolate_size_to_scales': _interpolate_size_to_scales, 'g': g, 'input': input, 'size': size, 'scale_factor': scale_factor, 'mode': mode, 'align_corners': align_corners}, 1)

def _scatter_helper(g, self, dim, index, src):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._scatter_helper', '_scatter_helper(g, self, dim, index, src)', {'_export_onnx_opset_version': _export_onnx_opset_version, 'g': g, 'self': self, 'dim': dim, 'index': index, 'src': src}, 1)

def _arange_cast_helper(g, end, start=None, step=None, dtype=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._arange_cast_helper', '_arange_cast_helper(g, end, start=None, step=None, dtype=None)', {'_is_value': _is_value, '_is_none': _is_none, 'scalar_type_to_pytorch_type': scalar_type_to_pytorch_type, 'torch': torch, 'scalar_type_to_onnx': scalar_type_to_onnx, 'g': g, 'end': end, 'start': start, 'step': step, 'dtype': dtype}, 1)

def _size_helper(g, self, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._size_helper', '_size_helper(g, self, dim)', {'torch': torch, 'g': g, 'self': self, 'dim': dim}, 1)

def _index_fill_reshape_helper(g, self, dim, index):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._index_fill_reshape_helper', '_index_fill_reshape_helper(g, self, dim, index)', {'_export_onnx_opset_version': _export_onnx_opset_version, '_unimplemented': _unimplemented, '_parse_arg': _parse_arg, 'g': g, 'self': self, 'dim': dim, 'index': index}, 1)

def _avgpool_helper(tuple_fn, padding, kernel_size, stride, divisor_override, name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._avgpool_helper', '_avgpool_helper(tuple_fn, padding, kernel_size, stride, divisor_override, name)', {'_unimplemented': _unimplemented, 'tuple_fn': tuple_fn, 'padding': padding, 'kernel_size': kernel_size, 'stride': stride, 'divisor_override': divisor_override, 'name': name}, 1)
_default_onnx_opset_version = 9
_onnx_master_opset = 10
_onnx_stable_opsets = [7, 8, 9, 10, 11, 12]
_export_onnx_opset_version = _default_onnx_opset_version

def _set_opset_version(opset_version):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_helper._set_opset_version', '_set_opset_version(opset_version)', {'_default_onnx_opset_version': _default_onnx_opset_version, '_onnx_stable_opsets': _onnx_stable_opsets, '_onnx_master_opset': _onnx_master_opset, 'opset_version': opset_version}, 1)
_operator_export_type = None

def _set_operator_export_type(operator_export_type):
    global _operator_export_type
    _operator_export_type = operator_export_type
cast_pytorch_to_onnx = {'Byte': torch.onnx.TensorProtoDataType.UINT8, 'Char': torch.onnx.TensorProtoDataType.INT8, 'Double': torch.onnx.TensorProtoDataType.DOUBLE, 'Float': torch.onnx.TensorProtoDataType.FLOAT, 'Half': torch.onnx.TensorProtoDataType.FLOAT16, 'Int': torch.onnx.TensorProtoDataType.INT32, 'Long': torch.onnx.TensorProtoDataType.INT64, 'Short': torch.onnx.TensorProtoDataType.INT16, 'Bool': torch.onnx.TensorProtoDataType.BOOL, 'ComplexFloat': torch.onnx.TensorProtoDataType.COMPLEX64, 'ComplexDouble': torch.onnx.TensorProtoDataType.COMPLEX128, 'Undefined': torch.onnx.TensorProtoDataType.UNDEFINED}
scalar_name_to_pytorch = {'uint8_t': 'Byte', 'int8_t': 'Char', 'double': 'Double', 'float': 'Float', 'half': 'Half', 'int': 'Int', 'int64_t': 'Long', 'int16_t': 'Short', 'bool': 'Bool', 'complex64': '', 'complex128': ''}
scalar_type_to_pytorch_type = [torch.uint8, torch.int8, torch.short, torch.int, torch.int64, torch.half, torch.float, torch.double, torch.complex64, torch.complex128, torch.bool]

def _cast_func_template(to_i, g, input, non_blocking):
    return g.op('Cast', input, to_i=to_i)
scalar_type_to_onnx = [cast_pytorch_to_onnx['Byte'], cast_pytorch_to_onnx['Char'], cast_pytorch_to_onnx['Short'], cast_pytorch_to_onnx['Int'], cast_pytorch_to_onnx['Long'], cast_pytorch_to_onnx['Half'], cast_pytorch_to_onnx['Float'], cast_pytorch_to_onnx['Double'], cast_pytorch_to_onnx['Undefined'], cast_pytorch_to_onnx['ComplexFloat'], cast_pytorch_to_onnx['ComplexDouble'], cast_pytorch_to_onnx['Bool']]
_quantized_ops = set()

