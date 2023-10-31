from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
from torch._C import ListType, OptionalType
from torch.nn.modules.utils import _single, _pair, _triple
import torch.onnx
import torch.onnx.utils
from functools import partial
from functools import wraps
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _parse_arg, _unimplemented
import numpy
import math
import warnings

def unused(g):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.unused', 'unused(g)', {'OptionalType': OptionalType, 'g': g}, 1)

def _shape_as_tensor(g, input):
    return g.op('Shape', input)

def _reshape_from_tensor(g, input, shape):
    return g.op('Reshape', input, shape)

def reshape(g, self, shape):
    return view(g, self, shape)

def reshape_as(g, self, other):
    shape = g.op('Shape', other)
    return reshape(g, self, shape)

def add(g, self, other, alpha=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.add', 'add(g, self, other, alpha=None)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'g': g, 'self': self, 'other': other, 'alpha': alpha}, 1)

def sub(g, self, other, alpha=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.sub', 'sub(g, self, other, alpha=None)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'g': g, 'self': self, 'other': other, 'alpha': alpha}, 1)

def rsub(g, self, other, alpha=None):
    return sub(g, other, self, alpha=alpha)

def mul(g, self, other):
    return g.op('Mul', self, other)

def div(g, self, other):
    return g.op('Div', self, other)

def floor_divide(g, self, other):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.floor_divide', 'floor_divide(g, self, other)', {'div': div, 'sym_help': sym_help, 'g': g, 'self': self, 'other': other}, 1)

def true_divide(g, self, other):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.true_divide', 'true_divide(g, self, other)', {'sym_help': sym_help, 'div': div, 'torch': torch, 'g': g, 'self': self, 'other': other}, 1)

def reciprocal(g, self):
    return g.op('Div', torch.ones(1), self)

@parse_args('v', 'i')
def cat(g, tensor_list, dim):
    tensors = sym_help._unpack_list(tensor_list)
    return g.op('Concat', *tensors, axis_i=dim)

@parse_args('v', 'i')
def stack(g, tensor_list, dim):
    unsqueezed = [g.op('Unsqueeze', t, axes_i=[dim]) for t in sym_help._unpack_list(tensor_list)]
    return g.op('Concat', *unsqueezed, axis_i=dim)

def mm(g, self, other):
    C = g.op('Constant', value_t=torch.tensor([1]))
    return g.op('Gemm', self, other, C, beta_f=0.0, alpha_f=1.0)

def bmm(g, self, other):
    return g.op('MatMul', self, other)

def matmul(g, self, other):
    return g.op('MatMul', self, other)

@parse_args('v', 'v', 'v', 't', 't')
def addmm(g, self, mat1, mat2, beta, alpha):
    return g.op('Gemm', mat1, mat2, self, beta_f=sym_help._scalar(beta), alpha_f=sym_help._scalar(alpha))

def neg(g, self):
    return g.op('Neg', self)

def sqrt(g, self):
    return g.op('Sqrt', self)

def rsqrt(g, self):
    return div(g, sym_help._if_scalar_type_as(g, torch.ones(1), self), sqrt(g, self))

def tanh(g, self):
    return g.op('Tanh', self)

def sin(g, self):
    return g.op('Sin', self)

def cos(g, self):
    return g.op('Cos', self)

def tan(g, self):
    return g.op('Tan', self)

def asin(g, self):
    return g.op('Asin', self)

def acos(g, self):
    return g.op('Acos', self)

def atan(g, self):
    return g.op('Atan', self)

def sigmoid(g, self):
    return g.op('Sigmoid', self)

def sign(g, self):
    return g.op('Sign', self)

def _slice(g, input, axes, starts, ends):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._slice', '_slice(g, input, axes, starts, ends)', {'g': g, 'input': input, 'axes': axes, 'starts': starts, 'ends': ends}, 1)

def _reduce_op_symbolic(onnx_op_name, allow_multi_dim_support=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._reduce_op_symbolic', '_reduce_op_symbolic(onnx_op_name, allow_multi_dim_support=True)', {'sym_help': sym_help, 'onnx_op_name': onnx_op_name, 'allow_multi_dim_support': allow_multi_dim_support}, 1)

def overload_by_arg_count(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.overload_by_arg_count', 'overload_by_arg_count(fn)', {'wraps': wraps, 'fn': fn}, 1)

def _reduce_with_dtype(onnx_op, name, allow_multi_dim_support=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._reduce_with_dtype', '_reduce_with_dtype(onnx_op, name, allow_multi_dim_support=True)', {'_reduce_op_symbolic': _reduce_op_symbolic, 'overload_by_arg_count': overload_by_arg_count, 'parse_args': parse_args, '_unimplemented': _unimplemented, 'onnx_op': onnx_op, 'name': name, 'allow_multi_dim_support': allow_multi_dim_support}, 1)
sum = _reduce_with_dtype('ReduceSum', 'sum')
mean = _reduce_with_dtype('ReduceMean', 'mean')
prod = _reduce_with_dtype('ReduceProd', 'prod', allow_multi_dim_support=False)

@parse_args('v', 'i', 'none')
def cumsum(g, input, dim, dtype):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.cumsum', 'cumsum(g, input, dim, dtype)', {'_unimplemented': _unimplemented, 'name': name, 'parse_args': parse_args, 'g': g, 'input': input, 'dim': dim, 'dtype': dtype}, 1)

def _sample_dirichlet(g, self, generator):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._sample_dirichlet', '_sample_dirichlet(g, self, generator)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'g': g, 'self': self, 'generator': generator}, 1)

def _standard_gamma(g, self, generator):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._standard_gamma', '_standard_gamma(g, self, generator)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'g': g, 'self': self, 'generator': generator}, 1)

def t(g, self):
    return g.op('Transpose', self, perm_i=(1, 0))

def expand(g, self, size, implicit):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.expand', 'expand(g, self, size, implicit)', {'sym_help': sym_help, 'torch': torch, 'view': view, 'stack': stack, 'ones_like': ones_like, 'mul': mul, 'where': where, 'g': g, 'self': self, 'size': size, 'implicit': implicit}, 1)

def expand_as(g, self, other):
    shape = g.op('Shape', other)
    return g.op('Expand', self, shape)

def embedding(g, weight, indices, padding_idx, scale_grad_by_freq, sparse):
    return g.op('Gather', weight, indices)

@parse_args('v', 'v', 'v', 'i', 'i', 'i', 'v', 'i')
def embedding_bag(g, embedding_matrix, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.embedding_bag', 'embedding_bag(g, embedding_matrix, indices, offsets, scale_grad_by_freq, mode, sparse, per_sample_weights, include_last_offset)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'embedding_matrix': embedding_matrix, 'indices': indices, 'offsets': offsets, 'scale_grad_by_freq': scale_grad_by_freq, 'mode': mode, 'sparse': sparse, 'per_sample_weights': per_sample_weights, 'include_last_offset': include_last_offset}, 1)

def size(g, self, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.size', 'size(g, self, dim)', {'sym_help': sym_help, 'torch': torch, 'g': g, 'self': self, 'dim': dim}, 1)

@parse_args('v', 'i', 'i')
def transpose(g, self, dim0, dim1):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.transpose', 'transpose(g, self, dim0, dim1)', {'parse_args': parse_args, 'g': g, 'self': self, 'dim0': dim0, 'dim1': dim1}, 1)

@parse_args('v', 'is')
def permute(g, self, dims):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.permute', 'permute(g, self, dims)', {'parse_args': parse_args, 'g': g, 'self': self, 'dims': dims}, 1)

def view(g, self, size):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.view', 'view(g, self, size)', {'sym_help': sym_help, 'torch': torch, 'g': g, 'self': self, 'size': size}, 1)

def prim_ConstantSplit(g, self, split_size, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.prim_ConstantSplit', 'prim_ConstantSplit(g, self, split_size, dim)', {'g': g, 'self': self, 'split_size': split_size, 'dim': dim}, 1)

def prim_ConstantChunk(g, self, chunks, dim):
    split_size = (self.type().sizes()[dim] + chunks - 1) // chunks
    return prim_ConstantSplit(g, self, split_size, dim)

def split(g, self, split_size_or_sizes, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.split', 'split(g, self, split_size_or_sizes, dim)', {'sym_help': sym_help, 'split_with_sizes': split_with_sizes, 'g': g, 'self': self, 'split_size_or_sizes': split_size_or_sizes, 'dim': dim}, 1)

@parse_args('v', 'is', 'i')
def split_with_sizes(g, self, split_sizes, dim):
    return g.op('Split', self, split_i=split_sizes, axis_i=dim, outputs=1)

@parse_args('v', 'i')
def unbind(g, self, dim=0):
    return g.op('aten::unbind', self, axis_i=dim)

@parse_args('v', 'i', 'v')
def select(g, self, dim, index):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.select', 'select(g, self, dim, index)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim, 'index': index}, 1)

def squeeze(g, self, dim=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.squeeze', 'squeeze(g, self, dim=None)', {'sym_help': sym_help, 'warnings': warnings, '_unimplemented': _unimplemented, 'g': g, 'self': self, 'dim': dim}, 1)

def prelu(g, self, weight):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.prelu', 'prelu(g, self, weight)', {'g': g, 'self': self, 'weight': weight}, 1)

def relu(g, input):
    return g.op('Relu', input)

def ceil(g, input):
    return g.op('Ceil', input)

def floor(g, input):
    return g.op('Floor', input)

@parse_args('v', 't', 't')
def threshold(g, self, threshold, value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.threshold', 'threshold(g, self, threshold, value)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'parse_args': parse_args, 'g': g, 'self': self, 'threshold': threshold, 'value': value}, 1)

def leaky_relu(g, input, negative_slope, inplace=False):
    negative_slope = sym_help._get_const(negative_slope, 't', 'negative_slope')
    return g.op('LeakyRelu', input, alpha_f=sym_help._scalar(negative_slope))

@parse_args('v', 'i')
def glu(g, input, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.glu', 'glu(g, input, dim)', {'parse_args': parse_args, 'g': g, 'input': input, 'dim': dim}, 1)

@parse_args('v', 'i', 'none')
def softmax(g, input, dim, dtype=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.softmax', 'softmax(g, input, dim, dtype=None)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'dim': dim, 'dtype': dtype}, 1)

@parse_args('v', 't', 'v')
def softplus(g, self, beta, threshold):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.softplus', 'softplus(g, self, beta, threshold)', {'_unimplemented': _unimplemented, 'parse_args': parse_args, 'g': g, 'self': self, 'beta': beta, 'threshold': threshold}, 1)

def get_pool_ceil_padding(input, kernel_size, stride, padding):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.get_pool_ceil_padding', 'get_pool_ceil_padding(input, kernel_size, stride, padding)', {'math': math, 'input': input, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding}, 1)

def _max_pool(name, tuple_fn, ndims, return_indices):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._max_pool', '_max_pool(name, tuple_fn, ndims, return_indices)', {'parse_args': parse_args, '_unimplemented': _unimplemented, 'get_pool_ceil_padding': get_pool_ceil_padding, 'numpy': numpy, 'sym_help': sym_help, 'sub': sub, 'name': name, 'tuple_fn': tuple_fn, 'ndims': ndims, 'return_indices': return_indices}, 1)
max_pool1d = _max_pool('max_pool1d', _single, 1, return_indices=False)
max_pool2d = _max_pool('max_pool2d', _pair, 2, return_indices=False)
max_pool3d = _max_pool('max_pool3d', _triple, 3, return_indices=False)
max_pool1d_with_indices = _max_pool('max_pool1d_with_indices', _single, 1, return_indices=True)
max_pool2d_with_indices = _max_pool('max_pool2d_with_indices', _pair, 2, return_indices=True)
max_pool3d_with_indices = _max_pool('max_pool3d_with_indices', _triple, 3, return_indices=True)

def _avg_pool(name, tuple_fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._avg_pool', '_avg_pool(name, tuple_fn)', {'parse_args': parse_args, '_unimplemented': _unimplemented, 'sym_help': sym_help, 'get_pool_ceil_padding': get_pool_ceil_padding, 'numpy': numpy, 'name': name, 'tuple_fn': tuple_fn}, 1)
avg_pool1d = _avg_pool('avg_pool1d', _single)
avg_pool2d = _avg_pool('avg_pool2d', _pair)
avg_pool3d = _avg_pool('avg_pool3d', _triple)

def _adaptive_pool(name, type, tuple_fn, fn=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._adaptive_pool', '_adaptive_pool(name, type, tuple_fn, fn=None)', {'parse_args': parse_args, '_unimplemented': _unimplemented, 'name': name, 'type': type, 'tuple_fn': tuple_fn, 'fn': fn}, 1)
adaptive_avg_pool1d = _adaptive_pool('adaptive_avg_pool1d', 'AveragePool', _single)
adaptive_avg_pool2d = _adaptive_pool('adaptive_avg_pool2d', 'AveragePool', _pair)
adaptive_avg_pool3d = _adaptive_pool('adaptive_avg_pool3d', 'AveragePool', _triple)
adaptive_max_pool1d = _adaptive_pool('adaptive_max_pool1d', 'MaxPool', _single, max_pool1d_with_indices)
adaptive_max_pool2d = _adaptive_pool('adaptive_max_pool2d', 'MaxPool', _pair, max_pool2d_with_indices)
adaptive_max_pool3d = _adaptive_pool('adaptive_max_pool3d', 'MaxPool', _triple, max_pool3d_with_indices)

def _prepare_onnx_paddings(dim, pad):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._prepare_onnx_paddings', '_prepare_onnx_paddings(dim, pad)', {'dim': dim, 'pad': pad}, 1)

@parse_args('v', 'is', 'f')
def constant_pad_nd(g, input, padding, value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.constant_pad_nd', 'constant_pad_nd(g, input, padding, value)', {'_prepare_onnx_paddings': _prepare_onnx_paddings, 'parse_args': parse_args, 'g': g, 'input': input, 'padding': padding, 'value': value}, 1)

@parse_args('v', 'is')
def reflection_pad(g, input, padding):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.reflection_pad', 'reflection_pad(g, input, padding)', {'_prepare_onnx_paddings': _prepare_onnx_paddings, 'parse_args': parse_args, 'g': g, 'input': input, 'padding': padding}, 1)

@parse_args('v', 'is')
def replication_pad(g, input, padding):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.replication_pad', 'replication_pad(g, input, padding)', {'_prepare_onnx_paddings': _prepare_onnx_paddings, 'parse_args': parse_args, 'g': g, 'input': input, 'padding': padding}, 1)
reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad
replication_pad1d = replication_pad
replication_pad2d = replication_pad
replication_pad3d = replication_pad

def _interpolate(name, dim, interpolate_mode):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._interpolate', '_interpolate(name, dim, interpolate_mode)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'name': name, 'dim': dim, 'interpolate_mode': interpolate_mode}, 1)
upsample_nearest1d = _interpolate('upsample_nearest1d', 3, 'nearest')
upsample_nearest2d = _interpolate('upsample_nearest2d', 4, 'nearest')
upsample_nearest3d = _interpolate('upsample_nearest3d', 5, 'nearest')
upsample_linear1d = _interpolate('upsample_linear1d', 3, 'linear')
upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, 'linear')
upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, 'linear')

def __interpolate(g, input, size, scale_factor, mode, align_corners, recompute_scale_factor):
    (scales, mode) = sym_help._interpolate_get_scales_and_mode(g, input, size, scale_factor, mode, align_corners)
    return g.op('Upsample', input, scales, mode_s=mode)

@parse_args('v')
def bitwise_not(g, inp):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.bitwise_not', 'bitwise_not(g, inp)', {'_unimplemented': _unimplemented, 'parse_args': parse_args, 'g': g, 'inp': inp}, 1)

def wrap_logical_op_with_cast_to(to_type):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.wrap_logical_op_with_cast_to', 'wrap_logical_op_with_cast_to(to_type)', {'sym_help': sym_help, 'to_type': to_type}, 1)

def wrap_logical_op_with_cast_to_and_from(to_type):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.wrap_logical_op_with_cast_to_and_from', 'wrap_logical_op_with_cast_to_and_from(to_type)', {'wrap_logical_op_with_cast_to': wrap_logical_op_with_cast_to, 'to_type': to_type}, 1)

def wrap_logical_op_with_negation(func):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.wrap_logical_op_with_negation', 'wrap_logical_op_with_negation(func)', {'func': func}, 1)

def eq(g, self, other):
    return g.op('Equal', self, other)

@wrap_logical_op_with_negation
def ne(g, self, other):
    return g.op('Equal', self, other)

def gt(g, input, other):
    return gt_impl(g, input, other)

def gt_impl(g, input, other):
    return g.op('Greater', input, other)

def lt(g, input, other):
    return lt_impl(g, input, other)

def lt_impl(g, input, other):
    return g.op('Less', input, other)

@wrap_logical_op_with_negation
def ge(g, input, other):
    return lt_impl(g, input, other)

@wrap_logical_op_with_negation
def le(g, input, other):
    return gt_impl(g, input, other)

@wrap_logical_op_with_cast_to_and_from('Bool')
def __and_(g, input, other):
    return g.op('And', input, other)

@wrap_logical_op_with_cast_to_and_from('Bool')
def __or_(g, input, other):
    return g.op('Or', input, other)

def __rshift_(g, self, other):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.__rshift_', '__rshift_(g, self, other)', {'sym_help': sym_help, 'torch': torch, 'g': g, 'self': self, 'other': other}, 1)

def __lshift_(g, self, other):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.__lshift_', '__lshift_(g, self, other)', {'sym_help': sym_help, 'torch': torch, 'g': g, 'self': self, 'other': other}, 1)

def where(g, condition, self, other):
    return g.op('Where', condition, self, other)

@parse_args('v', 'i', 'none')
def log_softmax(g, input, dim, dtype=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.log_softmax', 'log_softmax(g, input, dim, dtype=None)', {'_unimplemented': _unimplemented, 'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'dim': dim, 'dtype': dtype}, 1)

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'is', 'i', 'i', 'i', 'i')
def _convolution(g, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._convolution', '_convolution(g, input, weight, bias, stride, padding, dilation, transposed, output_padding, groups, benchmark, deterministic, cudnn_enabled)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'weight': weight, 'bias': bias, 'stride': stride, 'padding': padding, 'dilation': dilation, 'transposed': transposed, 'output_padding': output_padding, 'groups': groups, 'benchmark': benchmark, 'deterministic': deterministic, 'cudnn_enabled': cudnn_enabled}, 1)

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i')
def conv1d(g, input, weight, bias, stride, padding, dilation, groups):
    return _convolution(g, input, weight, bias, stride, padding, dilation, False, (), groups, None, None, None)

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i')
def conv2d(g, input, weight, bias, stride, padding, dilation, groups):
    return _convolution(g, input, weight, bias, stride, padding, dilation, False, (), groups, None, None, None)

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i')
def conv3d(g, input, weight, bias, stride, padding, dilation, groups):
    return _convolution(g, input, weight, bias, stride, padding, dilation, False, (), groups, None, None, None)

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'is')
def conv_transpose1d(g, input, weight, bias, stride, padding, output_padding, groups, dilation):
    return _convolution(g, input, weight, bias, stride, padding, dilation, True, output_padding, groups, None, None, None)

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'is')
def conv_transpose2d(g, input, weight, bias, stride, padding, output_padding, groups, dilation):
    return _convolution(g, input, weight, bias, stride, padding, dilation, True, output_padding, groups, None, None, None)

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'is')
def conv_transpose3d(g, input, weight, bias, stride, padding, output_padding, groups, dilation):
    return _convolution(g, input, weight, bias, stride, padding, dilation, True, output_padding, groups, None, None, None)

@parse_args('v', 'v', 'v', 'v', 'v', 'i', 'f', 'f', 'i')
def batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.batch_norm', 'batch_norm(g, input, weight, bias, running_mean, running_var, training, momentum, eps, cudnn_enabled)', {'sym_help': sym_help, 'torch': torch, 'parse_args': parse_args, 'g': g, 'input': input, 'weight': weight, 'bias': bias, 'running_mean': running_mean, 'running_var': running_var, 'training': training, 'momentum': momentum, 'eps': eps, 'cudnn_enabled': cudnn_enabled}, 1)

@parse_args('v', 'is', 'v', 'v', 'f', 'i')
def layer_norm(g, input, normalized_shape, weight, bias, eps, cudnn_enable):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.layer_norm', 'layer_norm(g, input, normalized_shape, weight, bias, eps, cudnn_enable)', {'sym_help': sym_help, 'torch': torch, 'sub': sub, 'sqrt': sqrt, 'div': div, 'mul': mul, 'parse_args': parse_args, 'g': g, 'input': input, 'normalized_shape': normalized_shape, 'weight': weight, 'bias': bias, 'eps': eps, 'cudnn_enable': cudnn_enable}, 1)

@parse_args('v', 'v', 'v', 'v', 'v', 'i', 'f', 'f', 'i')
def instance_norm(g, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.instance_norm', 'instance_norm(g, input, weight, bias, running_mean, running_var, use_input_stats, momentum, eps, cudnn_enabled)', {'sym_help': sym_help, 'torch': torch, 'parse_args': parse_args, 'g': g, 'input': input, 'weight': weight, 'bias': bias, 'running_mean': running_mean, 'running_var': running_var, 'use_input_stats': use_input_stats, 'momentum': momentum, 'eps': eps, 'cudnn_enabled': cudnn_enabled}, 1)

@parse_args('v', 'i', 'i', 'i')
def unfold(g, input, dimension, size, step):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.unfold', 'unfold(g, input, dimension, size, step)', {'sym_help': sym_help, 'torch': torch, '_unimplemented': _unimplemented, 'parse_args': parse_args, 'g': g, 'input': input, 'dimension': dimension, 'size': size, 'step': step}, 1)

@parse_args('v', 't', 't', 't')
def elu(g, input, alpha, scale, input_scale):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.elu', 'elu(g, input, alpha, scale, input_scale)', {'_unimplemented': _unimplemented, 'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'alpha': alpha, 'scale': scale, 'input_scale': input_scale}, 1)

def selu(g, input):
    return g.op('Selu', input)

@parse_args('v', 'i', 'v')
def index_select(g, self, dim, index):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.index_select', 'index_select(g, self, dim, index)', {'sym_help': sym_help, 'torch': torch, 'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim, 'index': index}, 1)

def index_put(g, self, indices_list_value, values, accumulate):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.index_put', 'index_put(g, self, indices_list_value, values, accumulate)', {'sym_help': sym_help, 'g': g, 'self': self, 'indices_list_value': indices_list_value, 'values': values, 'accumulate': accumulate}, 1)

def index_fill(g, self, dim, index, value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.index_fill', 'index_fill(g, self, dim, index, value)', {'sym_help': sym_help, 'torch': torch, 'expand': expand, 'scatter': scatter, 'g': g, 'self': self, 'dim': dim, 'index': index, 'value': value}, 1)

def index_copy(g, self, dim, index, source):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.index_copy', 'index_copy(g, self, dim, index, source)', {'sym_help': sym_help, 'torch': torch, 'scatter': scatter, 'g': g, 'self': self, 'dim': dim, 'index': index, 'source': source}, 1)

def type_as(g, self, other):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.type_as', 'type_as(g, self, other)', {'sym_help': sym_help, 'g': g, 'self': self, 'other': other}, 1)

@parse_args('v', 'v', 'i', 'f')
def cosine_similarity(g, x1, x2, dim, eps):
    return g.op('ATen', x1, x2, dim_i=dim, eps_f=eps, operator_s='cosine_similarity')

def clone(g, input, unused_memory_format):
    return input

def abs(g, self):
    return g.op('Abs', self)

def log(g, self):
    return g.op('Log', self)

def log1p(g, self):
    return log(g, add(g, sym_help._if_scalar_type_as(g, torch.ones(1), self), self))

def pow(g, self, exponent):
    return g.op('Pow', self, exponent)

def clamp(g, self, min, max):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.clamp', 'clamp(g, self, min, max)', {'sym_help': sym_help, 'clamp_max': clamp_max, 'clamp_min': clamp_min, '_parse_arg': _parse_arg, 'g': g, 'self': self, 'min': min, 'max': max}, 1)

@parse_args('v', 'f')
def clamp_min(g, self, min):
    return g.op('Clip', self, min_f=min)

@parse_args('v', 'f')
def clamp_max(g, self, max):
    return g.op('Clip', self, max_f=max)

def max(g, self, dim_or_y=None, keepdim=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.max', 'max(g, self, dim_or_y=None, keepdim=None)', {'sym_help': sym_help, 'g': g, 'self': self, 'dim_or_y': dim_or_y, 'keepdim': keepdim}, 1)

def min(g, self, dim_or_y=None, keepdim=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.min', 'min(g, self, dim_or_y=None, keepdim=None)', {'sym_help': sym_help, 'g': g, 'self': self, 'dim_or_y': dim_or_y, 'keepdim': keepdim}, 1)

def exp(g, self):
    return g.op('Exp', self)

@parse_args('v', 'f', 'i')
def dropout(g, input, p, train):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.dropout', 'dropout(g, input, p, train)', {'warnings': warnings, 'parse_args': parse_args, 'g': g, 'input': input, 'p': p, 'train': train}, 1)

def _unsupported_dropout(name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._unsupported_dropout', '_unsupported_dropout(name)', {'parse_args': parse_args, '_unimplemented': _unimplemented, 'name': name}, 1)
feature_dropout = _unsupported_dropout('feature_dropout')
alpha_dropout = _unsupported_dropout('alpha_dropout')
feature_alpha_dropout = _unsupported_dropout('feature_alpha_dropout')
dropout_ = dropout
feature_dropout_ = feature_dropout
alpha_dropout_ = alpha_dropout
feature_alpha_dropout_ = feature_alpha_dropout

@parse_args('v', 't', 'is', 'i')
def norm(g, self, p, dim, keepdim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.norm', 'norm(g, self, p, dim, keepdim)', {'_reduce_op_symbolic': _reduce_op_symbolic, 'parse_args': parse_args, 'g': g, 'self': self, 'p': p, 'dim': dim, 'keepdim': keepdim}, 1)

@parse_args('v', 'v', 'v', 'i')
def conv_tbc(g, input, weight, bias, pad):
    return g.op('ATen', input, weight, bias, operator_s='conv_tbc', pad_i=pad)

@parse_args('v', 'i', 'i')
def _unique(g, input, sorted, return_inverse):
    return g.op('ATen', input, operator_s='_unique', sorted_i=sorted, return_inverse_i=return_inverse, outputs=2)

@parse_args('v', 'i', 'i', 'i')
def _unique2(g, input, sorted, return_inverse, return_counts):
    return g.op('ATen', input, operator_s='_unique2', sorted_i=sorted, return_inverse_i=return_inverse, return_counts_i=return_counts, outputs=3)
for (k, v) in sym_help.cast_pytorch_to_onnx.items():
    name = '_cast_{}'.format(k)
    globals()[name] = parse_args('v', 'i')(partial(sym_help._cast_func_template, v))

@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def empty(g, sizes, dtype, layout, device, pin_memory=False, memory_format=None):
    return zeros(g, sizes, dtype, layout, device, pin_memory)

@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def empty_like(g, input, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    return zeros_like(g, input, dtype, layout, device, pin_memory)

def scalar_tensor(g, scalar, dtype, *options):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.scalar_tensor', 'scalar_tensor(g, scalar, dtype, *options)', {'sym_help': sym_help, 'g': g, 'scalar': scalar, 'dtype': dtype, 'options': options}, 1)

@parse_args('v', 'i', 'v', 'v', 'v')
def zeros(g, sizes, dtype, layout, device, pin_memory=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.zeros', 'zeros(g, sizes, dtype, layout, device, pin_memory=False)', {'torch': torch, 'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'sizes': sizes, 'dtype': dtype, 'layout': layout, 'device': device, 'pin_memory': pin_memory}, 1)

@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def zeros_like(g, input, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.zeros_like', 'zeros_like(g, input, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None)', {'torch': torch, 'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'dtype': dtype, 'layout': layout, 'device': device, 'pin_memory': pin_memory, 'memory_format': memory_format}, 1)

@parse_args('v', 'v', 'i', 'v', 'v', 'v')
def new_zeros(g, self, sizes, dtype, layout, device, pin_memory=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.new_zeros', 'new_zeros(g, self, sizes, dtype, layout, device, pin_memory=False)', {'sym_help': sym_help, 'zeros': zeros, 'parse_args': parse_args, 'g': g, 'self': self, 'sizes': sizes, 'dtype': dtype, 'layout': layout, 'device': device, 'pin_memory': pin_memory}, 1)

@parse_args('v', 'i', 'v', 'v', 'v')
def ones(g, sizes, dtype, layout, device, pin_memory=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.ones', 'ones(g, sizes, dtype, layout, device, pin_memory=False)', {'torch': torch, 'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'sizes': sizes, 'dtype': dtype, 'layout': layout, 'device': device, 'pin_memory': pin_memory}, 1)

@parse_args('v', 'i', 'v', 'v', 'v', 'v')
def ones_like(g, input, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.ones_like', 'ones_like(g, input, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None)', {'torch': torch, 'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'dtype': dtype, 'layout': layout, 'device': device, 'pin_memory': pin_memory, 'memory_format': memory_format}, 1)

def full(g, sizes, value, dtype, layout, device, pin_memory=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.full', 'full(g, sizes, value, dtype, layout, device, pin_memory=False)', {'sym_help': sym_help, 'zeros': zeros, 'torch': torch, 'g': g, 'sizes': sizes, 'value': value, 'dtype': dtype, 'layout': layout, 'device': device, 'pin_memory': pin_memory}, 1)

@parse_args('v', 'f', 'i', 'v', 'v', 'v', 'v')
def full_like(g, input, fill_value, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.full_like', 'full_like(g, input, fill_value, dtype=None, layout=None, device=None, pin_memory=False, memory_format=None)', {'torch': torch, 'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'fill_value': fill_value, 'dtype': dtype, 'layout': layout, 'device': device, 'pin_memory': pin_memory, 'memory_format': memory_format}, 1)

@parse_args('v', 'v', 'v', 'v', 'i')
def slice(g, self, dim, start, end, step):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.slice', 'slice(g, self, dim, start, end, step)', {'_unimplemented': _unimplemented, 'sym_help': sym_help, 'torch': torch, '_parse_arg': _parse_arg, 'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim, 'start': start, 'end': end, 'step': step}, 1)

@parse_args('v', 'f', 'f')
def hardtanh(g, self, min_val, max_val):
    return g.op('Clip', self, min_f=min_val, max_f=max_val)

def alias(g, self):
    return self

@parse_args('v', 'i')
def unsqueeze(g, self, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.unsqueeze', 'unsqueeze(g, self, dim)', {'warnings': warnings, '_unimplemented': _unimplemented, 'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim}, 1)

@parse_args('v', 'i', 'i', 'none')
def sort(g, self, dim, decending, out=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.sort', 'sort(g, self, dim, decending, out=None)', {'_unimplemented': _unimplemented, 'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim, 'decending': decending, 'out': out}, 1)

@parse_args('v', 'i', 'i', 'i', 'i', 'none')
def topk(g, self, k, dim, largest, sorted, out=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.topk', 'topk(g, self, k, dim, largest, sorted, out=None)', {'_unimplemented': _unimplemented, 'parse_args': parse_args, 'g': g, 'self': self, 'k': k, 'dim': dim, 'largest': largest, 'sorted': sorted, 'out': out}, 1)

def to(g, self, *args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.to', 'to(g, self, *args)', {'ListType': ListType, 'sym_help': sym_help, 'g': g, 'self': self, 'args': args}, 1)

def repeat(g, self, repeats):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.repeat', 'repeat(g, self, repeats)', {'sym_help': sym_help, 'torch': torch, 'view': view, 'g': g, 'self': self, 'repeats': repeats}, 1)

@parse_args('v', 'i')
def pixel_shuffle(g, self, upscale_factor):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.pixel_shuffle', 'pixel_shuffle(g, self, upscale_factor)', {'_unimplemented': _unimplemented, 'view': view, 'parse_args': parse_args, 'g': g, 'self': self, 'upscale_factor': upscale_factor}, 1)

def _generic_rnn(g, variant, input, initial_states, all_weights, has_biases, num_layers, dropout, train, bidirectional, batch_first=None, batch_sizes=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._generic_rnn', '_generic_rnn(g, variant, input, initial_states, all_weights, has_biases, num_layers, dropout, train, bidirectional, batch_first=None, batch_sizes=None)', {'warnings': warnings, '_unimplemented': _unimplemented, 'unused': unused, 'sym_help': sym_help, 'torch': torch, 'g': g, 'variant': variant, 'input': input, 'initial_states': initial_states, 'all_weights': all_weights, 'has_biases': has_biases, 'num_layers': num_layers, 'dropout': dropout, 'train': train, 'bidirectional': bidirectional, 'batch_first': batch_first, 'batch_sizes': batch_sizes}, 1)

@parse_args('v', 'v', 'v', 'i', 'i', 'f', 'i', 'i', 'i')
def _lstm_full(g, input, hidden_v, weight_v, has_biases, num_layers, dropout, train, bidirectional, batch_first):
    (hidden, weight) = (sym_help._unpack_list(hidden_v), sym_help._unpack_list(weight_v))
    return _generic_rnn(g, 'LSTM', input, hidden, weight, has_biases, num_layers, dropout, train, bidirectional, batch_first)

@parse_args('v', 'v', 'v', 'v', 'i', 'i', 'f', 'i', 'i')
def _lstm_packed(g, input, batch_sizes, hidden_v, weight_v, has_biases, num_layers, dropout, train, bidirectional):
    (hidden, weight) = (sym_help._unpack_list(hidden_v), sym_help._unpack_list(weight_v))
    return _generic_rnn(g, 'LSTM', input, hidden, weight, has_biases, num_layers, dropout, train, bidirectional, batch_sizes=batch_sizes)

def lstm(g, *args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.lstm', 'lstm(g, *args)', {'sym_help': sym_help, '_lstm_packed': _lstm_packed, '_lstm_full': _lstm_full, 'g': g, 'args': args}, 1)

def _one_hidden_rnn(kind):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._one_hidden_rnn', '_one_hidden_rnn(kind)', {'parse_args': parse_args, 'sym_help': sym_help, '_generic_rnn': _generic_rnn, 'kind': kind}, 1)
gru = _one_hidden_rnn('GRU')
rnn_tanh = _one_hidden_rnn('RNN_TANH')
rnn_relu = _one_hidden_rnn('RNN_RELU')

@parse_args('v', 'i')
def _dim_arange(g, like, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._dim_arange', '_dim_arange(g, like, dim)', {'torch': torch, 'sym_help': sym_help, 'arange': arange, 'parse_args': parse_args, 'g': g, 'like': like, 'dim': dim}, 1)

def detach(g, input):
    return input

@parse_args('v', 'i')
def contiguous(g, input, memory_format):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.contiguous', 'contiguous(g, input, memory_format)', {'parse_args': parse_args, 'g': g, 'input': input, 'memory_format': memory_format}, 1)

@parse_args('v', 'v', 'i')
def _pack_padded_sequence(g, input, lengths, batch_first):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._pack_padded_sequence', '_pack_padded_sequence(g, input, lengths, batch_first)', {'torch': torch, '_cast_Int': _cast_Int, 'parse_args': parse_args, 'g': g, 'input': input, 'lengths': lengths, 'batch_first': batch_first}, 1)

@parse_args('v', 'v', 'i', 't', 'v')
def _pad_packed_sequence(g, data, batch_sizes, batch_first, padding_value, total_length):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._pad_packed_sequence', '_pad_packed_sequence(g, data, batch_sizes, batch_first, padding_value, total_length)', {'parse_args': parse_args, 'g': g, 'data': data, 'batch_sizes': batch_sizes, 'batch_first': batch_first, 'padding_value': padding_value, 'total_length': total_length}, 2)

def randn(g, shapes, dtype, *options):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.randn', 'randn(g, shapes, dtype, *options)', {'sym_help': sym_help, 'torch': torch, 'g': g, 'shapes': shapes, 'dtype': dtype, 'options': options}, 1)

def rand(g, shapes, dtype, *options):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.rand', 'rand(g, shapes, dtype, *options)', {'sym_help': sym_help, 'torch': torch, 'g': g, 'shapes': shapes, 'dtype': dtype, 'options': options}, 1)

def randn_like(g, self, dtype, layout=None, device=None, pin_memory=False, memory_format=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.randn_like', 'randn_like(g, self, dtype, layout=None, device=None, pin_memory=False, memory_format=None)', {'sym_help': sym_help, 'g': g, 'self': self, 'dtype': dtype, 'layout': layout, 'device': device, 'pin_memory': pin_memory, 'memory_format': memory_format}, 1)

def rand_like(g, self, dtype, layout=None, device=None, pin_memory=False, memory_format=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.rand_like', 'rand_like(g, self, dtype, layout=None, device=None, pin_memory=False, memory_format=None)', {'sym_help': sym_help, 'g': g, 'self': self, 'dtype': dtype, 'layout': layout, 'device': device, 'pin_memory': pin_memory, 'memory_format': memory_format}, 1)

@parse_args('v', 'f', 'f', 'i', 'none')
def rrelu(g, input, lower, upper, training, generator):
    p = g.op('RandomUniformLike', input, high_f=upper, low_f=lower)
    return g.op('PRelu', input, p)

@parse_args('v')
def log_sigmoid(g, input):
    p = g.op('Sigmoid', input)
    return g.op('Log', p)

@parse_args('v')
def erf(g, input):
    return g.op('Erf', input)

@parse_args('v', 'i', 'i')
def flatten(g, input, start_dim, end_dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.flatten', 'flatten(g, input, start_dim, end_dim)', {'_unimplemented': _unimplemented, 'torch': torch, '_reshape_from_tensor': _reshape_from_tensor, 'parse_args': parse_args, 'g': g, 'input': input, 'start_dim': start_dim, 'end_dim': end_dim}, 1)

@parse_args('v')
def nonzero(g, input):
    return t(g, g.op('NonZero', input))

@parse_args('v')
def isnan(g, input):
    output = g.op('IsNaN', input)
    return output

@parse_args('v', 'i', 'i', 'i')
def narrow(g, input, dim, start, length):
    return sym_help._slice_helper(g, input, axes=[dim], starts=[start], ends=[start + length])

def argmax(g, input, dim, keepdim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.argmax', 'argmax(g, input, dim, keepdim)', {'sym_help': sym_help, 'reshape': reshape, '_parse_arg': _parse_arg, 'g': g, 'input': input, 'dim': dim, 'keepdim': keepdim}, 1)

def argmin(g, input, dim, keepdim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.argmin', 'argmin(g, input, dim, keepdim)', {'sym_help': sym_help, 'reshape': reshape, '_parse_arg': _parse_arg, 'g': g, 'input': input, 'dim': dim, 'keepdim': keepdim}, 1)

@parse_args('v', 'i', 'v', 'v')
def scatter(g, self, dim, index, src):
    return g.op('Scatter', self, index, src, axis_i=dim)

@parse_args('v', 'i', 'v', 'v')
def scatter_add(g, self, dim, index, src):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.scatter_add', 'scatter_add(g, self, dim, index, src)', {'_unimplemented': _unimplemented, 'sym_help': sym_help, 'torch': torch, 'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim, 'index': index, 'src': src}, 1)

def log2(g, self):
    _ln2 = 0.6931471805599453
    return g.op('Div', log(g, self), g.op('Constant', value_t=torch.Tensor([_ln2])))

def prim_shape(g, self):
    return g.op('Shape', self)

@parse_args('v', 'i')
def one_hot(g, self, num_classes):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.one_hot', 'one_hot(g, self, num_classes)', {'torch': torch, 'parse_args': parse_args, 'g': g, 'self': self, 'num_classes': num_classes}, 1)

@parse_args('v', 'i', 'v', 'v')
def gather(g, self, dim, index, sparse_grad=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.gather', 'gather(g, self, dim, index, sparse_grad=False)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'torch': torch, 'size': size, 'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim, 'index': index, 'sparse_grad': sparse_grad}, 1)

@parse_args('v', 'is', 'b', 'i')
def _std(g, input, dim, unbiased, keepdim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._std', '_std(g, input, dim, unbiased, keepdim)', {'numpy': numpy, 'torch': torch, '_unimplemented': _unimplemented, 'parse_args': parse_args, 'g': g, 'input': input, 'dim': dim, 'unbiased': unbiased, 'keepdim': keepdim}, 1)

def std(g, input, *args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.std', 'std(g, input, *args)', {'ListType': ListType, '_std': _std, 'g': g, 'input': input, 'args': args}, 1)

@parse_args('v', 'is', 'i')
def logsumexp(g, input, dim, keepdim):
    return g.op('ReduceLogSumExp', input, axes_i=dim, keepdims_i=keepdim)

def arange(g, *args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.arange', 'arange(g, *args)', {'sym_help': sym_help, 'torch': torch, 'nonzero': nonzero, 'ones': ones, 'g': g, 'args': args}, 1)

def masked_fill(g, self, mask, value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.masked_fill', 'masked_fill(g, self, mask, value)', {'_cast_Bool': _cast_Bool, 'sym_help': sym_help, 'g': g, 'self': self, 'mask': mask, 'value': value}, 1)

def index(g, self, index):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.index', 'index(g, self, index)', {'sym_help': sym_help, 'torch': torch, 'warnings': warnings, 'squeeze': squeeze, 'nonzero': nonzero, 'index_select': index_select, '_shape_as_tensor': _shape_as_tensor, 'g': g, 'self': self, 'index': index}, 1)

@parse_args('v', 'is', 'i')
def frobenius_norm(g, self, dim=None, keepdim=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.frobenius_norm', 'frobenius_norm(g, self, dim=None, keepdim=False)', {'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim, 'keepdim': keepdim}, 1)

@parse_args('v', 'i', 'b', 'v')
def multinomial(g, input, num_samples, replacement=False, generator=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.multinomial', 'multinomial(g, input, num_samples, replacement=False, generator=None)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'log': log, 'parse_args': parse_args, 'g': g, 'input': input, 'num_samples': num_samples, 'replacement': replacement, 'generator': generator}, 1)

def baddbmm(g, self, batch1, batch2, beta, alpha):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.baddbmm', 'baddbmm(g, self, batch1, batch2, beta, alpha)', {'matmul': matmul, 'mul': mul, 'sym_help': sym_help, 'g': g, 'self': self, 'batch1': batch1, 'batch2': batch2, 'beta': beta, 'alpha': alpha}, 1)

def meshgrid(g, tensor_list):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.meshgrid', 'meshgrid(g, tensor_list)', {'view': view, 'torch': torch, 'sym_help': sym_help, '_reshape_from_tensor': _reshape_from_tensor, 'g': g, 'tensor_list': tensor_list}, 1)

def remainder(g, input, other):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.remainder', 'remainder(g, input, other)', {'sym_help': sym_help, 'g': g, 'input': input, 'other': other}, 1)

def gelu(g, self):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.gelu', 'gelu(g, self)', {'div': div, 'torch': torch, 'mul': mul, 'g': g, 'self': self}, 1)

@parse_args('v', 'i', 'v', 'v', 'f', 'i')
def group_norm(g, input, num_groups, weight, bias, eps, cudnn_enabled):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.group_norm', 'group_norm(g, input, num_groups, weight, bias, eps, cudnn_enabled)', {'sym_help': sym_help, 'torch': torch, 'mul': mul, 'parse_args': parse_args, 'g': g, 'input': input, 'num_groups': num_groups, 'weight': weight, 'bias': bias, 'eps': eps, 'cudnn_enabled': cudnn_enabled}, 1)

@parse_args('v', 'v', 'i')
def _weight_norm(g, weight_v, weight_g, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9._weight_norm', '_weight_norm(g, weight_v, weight_g, dim)', {'norm': norm, 'parse_args': parse_args, 'g': g, 'weight_v': weight_v, 'weight_g': weight_g, 'dim': dim}, 1)

def dim(g, self):
    """Implement the dim functionality available for a pytorch tensor in ONNX"""
    shape = g.op('Shape', self)
    return g.op('Size', shape)

def __getitem_(g, self, i):
    return select(g, self, g.op('Constant', value_t=torch.tensor([0])), i)

def take(g, self, index):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset9.take', 'take(g, self, index)', {'torch': torch, 'index_select': index_select, 'reshape_as': reshape_as, 'g': g, 'self': self, 'index': index}, 1)

