from __future__ import absolute_import, division, print_function, unicode_literals
from sys import maxsize
import torch
import torch.onnx.symbolic_helper as sym_help
import warnings
import numpy
from torch.onnx.symbolic_helper import parse_args, _unimplemented
from torch.onnx.symbolic_opset9 import expand
from torch.nn.modules.utils import _single, _pair, _triple

@parse_args('v', 'f', 'f')
def hardtanh(g, self, min_val, max_val):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.hardtanh', 'hardtanh(g, self, min_val, max_val)', {'sym_help': sym_help, 'torch': torch, 'parse_args': parse_args, 'g': g, 'self': self, 'min_val': min_val, 'max_val': max_val}, 1)

def clamp(g, self, min, max):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.clamp', 'clamp(g, self, min, max)', {'sym_help': sym_help, 'g': g, 'self': self, 'min': min, 'max': max}, 1)

def index_put(g, self, indices_list_value, values, accumulate=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.index_put', 'index_put(g, self, indices_list_value, values, accumulate=False)', {'sym_help': sym_help, 'torch': torch, 'maxsize': maxsize, 'g': g, 'self': self, 'indices_list_value': indices_list_value, 'values': values, 'accumulate': accumulate}, 1)

@parse_args('v', 'i')
def pixel_shuffle(g, self, upscale_factor):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.pixel_shuffle', 'pixel_shuffle(g, self, upscale_factor)', {'_unimplemented': _unimplemented, 'parse_args': parse_args, 'g': g, 'self': self, 'upscale_factor': upscale_factor}, 1)

def _interpolate(name, dim, interpolate_mode):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11._interpolate', '_interpolate(name, dim, interpolate_mode)', {'sym_help': sym_help, 'torch': torch, 'name': name, 'dim': dim, 'interpolate_mode': interpolate_mode}, 1)
upsample_nearest1d = _interpolate('upsample_nearest1d', 3, 'nearest')
upsample_nearest2d = _interpolate('upsample_nearest2d', 4, 'nearest')
upsample_nearest3d = _interpolate('upsample_nearest3d', 5, 'nearest')
upsample_linear1d = _interpolate('upsample_linear1d', 3, 'linear')
upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, 'linear')
upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, 'linear')
upsample_bicubic2d = _interpolate('upsample_bicubic2d', 4, 'cubic')

def __interpolate(g, input, size, scale_factor, mode, align_corners, recompute_scale_factor):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.__interpolate', '__interpolate(g, input, size, scale_factor, mode, align_corners, recompute_scale_factor)', {'sym_help': sym_help, 'torch': torch, 'warnings': warnings, 'unsqueeze': unsqueeze, 'g': g, 'input': input, 'size': size, 'scale_factor': scale_factor, 'mode': mode, 'align_corners': align_corners, 'recompute_scale_factor': recompute_scale_factor}, 1)

@parse_args('v', 'i', 'v', 'v')
def gather(g, self, dim, index, sparse_grad=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.gather', 'gather(g, self, dim, index, sparse_grad=False)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'torch': torch, 'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim, 'index': index, 'sparse_grad': sparse_grad}, 1)

@parse_args('v', 'i', 'v', 'v')
def scatter(g, self, dim, index, src):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.scatter', 'scatter(g, self, dim, index, src)', {'sym_help': sym_help, 'torch': torch, 'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim, 'index': index, 'src': src}, 1)

@parse_args('v', 'i', 'none')
def cumsum(g, self, dim, dtype=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.cumsum', 'cumsum(g, self, dim, dtype=None)', {'torch': torch, 'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim, 'dtype': dtype}, 1)

def masked_select(g, self, mask):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.masked_select', 'masked_select(g, self, mask)', {'g': g, 'self': self, 'mask': mask}, 1)

def masked_scatter(g, self, mask, source):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.masked_scatter', 'masked_scatter(g, self, mask, source)', {'torch': torch, 'sym_help': sym_help, 'g': g, 'self': self, 'mask': mask, 'source': source}, 1)

def _len(g, self):
    return g.op('SequenceLength', self)

def __getitem_(g, self, i):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.__getitem_', '__getitem_(g, self, i)', {'torch': torch, 'g': g, 'self': self, 'i': i}, 1)

def append(g, self, tensor):
    return g.op('SequenceInsert', self, tensor)

def insert(g, self, pos, tensor):
    return g.op('SequenceInsert', self, tensor, pos)

def pop(g, tensor_list, dim):
    return g.op('SequenceErase', tensor_list, dim)

def cat(g, tensor_list, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.cat', 'cat(g, tensor_list, dim)', {'sym_help': sym_help, 'g': g, 'tensor_list': tensor_list, 'dim': dim}, 1)

def stack(g, tensor_list, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.stack', 'stack(g, tensor_list, dim)', {'sym_help': sym_help, 'g': g, 'tensor_list': tensor_list, 'dim': dim}, 1)

@parse_args('v', 'i', 'i', 'i')
def _unique2(g, self, sorted, return_inverse, return_counts):
    (u, indices, inverse_indices, counts) = g.op('Unique', self, sorted_i=sorted, outputs=4)
    return (u, inverse_indices, counts)

def _avg_pool(name, tuple_fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11._avg_pool', '_avg_pool(name, tuple_fn)', {'parse_args': parse_args, 'sym_help': sym_help, 'torch': torch, 'name': name, 'tuple_fn': tuple_fn}, 1)
avg_pool1d = _avg_pool('avg_pool1d', _single)
avg_pool2d = _avg_pool('avg_pool2d', _pair)
avg_pool3d = _avg_pool('avg_pool3d', _triple)

@parse_args('v', 'i', 'i', 'i', 'i')
def unique_dim(g, self, dim, sorted, return_inverse, return_counts):
    (u, indices, inverse_indices, counts) = g.op('Unique', self, axis_i=dim, sorted_i=sorted, outputs=4)
    return (u, inverse_indices, counts)

@parse_args('v', 'v', 'i', 'i', 'i', 'none')
def topk(g, self, k, dim, largest, sorted, out=None):
    return sym_help._topk_helper(g, self, k, dim, largest=largest, sorted=sorted, out=out)

@parse_args('v', 'i', 'i', 'none')
def sort(g, self, dim, decending, out=None):
    return sym_help._sort_helper(g, self, dim, decending=decending, out=out)

def round(g, self):
    return g.op('Round', self)

@parse_args('v', 'v', 'i')
def split_with_sizes(g, self, split_sizes, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.split_with_sizes', 'split_with_sizes(g, self, split_sizes, dim)', {'sym_help': sym_help, 'torch': torch, 'parse_args': parse_args, 'g': g, 'self': self, 'split_sizes': split_sizes, 'dim': dim}, 1)

def _prepare_onnx_paddings(g, dim, pad):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11._prepare_onnx_paddings', '_prepare_onnx_paddings(g, dim, pad)', {'torch': torch, 'sym_help': sym_help, 'g': g, 'dim': dim, 'pad': pad}, 1)

def constant_pad_nd(g, input, padding, value=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.constant_pad_nd', 'constant_pad_nd(g, input, padding, value=None)', {'sym_help': sym_help, '_prepare_onnx_paddings': _prepare_onnx_paddings, 'g': g, 'input': input, 'padding': padding, 'value': value}, 1)

def reflection_pad(g, input, padding):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.reflection_pad', 'reflection_pad(g, input, padding)', {'_prepare_onnx_paddings': _prepare_onnx_paddings, 'g': g, 'input': input, 'padding': padding}, 1)

def replication_pad(g, input, padding):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.replication_pad', 'replication_pad(g, input, padding)', {'_prepare_onnx_paddings': _prepare_onnx_paddings, 'g': g, 'input': input, 'padding': padding}, 1)
reflection_pad1d = reflection_pad
reflection_pad2d = reflection_pad
reflection_pad3d = reflection_pad
replication_pad1d = replication_pad
replication_pad2d = replication_pad
replication_pad3d = replication_pad

def det(g, self):
    return g.op('Det', self)

def logdet(g, input):
    from torch.onnx.symbolic_opset9 import log
    return log(g, det(g, input))

def arange(g, *args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.arange', 'arange(g, *args)', {'sym_help': sym_help, 'torch': torch, 'g': g, 'args': args}, 1)

@parse_args('v', 'i')
def _dim_arange(g, like, dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11._dim_arange', '_dim_arange(g, like, dim)', {'torch': torch, 'sym_help': sym_help, 'arange': arange, 'parse_args': parse_args, 'g': g, 'like': like, 'dim': dim}, 1)

def size(g, self, dim=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.size', 'size(g, self, dim=None)', {'sym_help': sym_help, 'g': g, 'self': self, 'dim': dim}, 1)

def squeeze(g, self, dim=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.squeeze', 'squeeze(g, self, dim=None)', {'sym_help': sym_help, 'g': g, 'self': self, 'dim': dim}, 1)

@parse_args('v', 'i')
def unsqueeze(g, self, dim):
    return g.op('Unsqueeze', self, axes_i=[dim])

def mm(g, self, other):
    return g.op('Gemm', self, other, beta_f=0.0, alpha_f=1.0)

def index_fill(g, self, dim, index, value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.index_fill', 'index_fill(g, self, dim, index, value)', {'sym_help': sym_help, 'torch': torch, 'expand': expand, 'scatter': scatter, 'g': g, 'self': self, 'dim': dim, 'index': index, 'value': value}, 1)

def index_copy(g, self, dim, index, source):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.index_copy', 'index_copy(g, self, dim, index, source)', {'sym_help': sym_help, 'torch': torch, 'scatter': scatter, 'g': g, 'self': self, 'dim': dim, 'index': index, 'source': source}, 1)

def __rshift_(g, self, other):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.__rshift_', '__rshift_(g, self, other)', {'sym_help': sym_help, 'torch': torch, 'g': g, 'self': self, 'other': other}, 1)

def __lshift_(g, self, other):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.__lshift_', '__lshift_(g, self, other)', {'sym_help': sym_help, 'torch': torch, 'g': g, 'self': self, 'other': other}, 1)

def _get_im2col_indices_along_dim(g, input_d, kernel_size_d, dilation_d, padding_d, stride_d):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11._get_im2col_indices_along_dim', '_get_im2col_indices_along_dim(g, input_d, kernel_size_d, dilation_d, padding_d, stride_d)', {'torch': torch, 'numpy': numpy, 'g': g, 'input_d': input_d, 'kernel_size_d': kernel_size_d, 'dilation_d': dilation_d, 'padding_d': padding_d, 'stride_d': stride_d}, 1)

def _get_im2col_padded_input(g, input, padding_h, padding_w):
    pad = g.op('Constant', value_t=torch.LongTensor([0, 0, padding_h, padding_w] * 2))
    return g.op('Pad', input, pad)

def _get_im2col_output_shape(g, input, kernel_h, kernel_w):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11._get_im2col_output_shape', '_get_im2col_output_shape(g, input, kernel_h, kernel_w)', {'size': size, 'torch': torch, 'g': g, 'input': input, 'kernel_h': kernel_h, 'kernel_w': kernel_w}, 1)

@parse_args('v', 'is', 'is', 'is', 'is')
def im2col(g, input, kernel_size, dilation, padding, stride):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.im2col', 'im2col(g, input, kernel_size, dilation, padding, stride)', {'size': size, 'torch': torch, '_get_im2col_indices_along_dim': _get_im2col_indices_along_dim, '_get_im2col_output_shape': _get_im2col_output_shape, '_get_im2col_padded_input': _get_im2col_padded_input, 'parse_args': parse_args, 'g': g, 'input': input, 'kernel_size': kernel_size, 'dilation': dilation, 'padding': padding, 'stride': stride}, 1)

@parse_args('v', 'i', 'i')
def flatten(g, input, start_dim, end_dim):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset11.flatten', 'flatten(g, input, start_dim, end_dim)', {'_unimplemented': _unimplemented, 'torch': torch, 'parse_args': parse_args, 'g': g, 'input': input, 'start_dim': start_dim, 'end_dim': end_dim}, 1)

