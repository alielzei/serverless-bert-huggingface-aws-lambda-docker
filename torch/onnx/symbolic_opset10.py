from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch.nn.modules.utils import _single, _pair, _triple
import torch.onnx
import torch.onnx.utils
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args, _unimplemented
import torch.onnx.symbolic_opset9

@parse_args('v', 'i', 'i', 'none')
def sort(g, self, dim, decending, out=None):
    return sym_help._sort_helper(g, self, dim, decending=decending, out=out)

@parse_args('v', 'v', 'i', 'i', 'i', 'none')
def topk(g, self, k, dim, largest, sorted, out=None):
    return sym_help._topk_helper(g, self, k, dim, largest=largest, sorted=sorted, out=out)

def _max_pool(name, tuple_fn, ndims, return_indices):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset10._max_pool', '_max_pool(name, tuple_fn, ndims, return_indices)', {'parse_args': parse_args, 'sym_help': sym_help, 'name': name, 'tuple_fn': tuple_fn, 'ndims': ndims, 'return_indices': return_indices}, 2)
max_pool1d = _max_pool('max_pool1d', _single, 1, return_indices=False)
max_pool2d = _max_pool('max_pool2d', _pair, 2, return_indices=False)
max_pool3d = _max_pool('max_pool3d', _triple, 3, return_indices=False)
max_pool1d_with_indices = _max_pool('max_pool1d_with_indices', _single, 1, return_indices=True)
max_pool2d_with_indices = _max_pool('max_pool2d_with_indices', _pair, 2, return_indices=True)
max_pool3d_with_indices = _max_pool('max_pool3d_with_indices', _triple, 3, return_indices=True)

def _avg_pool(name, tuple_fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset10._avg_pool', '_avg_pool(name, tuple_fn)', {'parse_args': parse_args, 'sym_help': sym_help, 'name': name, 'tuple_fn': tuple_fn}, 1)
avg_pool1d = _avg_pool('avg_pool1d', _single)
avg_pool2d = _avg_pool('avg_pool2d', _pair)
avg_pool3d = _avg_pool('avg_pool3d', _triple)

def _interpolate(name, dim, interpolate_mode):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset10._interpolate', '_interpolate(name, dim, interpolate_mode)', {'sym_help': sym_help, '_unimplemented': _unimplemented, 'name': name, 'dim': dim, 'interpolate_mode': interpolate_mode}, 1)
upsample_nearest1d = _interpolate('upsample_nearest1d', 3, 'nearest')
upsample_nearest2d = _interpolate('upsample_nearest2d', 4, 'nearest')
upsample_nearest3d = _interpolate('upsample_nearest3d', 5, 'nearest')
upsample_linear1d = _interpolate('upsample_linear1d', 3, 'linear')
upsample_bilinear2d = _interpolate('upsample_bilinear2d', 4, 'linear')
upsample_trilinear3d = _interpolate('upsample_trilinear3d', 5, 'linear')

def __interpolate(g, input, size, scale_factor, mode, align_corners, recompute_scale_factor):
    (scales, mode) = sym_help._interpolate_get_scales_and_mode(g, input, size, scale_factor, mode, align_corners)
    return g.op('Resize', input, scales, mode_s=mode)

def _slice(g, input, axes, starts, ends, steps=None, dynamic_slice=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset10._slice', '_slice(g, input, axes, starts, ends, steps=None, dynamic_slice=False)', {'torch': torch, 'g': g, 'input': input, 'axes': axes, 'starts': starts, 'ends': ends, 'steps': steps, 'dynamic_slice': dynamic_slice}, 1)

@parse_args('v', 'v', 'v', 'v', 'i')
def slice(g, self, dim, start, end, step):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset10.slice', 'slice(g, self, dim, start, end, step)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'self': self, 'dim': dim, 'start': start, 'end': end, 'step': step}, 1)

@parse_args('v', 'is')
def flip(g, input, dims):
    return sym_help._slice_helper(g, input, axes=dims, starts=[-1] * len(dims), ends=[-9223372036854775807] * len(dims), steps=[-1] * len(dims))

def fmod(g, input, other):
    return g.op('Mod', input, other, fmod_i=1)

