from torch.onnx.symbolic_helper import parse_args
import torch.onnx.symbolic_helper as sym_help
import torch.onnx.symbolic_registry as sym_registry
import importlib
from inspect import getmembers, isfunction

def register_quantized_ops(domain, version):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.register_quantized_ops', 'register_quantized_ops(domain, version)', {'sym_registry': sym_registry, 'importlib': importlib, 'getmembers': getmembers, 'isfunction': isfunction, 'domain': domain, 'version': version}, 0)

def _permute_helper(g, input, axes):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2._permute_helper', '_permute_helper(g, input, axes)', {'sym_help': sym_help, 'g': g, 'input': input, 'axes': axes}, 1)

def nchw2nhwc(g, input):
    axes = [0, 2, 3, 1]
    return _permute_helper(g, input, axes)

def nhwc2nchw(g, input):
    axes = [0, 3, 1, 2]
    return _permute_helper(g, input, axes)

def linear_prepack(g, weight, bias):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.linear_prepack', 'linear_prepack(g, weight, bias)', {'sym_help': sym_help, 'g': g, 'weight': weight, 'bias': bias}, 1)

@parse_args('v', 'v', 'v', 'f', 'i')
def linear(g, input, weight, bias, scale, zero_point):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.linear', 'linear(g, input, weight, bias, scale, zero_point)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'weight': weight, 'bias': bias, 'scale': scale, 'zero_point': zero_point}, 1)

def conv_prepack(g, input, weight, bias, stride, padding, dilation, groups):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.conv_prepack', 'conv_prepack(g, input, weight, bias, stride, padding, dilation, groups)', {'sym_help': sym_help, 'g': g, 'input': input, 'weight': weight, 'bias': bias, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups}, 1)

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'f', 'i')
def conv2d(g, input, weight, bias, stride, padding, dilation, groups, scale, zero_point):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.conv2d', 'conv2d(g, input, weight, bias, stride, padding, dilation, groups, scale, zero_point)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'weight': weight, 'bias': bias, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups, 'scale': scale, 'zero_point': zero_point}, 1)

@parse_args('v', 'v', 'v', 'is', 'is', 'is', 'i', 'f', 'i')
def conv2d_relu(g, input, weight, bias, stride, padding, dilation, groups, scale, zero_point):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.conv2d_relu', 'conv2d_relu(g, input, weight, bias, stride, padding, dilation, groups, scale, zero_point)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'weight': weight, 'bias': bias, 'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups, 'scale': scale, 'zero_point': zero_point}, 1)

@parse_args('v', 'v', 'f', 'i')
def add(g, input_a, input_b, scale, zero_point):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.add', 'add(g, input_a, input_b, scale, zero_point)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input_a': input_a, 'input_b': input_b, 'scale': scale, 'zero_point': zero_point}, 1)

@parse_args('v')
def relu(g, input):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.relu', 'relu(g, input)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input}, 1)

@parse_args('v', 'f', 'i', 't')
def quantize_per_tensor(g, input, scale, zero_point, dtype):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.quantize_per_tensor', 'quantize_per_tensor(g, input, scale, zero_point, dtype)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'scale': scale, 'zero_point': zero_point, 'dtype': dtype}, 1)

@parse_args('v')
def dequantize(g, input):
    return g.op('_caffe2::Int8Dequantize', input)

@parse_args('v', 't', 't', 't', 't', 't', 't', 't')
def _empty_affine_quantized(g, input, shape, scale, zero_point, dtype, pin_memory, memory_format, layout):
    return input

def upsample_nearest2d(g, input, output_size, align_corners=None, scales_h=None, scales_w=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.upsample_nearest2d', 'upsample_nearest2d(g, input, output_size, align_corners=None, scales_h=None, scales_w=None)', {'sym_help': sym_help, 'nchw2nhwc': nchw2nhwc, 'nhwc2nchw': nhwc2nchw, 'g': g, 'input': input, 'output_size': output_size, 'align_corners': align_corners, 'scales_h': scales_h, 'scales_w': scales_w}, 1)

@parse_args('v', 'is', 'is', 'is', 'is', 'i')
def max_pool2d(g, input, kernel_size, stride, padding, dilation, ceil_mode):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.max_pool2d', 'max_pool2d(g, input, kernel_size, stride, padding, dilation, ceil_mode)', {'sym_help': sym_help, 'nchw2nhwc': nchw2nhwc, 'nhwc2nchw': nhwc2nchw, 'parse_args': parse_args, 'g': g, 'input': input, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'dilation': dilation, 'ceil_mode': ceil_mode}, 1)

@parse_args('v', 'is', 'is', 'is', 'i', 'i', 'none')
def avg_pool2d(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.avg_pool2d', 'avg_pool2d(g, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override=None)', {'sym_help': sym_help, 'nchw2nhwc': nchw2nhwc, 'nhwc2nchw': nhwc2nchw, 'parse_args': parse_args, 'g': g, 'input': input, 'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'ceil_mode': ceil_mode, 'count_include_pad': count_include_pad, 'divisor_override': divisor_override}, 1)

def reshape(g, input, shape):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.reshape', 'reshape(g, input, shape)', {'sym_help': sym_help, 'g': g, 'input': input, 'shape': shape}, 1)

@parse_args('v', 'v', 'v', 'v', 'i')
def slice(g, input, dim, start, end, step):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.slice', 'slice(g, input, dim, start, end, step)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input, 'dim': dim, 'start': start, 'end': end, 'step': step}, 1)

def cat(g, tensor_list, dim, scale=None, zero_point=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.cat', 'cat(g, tensor_list, dim, scale=None, zero_point=None)', {'sym_help': sym_help, 'g': g, 'tensor_list': tensor_list, 'dim': dim, 'scale': scale, 'zero_point': zero_point}, 1)

@parse_args('v')
def sigmoid(g, input):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_caffe2.sigmoid', 'sigmoid(g, input)', {'sym_help': sym_help, 'parse_args': parse_args, 'g': g, 'input': input}, 1)

