from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from .qconfig import QConfig
from torch.jit._recursive import wrap_cpp_module


class ConvPackedParams(torch.nn.Module):
    
    def __init__(self):
        super(ConvPackedParams, self).__init__()
        wq = torch._empty_affine_quantized([1, 1, 1, 1], scale=1.0, zero_point=0, dtype=torch.qint8)
        self.stride = [1, 1]
        self.padding = [0, 0]
        self.dilation = [1, 1]
        self.groups = 1
        self.set_weight_bias(wq, None)
    
    @torch.jit.export
    def set_conv_params(self, stride, padding, dilation, groups):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
    
    @torch.jit.export
    def set_weight_bias(self, weight, bias):
        self._packed_params = torch.ops.quantized.conv2d_prepack(weight, bias, self.stride, self.padding, self.dilation, self.groups)
    
    @torch.jit.export
    def _weight_bias(self):
        return torch.ops.quantized.conv2d_unpack(self._packed_params)
    
    def forward(self, x):
        return x
    
    @torch.jit.export
    def __getstate__(self):
        (qweight, bias) = self._weight_bias()
        return (qweight, bias, self.stride, self.padding, self.dilation, self.groups, self.training)
    
    @torch.jit.export
    def __setstate__(self, state):
        self.stride = state[2]
        self.padding = state[3]
        self.dilation = state[4]
        self.groups = state[5]
        self.set_weight_bias(state[0], state[1])
        self.training = state[6]

linear_packed_params = None
conv_packed_params = None
if 'fbgemm' in torch.backends.quantized.supported_engines:
    linear_packed_params = torch.jit.script(torch.nn.quantized.modules.linear.LinearPackedParams())._c
    conv_packed_params = torch.jit.script(ConvPackedParams())._c

def _check_is_script_module(model):
    if not isinstance(model, torch.jit.ScriptModule):
        raise ValueError('input must be a script module, got: ' + str(type(model)))

def prepare_script(model, qconfig_dict, inplace=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization._quantize_script.prepare_script', 'prepare_script(model, qconfig_dict, inplace=False)', {'_check_is_script_module': _check_is_script_module, 'wrap_cpp_module': wrap_cpp_module, 'torch': torch, 'model': model, 'qconfig_dict': qconfig_dict, 'inplace': inplace}, 1)

def convert_script(model, inplace=False, debug=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization._quantize_script.convert_script', 'convert_script(model, inplace=False, debug=False)', {'_check_is_script_module': _check_is_script_module, 'wrap_cpp_module': wrap_cpp_module, 'torch': torch, 'model': model, 'inplace': inplace, 'debug': debug}, 1)

def script_qconfig(qconfig):
    return QConfig(activation=torch.jit.script(qconfig.activation())._c, weight=torch.jit.script(qconfig.weight())._c)

def quantize_script(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization._quantize_script.quantize_script', 'quantize_script(model, qconfig_dict, run_fn, run_args, inplace=False, debug=False)', {'_check_is_script_module': _check_is_script_module, 'script_qconfig': script_qconfig, 'torch': torch, 'wrap_cpp_module': wrap_cpp_module, 'prepare_script': prepare_script, 'convert_script': convert_script, 'model': model, 'qconfig_dict': qconfig_dict, 'run_fn': run_fn, 'run_args': run_args, 'inplace': inplace, 'debug': debug}, 1)

