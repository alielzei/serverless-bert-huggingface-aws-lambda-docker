from __future__ import absolute_import, division, print_function, unicode_literals
import torch.onnx.symbolic_helper as sym_help
from torch.onnx.symbolic_helper import parse_args

@parse_args('s', 'v')
def einsum(g, equation, tensor_list):
    tensors = sym_help._unpack_list(tensor_list)
    return g.op('Einsum', *tensors, equation_s=equation)

def nll_loss(g, self, target, weight, reduction, ignore_index):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset12.nll_loss', 'nll_loss(g, self, target, weight, reduction, ignore_index)', {'sym_help': sym_help, 'g': g, 'self': self, 'target': target, 'weight': weight, 'reduction': reduction, 'ignore_index': ignore_index}, 1)

def nll_loss2d(g, self, target, weight, reduction, ignore_index):
    return nll_loss(g, self, target, weight, reduction, ignore_index)

