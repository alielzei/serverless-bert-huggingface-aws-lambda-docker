from torch.onnx.symbolic_helper import _black_list_in_opset
import torch.onnx.symbolic_opset9 as sym_opset9
import warnings
black_listed_operators = ['scan', 'expand', 'expand_as', 'meshgrid', 'adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d', 'max_pool1d_with_indices', 'max_pool2d_with_indices', 'max_pool3d_with_indices']

def max(g, self, dim_or_y=None, keepdim=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset7.max', 'max(g, self, dim_or_y=None, keepdim=None)', {'warnings': warnings, 'sym_opset9': sym_opset9, 'g': g, 'self': self, 'dim_or_y': dim_or_y, 'keepdim': keepdim}, 1)

def min(g, self, dim_or_y=None, keepdim=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_opset7.min', 'min(g, self, dim_or_y=None, keepdim=None)', {'warnings': warnings, 'sym_opset9': sym_opset9, 'g': g, 'self': self, 'dim_or_y': dim_or_y, 'keepdim': keepdim}, 1)
for black_listed_op in black_listed_operators:
    vars()[black_listed_op] = _black_list_in_opset(black_listed_op)

