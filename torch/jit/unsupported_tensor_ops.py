import torch.jit
from textwrap import dedent
from torch._six import PY2

def execWrapper(code, glob, loc):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit.unsupported_tensor_ops.execWrapper', 'execWrapper(code, glob, loc)', {'PY2': PY2, 'code': code, 'glob': glob, 'loc': loc}, 0)

def _gen_unsupported_methods_properties():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.unsupported_tensor_ops._gen_unsupported_methods_properties', '_gen_unsupported_methods_properties()', {'torch': torch, 'dedent': dedent, 'execWrapper': execWrapper}, 2)

def _list_unsupported_tensor_ops():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.unsupported_tensor_ops._list_unsupported_tensor_ops', '_list_unsupported_tensor_ops()', {'_gen_unsupported_methods_properties': _gen_unsupported_methods_properties}, 1)
__doc__ = _list_unsupported_tensor_ops()

