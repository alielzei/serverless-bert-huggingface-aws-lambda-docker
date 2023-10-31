import torch.jit
import inspect
import textwrap

def _hidden(name):
    return (name.startswith('_') and not name.startswith('__'))

def _emit_type(type):
    return str(type)

def _emit_arg(indent, i, arg):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.supported_ops._emit_arg', '_emit_arg(indent, i, arg)', {'_emit_type': _emit_type, 'indent': indent, 'i': i, 'arg': arg}, 1)

def _emit_args(indent, arguments):
    return ','.join((_emit_arg(indent, i, arg) for (i, arg) in enumerate(arguments)))

def _emit_ret(ret):
    return _emit_type(ret.type)

def _emit_rets(returns):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.supported_ops._emit_rets', '_emit_rets(returns)', {'_emit_ret': _emit_ret, 'returns': returns}, 1)

def _emit_schema(mod, name, schema, arg_start=0, padding=4):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.supported_ops._emit_schema', '_emit_schema(mod, name, schema, arg_start=0, padding=4)', {'_emit_args': _emit_args, '_emit_rets': _emit_rets, 'mod': mod, 'name': name, 'schema': schema, 'arg_start': arg_start, 'padding': padding}, 1)

def _get_tensor_ops():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.supported_ops._get_tensor_ops', '_get_tensor_ops()', {'torch': torch, '_hidden': _hidden, '_emit_schema': _emit_schema}, 1)

def _get_nn_functional_ops():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.supported_ops._get_nn_functional_ops', '_get_nn_functional_ops()', {'torch': torch, 'inspect': inspect, '_hidden': _hidden, '_emit_schema': _emit_schema}, 2)

def _get_builtins_helper():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.supported_ops._get_builtins_helper', '_get_builtins_helper()', {'torch': torch, 'inspect': inspect, '_hidden': _hidden}, 1)

def _is_math_fn(fn):
    mod = inspect.getmodule(fn)
    return mod.__name__ == 'math'

def _get_torchscript_builtins():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.supported_ops._get_torchscript_builtins', '_get_torchscript_builtins()', {'_is_math_fn': _is_math_fn, '_get_builtins_helper': _get_builtins_helper, 'inspect': inspect, 'torch': torch, '_emit_schema': _emit_schema}, 2)

def _get_math_builtins():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.supported_ops._get_math_builtins', '_get_math_builtins()', {'_is_math_fn': _is_math_fn, '_get_builtins_helper': _get_builtins_helper, 'inspect': inspect, 'torch': torch, '_emit_schema': _emit_schema}, 2)

def _get_global_builtins():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.supported_ops._get_global_builtins', '_get_global_builtins()', {'torch': torch, '_emit_schema': _emit_schema, 'textwrap': textwrap}, 2)

def _list_supported_ops():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.supported_ops._list_supported_ops', '_list_supported_ops()', {'_get_tensor_ops': _get_tensor_ops, '_get_nn_functional_ops': _get_nn_functional_ops, '_get_torchscript_builtins': _get_torchscript_builtins, '_get_global_builtins': _get_global_builtins, '_get_math_builtins': _get_math_builtins}, 1)
__doc__ = _list_supported_ops()

