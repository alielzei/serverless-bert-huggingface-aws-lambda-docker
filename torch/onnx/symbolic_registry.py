import warnings
import importlib
from inspect import getmembers, isfunction
_registry = {}
_symbolic_versions = {}
from torch.onnx.symbolic_helper import _onnx_stable_opsets
for opset_version in _onnx_stable_opsets:
    module = importlib.import_module('torch.onnx.symbolic_opset{}'.format(opset_version))
    _symbolic_versions[opset_version] = module

def register_version(domain, version):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.onnx.symbolic_registry.register_version', 'register_version(domain, version)', {'is_registered_version': is_registered_version, '_registry': _registry, 'register_ops_in_version': register_ops_in_version, 'domain': domain, 'version': version}, 0)

def register_ops_helper(domain, version, iter_version):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.onnx.symbolic_registry.register_ops_helper', 'register_ops_helper(domain, version, iter_version)', {'get_ops_in_version': get_ops_in_version, 'isfunction': isfunction, 'is_registered_op': is_registered_op, 'register_op': register_op, 'domain': domain, 'version': version, 'iter_version': iter_version}, 0)

def register_ops_in_version(domain, version):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.onnx.symbolic_registry.register_ops_in_version', 'register_ops_in_version(domain, version)', {'register_ops_helper': register_ops_helper, 'domain': domain, 'version': version}, 0)

def get_ops_in_version(version):
    return getmembers(_symbolic_versions[version])

def is_registered_version(domain, version):
    global _registry
    return (domain, version) in _registry

def register_op(opname, op, domain, version):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.onnx.symbolic_registry.register_op', 'register_op(opname, op, domain, version)', {'warnings': warnings, 'is_registered_version': is_registered_version, '_registry': _registry, 'opname': opname, 'op': op, 'domain': domain, 'version': version}, 0)

def is_registered_op(opname, domain, version):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_registry.is_registered_op', 'is_registered_op(opname, domain, version)', {'warnings': warnings, '_registry': _registry, 'opname': opname, 'domain': domain, 'version': version}, 1)

def get_op_supported_version(opname, domain, version):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_registry.get_op_supported_version', 'get_op_supported_version(opname, domain, version)', {'_onnx_stable_opsets': _onnx_stable_opsets, 'get_ops_in_version': get_ops_in_version, 'opname': opname, 'domain': domain, 'version': version}, 1)

def get_registered_op(opname, domain, version):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.symbolic_registry.get_registered_op', 'get_registered_op(opname, domain, version)', {'warnings': warnings, 'is_registered_op': is_registered_op, 'get_op_supported_version': get_op_supported_version, '_registry': _registry, 'opname': opname, 'domain': domain, 'version': version}, 1)

