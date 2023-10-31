from __future__ import absolute_import, division, print_function, unicode_literals
'\nThe torch.onnx module contains functions to export models into the ONNX\nIR format.  These models can be loaded with the ONNX library and then\nconverted to models which run on other deep learning frameworks.\n'
import torch
import torch.jit
import torch.autograd
import torch.serialization
import re
from torch._six import container_abcs
import contextlib
import numbers
import warnings
from torch._six import string_classes
from torch.jit import _unique_state_dict
from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes
from torch._C import ListType, _propagate_and_assign_input_shapes, _assign_output_shapes, _check_onnx_proto
__IN_ONNX_EXPORT = False

def is_in_onnx_export():
    global __IN_ONNX_EXPORT
    return __IN_ONNX_EXPORT

@contextlib.contextmanager
def set_training(model, mode):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils.set_training', 'set_training(model, mode)', {'contextlib': contextlib, 'model': model, 'mode': mode}, 1)

def export(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, aten=False, export_raw_ir=False, operator_export_type=None, opset_version=None, _retain_param_name=True, do_constant_folding=True, example_outputs=None, strip_doc_string=True, dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None, enable_onnx_checker=True, use_external_data_format=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.onnx.utils.export', 'export(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, aten=False, export_raw_ir=False, operator_export_type=None, opset_version=None, _retain_param_name=True, do_constant_folding=True, example_outputs=None, strip_doc_string=True, dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None, enable_onnx_checker=True, use_external_data_format=False)', {'OperatorExportTypes': OperatorExportTypes, 'torch': torch, '_export': _export, 'model': model, 'args': args, 'f': f, 'export_params': export_params, 'verbose': verbose, 'training': training, 'input_names': input_names, 'output_names': output_names, 'aten': aten, 'export_raw_ir': export_raw_ir, 'operator_export_type': operator_export_type, 'opset_version': opset_version, '_retain_param_name': _retain_param_name, 'do_constant_folding': do_constant_folding, 'example_outputs': example_outputs, 'strip_doc_string': strip_doc_string, 'dynamic_axes': dynamic_axes, 'keep_initializers_as_inputs': keep_initializers_as_inputs, 'custom_opsets': custom_opsets, 'enable_onnx_checker': enable_onnx_checker, 'use_external_data_format': use_external_data_format}, 0)

def _split_tensor_list_constants(g, block):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.onnx.utils._split_tensor_list_constants', '_split_tensor_list_constants(g, block)', {'_split_tensor_list_constants': _split_tensor_list_constants, 'ListType': ListType, 'g': g, 'block': block}, 0)

def _optimize_graph(graph, operator_export_type, _disable_torch_constant_prop=False, fixed_batch_size=False, params_dict=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._optimize_graph', '_optimize_graph(graph, operator_export_type, _disable_torch_constant_prop=False, fixed_batch_size=False, params_dict=None)', {'torch': torch, '_split_tensor_list_constants': _split_tensor_list_constants, 'OperatorExportTypes': OperatorExportTypes, 'graph': graph, 'operator_export_type': operator_export_type, '_disable_torch_constant_prop': _disable_torch_constant_prop, 'fixed_batch_size': fixed_batch_size, 'params_dict': params_dict}, 1)

def warn_on_static_input_change(input_states):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.onnx.utils.warn_on_static_input_change', 'warn_on_static_input_change(input_states)', {'warnings': warnings, 'input_states': input_states}, 0)

def _resolve_args_by_export_type(arg_name, arg_value, operator_export_type):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._resolve_args_by_export_type', '_resolve_args_by_export_type(arg_name, arg_value, operator_export_type)', {'warnings': warnings, 'arg_name': arg_name, 'arg_value': arg_value, 'operator_export_type': operator_export_type}, 1)

def _decide_keep_init_as_input(keep_initializers_as_inputs, operator_export_type, opset_version):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._decide_keep_init_as_input', '_decide_keep_init_as_input(keep_initializers_as_inputs, operator_export_type, opset_version)', {'warnings': warnings, 'OperatorExportTypes': OperatorExportTypes, 'keep_initializers_as_inputs': keep_initializers_as_inputs, 'operator_export_type': operator_export_type, 'opset_version': opset_version}, 1)

def _decide_add_node_names(add_node_names, operator_export_type):
    return _resolve_args_by_export_type('add_node_names', add_node_names, operator_export_type)

def _decide_constant_folding(do_constant_folding, operator_export_type):
    return _resolve_args_by_export_type('do_constant_folding', do_constant_folding, operator_export_type)

def _decide_external_data_format(use_external_data_format, operator_export_type, f):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._decide_external_data_format', '_decide_external_data_format(use_external_data_format, operator_export_type, f)', {'_resolve_args_by_export_type': _resolve_args_by_export_type, 'use_external_data_format': use_external_data_format, 'operator_export_type': operator_export_type, 'f': f}, 2)

def _trace(func, args, operator_export_type, return_outs=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._trace', '_trace(func, args, operator_export_type, return_outs=False)', {'torch': torch, 'warn_on_static_input_change': warn_on_static_input_change, '_optimize_graph': _optimize_graph, 'func': func, 'args': args, 'operator_export_type': operator_export_type, 'return_outs': return_outs}, 2)

def _trace_and_get_graph_from_model(model, args, training):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._trace_and_get_graph_from_model', '_trace_and_get_graph_from_model(model, args, training)', {'_unique_state_dict': _unique_state_dict, 'set_training': set_training, 'torch': torch, 'warn_on_static_input_change': warn_on_static_input_change, 'model': model, 'args': args, 'training': training}, 2)

def _model_to_graph(model, args, verbose=False, training=False, input_names=None, output_names=None, operator_export_type=OperatorExportTypes.ONNX, example_outputs=None, propagate=False, _retain_param_name=False, do_constant_folding=True, _disable_torch_constant_prop=False, fixed_batch_size=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._model_to_graph', '_model_to_graph(model, args, verbose=False, training=False, input_names=None, output_names=None, operator_export_type=OperatorExportTypes.ONNX, example_outputs=None, propagate=False, _retain_param_name=False, do_constant_folding=True, _disable_torch_constant_prop=False, fixed_batch_size=False)', {'torch': torch, '_propagate_and_assign_input_shapes': _propagate_and_assign_input_shapes, '_trace_and_get_graph_from_model': _trace_and_get_graph_from_model, '_unique_state_dict': _unique_state_dict, '_optimize_graph': _optimize_graph, '_assign_output_shapes': _assign_output_shapes, '_set_input_and_output_names': _set_input_and_output_names, 'model': model, 'args': args, 'verbose': verbose, 'training': training, 'input_names': input_names, 'output_names': output_names, 'operator_export_type': operator_export_type, 'example_outputs': example_outputs, 'propagate': propagate, '_retain_param_name': _retain_param_name, 'do_constant_folding': do_constant_folding, '_disable_torch_constant_prop': _disable_torch_constant_prop, 'fixed_batch_size': fixed_batch_size}, 3)

def export_to_pretty_string(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, aten=False, export_raw_ir=False, operator_export_type=None, export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False, google_printer=False, opset_version=None, _retain_param_name=True, keep_initializers_as_inputs=None, custom_opsets=None, add_node_names=True, do_constant_folding=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils.export_to_pretty_string', 'export_to_pretty_string(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, aten=False, export_raw_ir=False, operator_export_type=None, export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False, google_printer=False, opset_version=None, _retain_param_name=True, keep_initializers_as_inputs=None, custom_opsets=None, add_node_names=True, do_constant_folding=True)', {'OperatorExportTypes': OperatorExportTypes, '_export_to_pretty_string': _export_to_pretty_string, 'model': model, 'args': args, 'f': f, 'export_params': export_params, 'verbose': verbose, 'training': training, 'input_names': input_names, 'output_names': output_names, 'aten': aten, 'export_raw_ir': export_raw_ir, 'operator_export_type': operator_export_type, 'export_type': export_type, 'example_outputs': example_outputs, 'propagate': propagate, 'google_printer': google_printer, 'opset_version': opset_version, '_retain_param_name': _retain_param_name, 'keep_initializers_as_inputs': keep_initializers_as_inputs, 'custom_opsets': custom_opsets, 'add_node_names': add_node_names, 'do_constant_folding': do_constant_folding}, 1)

def _export_to_pretty_string(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, operator_export_type=OperatorExportTypes.ONNX, export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False, google_printer=False, opset_version=None, _retain_param_name=False, do_constant_folding=True, keep_initializers_as_inputs=None, fixed_batch_size=False, custom_opsets=None, add_node_names=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._export_to_pretty_string', '_export_to_pretty_string(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, operator_export_type=OperatorExportTypes.ONNX, export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False, google_printer=False, opset_version=None, _retain_param_name=False, do_constant_folding=True, keep_initializers_as_inputs=None, fixed_batch_size=False, custom_opsets=None, add_node_names=True)', {'_decide_keep_init_as_input': _decide_keep_init_as_input, '_decide_add_node_names': _decide_add_node_names, '_decide_constant_folding': _decide_constant_folding, '_model_to_graph': _model_to_graph, 'model': model, 'args': args, 'f': f, 'export_params': export_params, 'verbose': verbose, 'training': training, 'input_names': input_names, 'output_names': output_names, 'operator_export_type': operator_export_type, 'export_type': export_type, 'example_outputs': example_outputs, 'propagate': propagate, 'google_printer': google_printer, 'opset_version': opset_version, '_retain_param_name': _retain_param_name, 'do_constant_folding': do_constant_folding, 'keep_initializers_as_inputs': keep_initializers_as_inputs, 'fixed_batch_size': fixed_batch_size, 'custom_opsets': custom_opsets, 'add_node_names': add_node_names}, 1)

def _export(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, operator_export_type=None, export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False, opset_version=None, _retain_param_name=False, do_constant_folding=True, strip_doc_string=True, dynamic_axes=None, keep_initializers_as_inputs=None, fixed_batch_size=False, custom_opsets=None, add_node_names=True, enable_onnx_checker=True, use_external_data_format=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._export', '_export(model, args, f, export_params=True, verbose=False, training=False, input_names=None, output_names=None, operator_export_type=None, export_type=ExportTypes.PROTOBUF_FILE, example_outputs=None, propagate=False, opset_version=None, _retain_param_name=False, do_constant_folding=True, strip_doc_string=True, dynamic_axes=None, keep_initializers_as_inputs=None, fixed_batch_size=False, custom_opsets=None, add_node_names=True, enable_onnx_checker=True, use_external_data_format=False)', {'torch': torch, 'OperatorExportTypes': OperatorExportTypes, '_decide_keep_init_as_input': _decide_keep_init_as_input, '_decide_add_node_names': _decide_add_node_names, '_decide_constant_folding': _decide_constant_folding, '_decide_external_data_format': _decide_external_data_format, '_model_to_graph': _model_to_graph, 'ExportTypes': ExportTypes, '_validate_dynamic_axes': _validate_dynamic_axes, '_check_onnx_proto': _check_onnx_proto, 'ONNX_ARCHIVE_MODEL_PROTO_NAME': ONNX_ARCHIVE_MODEL_PROTO_NAME, 'model': model, 'args': args, 'f': f, 'export_params': export_params, 'verbose': verbose, 'training': training, 'input_names': input_names, 'output_names': output_names, 'operator_export_type': operator_export_type, 'export_type': export_type, 'example_outputs': example_outputs, 'propagate': propagate, 'opset_version': opset_version, '_retain_param_name': _retain_param_name, 'do_constant_folding': do_constant_folding, 'strip_doc_string': strip_doc_string, 'dynamic_axes': dynamic_axes, 'keep_initializers_as_inputs': keep_initializers_as_inputs, 'fixed_batch_size': fixed_batch_size, 'custom_opsets': custom_opsets, 'add_node_names': add_node_names, 'enable_onnx_checker': enable_onnx_checker, 'use_external_data_format': use_external_data_format}, 1)

def _set_input_and_output_names(graph, input_names, output_names):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._set_input_and_output_names', '_set_input_and_output_names(graph, input_names, output_names)', {'graph': graph, 'input_names': input_names, 'output_names': output_names}, 1)
attr_pattern = re.compile('^(.+)_([ifstgz])$')

def _run_symbolic_method(op_name, symbolic_fn, args):
    """
    This trampoline function gets invoked for every symbolic method
    call from C++.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._run_symbolic_method', '_run_symbolic_method(op_name, symbolic_fn, args)', {'op_name': op_name, 'symbolic_fn': symbolic_fn, 'args': args}, 1)

def _is_onnx_list(value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._is_onnx_list', '_is_onnx_list(value)', {'string_classes': string_classes, 'torch': torch, 'container_abcs': container_abcs, 'value': value}, 1)

def _add_attribute(node, key, value, aten):
    """ initializes the right attribute based on type of value """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._add_attribute', '_add_attribute(node, key, value, aten)', {'attr_pattern': attr_pattern, '_is_onnx_list': _is_onnx_list, 'torch': torch, '_scalar': _scalar, 'node': node, 'key': key, 'value': value, 'aten': aten}, 1)

def _scalar(x):
    """Convert a scalar tensor into a Python value."""
    assert x.numel() == 1
    return x[0]

def _newNode(g, opname, outputs, *args, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._newNode', '_newNode(g, opname, outputs, *args, **kwargs)', {'_add_attribute': _add_attribute, 'g': g, 'opname': opname, 'outputs': outputs, 'args': args, 'kwargs': kwargs}, 1)

def _graph_op(g, opname, *raw_args, **kwargs):
    """
    Create an ONNX operator 'opname', taking 'args' as inputs and attributes
    'kwargs'; returning the node representing the single output of this operator
    (see the `outputs` keyword argument for multi-return nodes).

    The set of operators and the inputs/attributes they take
    is documented at https://github.com/onnx/onnx/blob/master/docs/Operators.md

    This function is monkey-patched onto Graph.

    Arguments:
        opname (string): The ONNX operator name, e.g., `Abs` or `Add`.
        args (Node...): The inputs to the operator; usually provided
            as arguments to the `symbolic` definition.
        kwargs: The attributes of the ONNX operator, with keys named
            according to the following convention: `alpha_f` indicates
            the `alpha` attribute with type `f`.  The valid type specifiers are
            `f` (float), `i` (int), `s` (string) or `t` (Tensor).  An attribute
            specified with type float accepts either a single float, or a
            list of floats (e.g., you would say `dims_i` for a `dims` attribute
            that takes a list of integers).
        outputs (int, optional):  The number of outputs this operator returns;
            by default an operator is assumed to return a single output.
            If `outputs` is greater than one, this functions returns a tuple
            of output `Node`, representing each output of the ONNX operator
            in positional.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._graph_op', '_graph_op(g, opname, *raw_args, **kwargs)', {'torch': torch, '_newNode': _newNode, 'g': g, 'opname': opname, 'raw_args': raw_args, 'kwargs': kwargs}, 1)

def _run_symbolic_function(g, n, inputs, env, operator_export_type=OperatorExportTypes.ONNX):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._run_symbolic_function', '_run_symbolic_function(g, n, inputs, env, operator_export_type=OperatorExportTypes.ONNX)', {'OperatorExportTypes': OperatorExportTypes, '_graph_at': _graph_at, 'warnings': warnings, 'g': g, 'n': n, 'inputs': inputs, 'env': env, 'operator_export_type': operator_export_type}, 1)

def _graph_at(g, opname, *args, **kwargs):
    return g.op('ATen', *args, operator_s=opname, **kwargs)

def _graph_constant(g, value, dims, type, *args, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._graph_constant', '_graph_constant(g, value, dims, type, *args, **kwargs)', {'numbers': numbers, 'torch': torch, 'g': g, 'value': value, 'dims': dims, 'type': type, 'args': args, 'kwargs': kwargs}, 1)

def _node_getitem(self, k):
    """
    Accessor for attributes of a node which is polymorphic over
    return type.

    NB: This is monkey-patched onto Node.
    """
    sel = self.kindOf(k)
    return getattr(self, sel)(k)

def register_custom_op_symbolic(symbolic_name, symbolic_fn, opset_version):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.onnx.utils.register_custom_op_symbolic', 'register_custom_op_symbolic(symbolic_name, symbolic_fn, opset_version)', {'re': re, 'symbolic_name': symbolic_name, 'symbolic_fn': symbolic_fn, 'opset_version': opset_version}, 0)

def _validate_dynamic_axes(dynamic_axes, model, input_names, output_names):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.onnx.utils._validate_dynamic_axes', '_validate_dynamic_axes(dynamic_axes, model, input_names, output_names)', {'warnings': warnings, 'dynamic_axes': dynamic_axes, 'model': model, 'input_names': input_names, 'output_names': output_names}, 1)
torch._C.Graph.op = _graph_op
torch._C.Graph.at = _graph_at
torch._C.Graph.constant = _graph_constant
torch._C.Node.__getitem__ = _node_getitem

