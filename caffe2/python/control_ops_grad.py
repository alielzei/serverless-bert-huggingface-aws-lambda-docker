from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2

def gen_do_gradient(op, g_output):
    """
    Generates gradient Do operator, given forward Do op and a list
    of gradient blobs corresponding to forward op's outputs
    Returns a gradient op and a list of blobs corresponding to input gradients
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad.gen_do_gradient', 'gen_do_gradient(op, g_output)', {'_do_op_sanity_check_and_process': _do_op_sanity_check_and_process, 'dedupe_g_output': dedupe_g_output, '_gen_subgradient_pass': _gen_subgradient_pass, '_prepare_blob_copy_op': _prepare_blob_copy_op, '_prepare_gradient_do_op': _prepare_gradient_do_op, 'op': op, 'g_output': g_output}, 2)

def dedupe_g_output(op, g_output):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad.dedupe_g_output', 'dedupe_g_output(op, g_output)', {'caffe2_pb2': caffe2_pb2, 'op': op, 'g_output': g_output}, 2)

def gen_while_gradient(op, g_output):
    """
    Generates gradient While operator
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad.gen_while_gradient', 'gen_while_gradient(op, g_output)', {'dedupe_g_output': dedupe_g_output, '_get_net_argument': _get_net_argument, '_get_do_arguments': _get_do_arguments, '_gen_subnet_gradient': _gen_subnet_gradient, '_prepare_gradient_while_ops': _prepare_gradient_while_ops, 'op': op, 'g_output': g_output}, 2)

def _prepare_gradient_while_ops(fwd_op, input_names, output_names, loop_grad_net, workspace_blob, init_grad_map, loop_grad_map):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad._prepare_gradient_while_ops', '_prepare_gradient_while_ops(fwd_op, input_names, output_names, loop_grad_net, workspace_blob, init_grad_map, loop_grad_map)', {'caffe2_pb2': caffe2_pb2, 'fwd_op': fwd_op, 'input_names': input_names, 'output_names': output_names, 'loop_grad_net': loop_grad_net, 'workspace_blob': workspace_blob, 'init_grad_map': init_grad_map, 'loop_grad_map': loop_grad_map}, 1)

def _get_do_arguments(do_op):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad._get_do_arguments', '_get_do_arguments(do_op)', {'do_op': do_op}, 1)

def gen_if_gradient(op, g_output):
    """
    Generates gradient If operator, given forward If op and a list
    of gradient blobs corresponding to forward op's outputs
    Returns a gradient op and a list of blobs corresponding to input gradients
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad.gen_if_gradient', 'gen_if_gradient(op, g_output)', {'dedupe_g_output': dedupe_g_output, '_get_net_argument': _get_net_argument, '_gen_subnet_gradient': _gen_subnet_gradient, '_gen_grad_zero_init_ops': _gen_grad_zero_init_ops, 'caffe2_pb2': caffe2_pb2, '_prepare_gradient_if_op': _prepare_gradient_if_op, 'op': op, 'g_output': g_output}, 2)

def _gen_subnet_gradient(subnet, init_grad):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad._gen_subnet_gradient', '_gen_subnet_gradient(subnet, init_grad)', {'_gen_subgradient_pass': _gen_subgradient_pass, 'caffe2_pb2': caffe2_pb2, 'subnet': subnet, 'init_grad': init_grad}, 4)

def _get_net_argument(op, net_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad._get_net_argument', '_get_net_argument(op, net_name)', {'op': op, 'net_name': net_name}, 1)

def getNetArgument(op, net_name):
    """A wrapper for external call"""
    return _get_net_argument(op, net_name)

def _gen_subgradient_pass(subnet, init_grad):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad._gen_subgradient_pass', '_gen_subgradient_pass(subnet, init_grad)', {'subnet': subnet, 'init_grad': init_grad}, 2)

def _do_op_sanity_check_and_process(op):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad._do_op_sanity_check_and_process', '_do_op_sanity_check_and_process(op)', {'_get_net_argument': _get_net_argument, 'op': op}, 4)

def _prepare_blob_copy_op(from_name, to_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad._prepare_blob_copy_op', '_prepare_blob_copy_op(from_name, to_name)', {'caffe2_pb2': caffe2_pb2, 'from_name': from_name, 'to_name': to_name}, 1)

def _prepare_gradient_do_op(fwd_op, fwd_net, grad_ops, inputs, outputs, blob_bindings, saved_fwd_blobs, workspace_blob_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad._prepare_gradient_do_op', '_prepare_gradient_do_op(fwd_op, fwd_net, grad_ops, inputs, outputs, blob_bindings, saved_fwd_blobs, workspace_blob_name)', {'caffe2_pb2': caffe2_pb2, 'fwd_op': fwd_op, 'fwd_net': fwd_net, 'grad_ops': grad_ops, 'inputs': inputs, 'outputs': outputs, 'blob_bindings': blob_bindings, 'saved_fwd_blobs': saved_fwd_blobs, 'workspace_blob_name': workspace_blob_name}, 1)

def _gen_grad_zero_init_ops(init_grad_map, grad_map, grad_output_names):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad._gen_grad_zero_init_ops', '_gen_grad_zero_init_ops(init_grad_map, grad_map, grad_output_names)', {'caffe2_pb2': caffe2_pb2, 'init_grad_map': init_grad_map, 'grad_map': grad_map, 'grad_output_names': grad_output_names}, 1)

def _prepare_gradient_if_op(fwd_op, input_names, output_names, then_grad_net, else_grad_net):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad._prepare_gradient_if_op', '_prepare_gradient_if_op(fwd_op, input_names, output_names, then_grad_net, else_grad_net)', {'caffe2_pb2': caffe2_pb2, 'fwd_op': fwd_op, 'input_names': input_names, 'output_names': output_names, 'then_grad_net': then_grad_net, 'else_grad_net': else_grad_net}, 1)

def disambiguate_grad_if_op_output(grad_op, idx, new_grad_output):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.control_ops_grad.disambiguate_grad_if_op_output', 'disambiguate_grad_if_op_output(grad_op, idx, new_grad_output)', {'_get_net_argument': _get_net_argument, 'grad_op': grad_op, 'idx': idx, 'new_grad_output': new_grad_output}, 0)

