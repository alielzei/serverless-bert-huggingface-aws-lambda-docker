from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
import logging
import os
import re
import six
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from builtins import bytes
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

def _make_unique_name(seen, name, min_version=0):
    """
    Make the name unique by appending a unique number to the name. Used for SSA.

    Args:
        seen (set): Set of names that have already been used (with respect to
            some context).
        name (string): The name to make unique
        min_version (number): Starting index. Is incremented continually until
            it can make the resulting name unique relative to 'seen'.

    Returns:
        x (string): A version of name that is not in seen.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._make_unique_name', '_make_unique_name(seen, name, min_version=0)', {'seen': seen, 'name': name, 'min_version': min_version}, 1)

def _rename_tensorflow_style(shapes, blob_name_tracker, ops):
    """
    Convert some of the common names in Caffe2 to tensorflow.
    NOTE: The common names in both Caffe2 and Tensorflow are currently
        hardcoded, if either side changes at some point, then this code should
        change as well.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        blob_name_tracker: Dictionary of all unique blob names (with respect to
            some context).
        ops: List of Caffe2 operators

    Returns:
        None. The _rename_all() call modifies blob_name_tracker and ops in-place.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._rename_tensorflow_style', '_rename_tensorflow_style(shapes, blob_name_tracker, ops)', {'re': re, '_rename_all': _rename_all, 'shapes': shapes, 'blob_name_tracker': blob_name_tracker, 'ops': ops}, 1)

def _convert_to_ssa(shapes, blob_name_tracker, ops):
    """
    Convert an operator graph to SSA (i.e. out-of-place).
    i.e. blobs will be renamed so that each blob is produced only once.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        blob_name_tracker: Dictionary of all unique blob names (with respect to
            some context).
        ops: List of Caffe2 operators

    Returns:
        None. Modifies blob_name_tracker and ops in-place.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._convert_to_ssa', '_convert_to_ssa(shapes, blob_name_tracker, ops)', {'core': core, '_make_unique_name': _make_unique_name, 'shapes': shapes, 'blob_name_tracker': blob_name_tracker, 'ops': ops}, 1)

def _get_blob_names(ops):
    """
    Get all the operator input and output blobs and perform dedup on their names.

    Args:
        ops: List of Caffe2 operators to extract inputs and outputs from

    Returns:
        set containing distinct inputs and outputs from 'ops'
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._get_blob_names', '_get_blob_names(ops)', {'ops': ops}, 1)

def _remap_keys(old_dict, rename_fn):
    """
    Rename keys of 'old_dict' according to 'rename_fn'.

    Args:
        old_dict: Dictionary (i.e. containing blob_name -> blob_name
            relationships.)
        remap_fn: Function string -> string for renaming.

    Returns:
        None. Modifies old_dict in-place.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._remap_keys', '_remap_keys(old_dict, rename_fn)', {'six': six, 'old_dict': old_dict, 'rename_fn': rename_fn}, 0)

def _rename_all(shapes, blob_name_tracker, ops, rename_fn):
    """
    Rename all the names in the operators.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        blob_name_tracker: Dictionary of all unique blob names (with respect to
            some context).
        ops: List of Caffe2 operators
        rename_fn: Function string -> string that specifies how to rename

    Returns:
        None. Modifies shapes, blob_name_tracker and ops in-place using the
            specified 'rename_fn'.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._rename_all', '_rename_all(shapes, blob_name_tracker, ops, rename_fn)', {'_make_unique_name': _make_unique_name, '_remap_keys': _remap_keys, 'shapes': shapes, 'blob_name_tracker': blob_name_tracker, 'ops': ops, 'rename_fn': rename_fn}, 1)

def _add_gradient_scope(shapes, blob_name_tracker, ops):
    """
    For all operators or blobs with name containing "_grad", add a
    "GRADIENTS/" scope.
    Note: breaks graph execution since the blob -> gradient mapping is
    hardcoded.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        blob_name_tracker: Dictionary of all unique blob names (with respect to
            some context).
        ops: List of Caffe2 operators

    Returns:
        None. Modifies shapes, blob_name_tracker and ops in-place by renaming.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._add_gradient_scope', '_add_gradient_scope(shapes, blob_name_tracker, ops)', {'_rename_all': _rename_all, 'shapes': shapes, 'blob_name_tracker': blob_name_tracker, 'ops': ops}, 1)

def _replace_colons(shapes, blob_name_tracker, ops, repl):
    """
    `:i` has a special meaning in Tensorflow. This function replaces all colons
    with $ to avoid any possible conflicts.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        blob_name_tracker: Dictionary of all unique blob names (with respect to
            some context).
        ops: List of Caffe2 operators
        repl: String representing the text to replace ':' with. Usually this is
            '$'.

    Returns:
        None. Modifies blob_name_tracker in-place.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._replace_colons', '_replace_colons(shapes, blob_name_tracker, ops, repl)', {'_rename_all': _rename_all, 'shapes': shapes, 'blob_name_tracker': blob_name_tracker, 'ops': ops, 'repl': repl}, 1)

def _fill_missing_operator_names(ops):
    """
    Give missing operators a name.
    We expect C2 operators to be generally unnamed. This gives them a scope
    (inferred from their outputs) and a name after their type. Duplicates will
    be postfixed by an index.

    Args:
        ops: List of Caffe2 operators to assign names to.

    Returns:
        None: Modifies 'ops' in-place.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._fill_missing_operator_names', '_fill_missing_operator_names(ops)', {'os': os, '_make_unique_name': _make_unique_name, 'ops': ops}, 0)

def _tf_device(device_option):
    """
    Handle the devices.

    Args:
        device_option (caffe2_pb2.DeviceOption): DeviceOption protobuf,
            associated to an operator, that contains information such as
            device_type (optional), cuda_gpu_id (optional), node_name (optional,
            tells which node the operator should execute on). See caffe2.proto
            in caffe2/proto for the full list.

    Returns:
        Formatted string representing device information contained in
            device_option.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._tf_device', '_tf_device(device_option)', {'caffe2_pb2': caffe2_pb2, 'device_option': device_option}, 1)

def _add_tf_shape(attr_dict, ints):
    """
    Converts a list of ints to a TensorShapeProto representing the dimensions of
    a blob/object.

    Args:
        attr_dict: Dictionary to update (usually attributes of a Node)
        ints: List of integers representing dimensions of some object.

    Returns:
        None. Modifies attr_dict in-place.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._add_tf_shape', '_add_tf_shape(attr_dict, ints)', {'TensorShapeProto': TensorShapeProto, 'attr_dict': attr_dict, 'ints': ints}, 0)

def _set_tf_attr(attr_dict, arg):
    """
    Add attributes to a node. Key is the arg.name, and values can be shape,
        floats, strings, ints or an empty list.

    Args:
        attr_dict: Dictionary to update (usually attributes of a Node)
        arg: Object with name and data fields.

    Returns:
        None. Modifies attr_dict in-place.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._set_tf_attr', '_set_tf_attr(attr_dict, arg)', {'_add_tf_shape': _add_tf_shape, 'attr_dict': attr_dict, 'arg': arg}, 1)

def _operator_to_node(shapes, op):
    """
    Converts an operator to a node in a TF graph.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        op: The Caffe2 operator to convert to a TF graph node.

    Returns:
        n: The TF graph node created from op.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._operator_to_node', '_operator_to_node(shapes, op)', {'NodeDef': NodeDef, '_tf_device': _tf_device, '_add_tf_shape': _add_tf_shape, '_set_tf_attr': _set_tf_attr, 'shapes': shapes, 'op': op}, 1)

def _operator_to_node_simp(op, inter_blobs, seen):
    """
    Convert the operators to nodes.

    Args:
        op: Caffe2 operator to convert to node
        inter_blobs: Set of intermediate blobs
        seen: Names that have already been used and are not unique

    Returns:
        nodes: Nodes representing 'op' and the outputs of 'op'
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._operator_to_node_simp', '_operator_to_node_simp(op, inter_blobs, seen)', {'NodeDef': NodeDef, '_tf_device': _tf_device, '_set_tf_attr': _set_tf_attr, 'os': os, '_make_unique_name': _make_unique_name, 'op': op, 'inter_blobs': inter_blobs, 'seen': seen}, 1)

def _blob_to_node(producing_ops, shapes, name):
    """
    Converts a blob (operator input or output) to a node in a TF graph.

    Args:
        producing_ops: Dictionary of blob name to list of
            (producing_op, blob_index within producing_op.output) mapping.
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        name: String representing the name of this blob.

    Returns:
        n: The TF graph node created from this blob.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._blob_to_node', '_blob_to_node(producing_ops, shapes, name)', {'NodeDef': NodeDef, '_tf_device': _tf_device, '_add_tf_shape': _add_tf_shape, 'producing_ops': producing_ops, 'shapes': shapes, 'name': name}, 1)

def _clear_debug_info(ops, perform_clear):
    """
    Removes debug information from operators, they are copious.

    Args:
        ops: List of Caffe2 operators
        perform_clear: Boolean passed from _operators_to_graph_def specifying
            whether to remove the debug information. This boolean is passed into
            this function to reduce the complexity of _operators_to_graph_def.

    Returns:
        None. Modifies the list of Caffe2 operators in-place and removes the
        'debug_info' field.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._clear_debug_info', '_clear_debug_info(ops, perform_clear)', {'ops': ops, 'perform_clear': perform_clear}, 1)

def _check_if_forward(blob):
    """
    Blobs with names containing '_m' or 'grad' are part of the backward pass.
        This function references facebookresearch/Detectron/detectron/utils/net.py.

    Args:
        blob: The blob to inspect

    Returns:
        Boolean representing whether this blob is part of the forward pass
    """
    return (blob.find('__m') < 0 or blob.find('grad') < 0)

def _check_if_cpu(blob):
    """
    Check if the blob's name starts with '_gpu'.

    Args:
        blob: The blob to inspect

    Returns:
        Boolean representing whether this blob is associated with a gpu
    """
    return not blob.startswith('_gpu')

def _compute_in_out(ops):
    """
    Find the input, intermediate and output nodes of a set of operators.

    Args:
        ops: List of Caffe2 operators to look through

    Returns:
        input_blobs: The input nodes of the set of operators
        inter_blobs: The intermediate nodes of the set of operators
        output_blobs: The output nodes of the set of operators
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._compute_in_out', '_compute_in_out(ops)', {'ops': ops}, 3)

def _filter_ops(ops, filter_fn, perform_filter):
    """
    Filter unwanted operators based on criteria in 'filter_fn'.

    Args:
        ops: List of Caffe2 operators to filter
        filter_fn: Criteria function for whether inputs/outputs in an operator
            should be filtered.
        perform_filter: Boolean passed from _operators_to_graph_def specifying
            whether to filter operators

    Returns:
        new_ops: Subset of ops containing a subset of their inputs and outputs.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._filter_ops', '_filter_ops(ops, filter_fn, perform_filter)', {'ops': ops, 'filter_fn': filter_fn, 'perform_filter': perform_filter}, 1)

def _operators_to_graph_def(shapes, ops, colon_replacement='$', with_ssa=True, with_gradient_scope=True, blob_name_tracker=None, show_simplified=False, custom_rename=None):
    """
    Main function to convert set of operators to a graph.

    Args:
        shapes: Dictionary mapping blob names to their shapes/dimensions.
        ops: List of Caffe2 operators, representing some computation graph
        ### **kwargs (model_to_graph_def, nets_to_graph_def, protos_to_graph_def) ###
        colon_replacement: Symbol to replace ':' with. ':i' in TF has a special
            meaning, so we need to replace it with a non-conflicting symbol.
        with_ssa: Boolean
        with_gradient_scope: Boolean
        blob_name_tracker: Dictionary tracking names of blobs (inputs/outputs
            from operators)
        show_simplified: Whether to show a simplified version of the model graph
            Sets all of the following values:
                clear_debug_info: Boolean representing whether to silence debug
                    info (which can be very verbose)
                show_forward_only: Boolean representing whether to only show
                    blobs involved in the forward pass
                show_cpu_only: Boolean representing whether to only show blobs
                    that are not associated with a gpu
                use_tensorflow_naming: Boolean representing whether to convert
                    some common Caffe2 naming conventions to their Tensorflow
                    counterparts
        custom_rename: Function string -> string that defines a custom
            renaming function to use.

    Returns:
        current_graph: GraphDef representing the computation graph formed by the
            set of operators.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._operators_to_graph_def', "_operators_to_graph_def(shapes, ops, colon_replacement='$', with_ssa=True, with_gradient_scope=True, blob_name_tracker=None, show_simplified=False, custom_rename=None)", {'_get_blob_names': _get_blob_names, '_clear_debug_info': _clear_debug_info, '_filter_ops': _filter_ops, '_check_if_forward': _check_if_forward, '_check_if_cpu': _check_if_cpu, '_rename_all': _rename_all, '_replace_colons': _replace_colons, '_convert_to_ssa': _convert_to_ssa, '_add_gradient_scope': _add_gradient_scope, '_fill_missing_operator_names': _fill_missing_operator_names, '_rename_tensorflow_style': _rename_tensorflow_style, '_compute_in_out': _compute_in_out, 'GraphDef': GraphDef, '_operator_to_node_simp': _operator_to_node_simp, '_operator_to_node': _operator_to_node, '_blob_to_node': _blob_to_node, 'shapes': shapes, 'ops': ops, 'colon_replacement': colon_replacement, 'with_ssa': with_ssa, 'with_gradient_scope': with_gradient_scope, 'blob_name_tracker': blob_name_tracker, 'show_simplified': show_simplified, 'custom_rename': custom_rename}, 1)

def _propagate_device_option(net_def):
    """
    Propagate the device options from net to operators.

    Args:
        net_def: A caffe2_pb2.NetDef representing a computation graph. The graph
            consists of Caffe2 operators.

    Returns:
        None. Iterates through all ops contained within the net. For each op,
            modifies the op device_option in-place to be the net device_option
            if the op has no pre-existing device_option, and leaves the op as-is
            if it already has a device_option.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._propagate_device_option', '_propagate_device_option(net_def)', {'net_def': net_def}, 1)

def _try_get_shapes(nets):
    """
    Get missing shapes for all blobs contained in the nets.

    Args:
        nets: List of core.Net to extract blob shape information from.

    Returns:
        Dictionary containing blob name to shape/dimensions mapping. The net
            is a computation graph that is composed of operators, and the
            operators have input and output blobs, each with their own dims.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph._try_get_shapes', '_try_get_shapes(nets)', {'workspace': workspace, 'logging': logging, 'nets': nets}, 1)

def model_to_graph_def(model, **kwargs):
    """
    Convert a Caffe2 model to a Tensorflow graph. This function extracts
    'param_init_net' and 'net' from the model and passes it to nets_to_graph()
    for further processing.

    Args:
        model (cnn.CNNModelHelper, model_helper.ModelHelper): The model to
            extract the nets (instances of core.Net) from.

    Returns:
        Call to nets_to_graph_def() with extracted 'param_init_net', 'net' and
            **kwargs. See _operators_to_graph_def for detailed **kwargs.
    """
    nets = [model.param_init_net, model.net]
    return nets_to_graph_def(nets, **kwargs)

def nets_to_graph_def(nets, shapes=None, **kwargs):
    """
    Convert a set of Caffe2 nets to a Tensorflow graph.

    Args:
        nets: List of core.Nets. core.Net is a wrapper around a NetDef protobuf.
            The corresponding protobuf can be extracted using .Proto().
        shapes: Dictionary mapping blob names to their shapes/dimensions.

    Returns:
        Call to protos_to_graph_def() with the extracted NetDef protobufs and
            **kwargs. See _operators_to_graph_def for detailed **kwargs.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph.nets_to_graph_def', 'nets_to_graph_def(nets, shapes=None, **kwargs)', {'copy': copy, 'protos_to_graph_def': protos_to_graph_def, 'nets': nets, 'shapes': shapes, 'kwargs': kwargs}, 1)

def protos_to_graph_def(net_defs, shapes=None, **kwargs):
    """
    Convert a set of Caffe2 net definitions to a Tensorflow graph.

    Args:
        net_defs: List of caffe2_pb2.NetDef protobufs representing computation
            graphs.
        shapes: Dictionary mapping blob names to their shapes/dimensions.

    Returns:
        Call to _operators_to_graph_def() with the extracted operators from the
            NetDefs and **kwargs. See _operators_to_graph_def for detailed
            **kwargs.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._caffe2_graph.protos_to_graph_def', 'protos_to_graph_def(net_defs, shapes=None, **kwargs)', {'_propagate_device_option': _propagate_device_option, 'copy': copy, '_operators_to_graph_def': _operators_to_graph_def, 'net_defs': net_defs, 'shapes': shapes, 'kwargs': kwargs}, 1)

