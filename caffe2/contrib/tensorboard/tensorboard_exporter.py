from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import bytes
import copy
import logging
import os
import six
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
try:
    from tensorboard.compat.proto import tensor_shape_pb2
    from tensorboard.compat.proto.node_def_pb2 import NodeDef
    from tensorboard.compat.proto.graph_pb2 import GraphDef
except ImportError:
    from tensorflow.core.framework import tensor_shape_pb2
    try:
        from tensorflow import NodeDef, GraphDef
    except ImportError:
        from tensorflow.core.framework.graph_pb2 import NodeDef, GraphDef

def _make_unique_name(seen, name, min_version=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._make_unique_name', '_make_unique_name(seen, name, min_version=0)', {'seen': seen, 'name': name, 'min_version': min_version}, 1)

def _convert_to_ssa(shapes, track_blob_names, ops):
    """
    Convert an operator graph to SSA (i.e. out-of-place).

    I.e. blobs will be renamed so that each blob is produced only once.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._convert_to_ssa', '_convert_to_ssa(shapes, track_blob_names, ops)', {'core': core, '_make_unique_name': _make_unique_name, 'shapes': shapes, 'track_blob_names': track_blob_names, 'ops': ops}, 1)

def _get_blob_names(ops):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._get_blob_names', '_get_blob_names(ops)', {'ops': ops}, 1)

def _remap_keys(m, f):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._remap_keys', '_remap_keys(m, f)', {'six': six, 'm': m, 'f': f}, 0)

def _rename_all(shapes, track_blob_names, ops, f):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._rename_all', '_rename_all(shapes, track_blob_names, ops, f)', {'_make_unique_name': _make_unique_name, '_remap_keys': _remap_keys, 'shapes': shapes, 'track_blob_names': track_blob_names, 'ops': ops, 'f': f}, 1)

def _add_gradient_scope(shapes, track_blob_names, ops):
    """
    For all operators or blobs with name containing "_grad", add a
    "GRADIENTS/" scope.

    Note: breaks graph execution since the blob -> gradient mapping is
    hardcoded.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._add_gradient_scope', '_add_gradient_scope(shapes, track_blob_names, ops)', {'_rename_all': _rename_all, 'shapes': shapes, 'track_blob_names': track_blob_names, 'ops': ops}, 1)

def _replace_colons(shapes, track_blob_names, ops, repl):
    """
    `:i` has a special meaning in Tensorflow.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._replace_colons', '_replace_colons(shapes, track_blob_names, ops, repl)', {'_rename_all': _rename_all, 'shapes': shapes, 'track_blob_names': track_blob_names, 'ops': ops, 'repl': repl}, 1)

def _fill_missing_operator_names(ops):
    """ Give missing operators a name.

    We expect C2 operators to be generally unnamed. This gives them a scope
    (inferred from their outputs) and a name after their type. Duplicates will
    be postfixed by an index.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._fill_missing_operator_names', '_fill_missing_operator_names(ops)', {'os': os, '_make_unique_name': _make_unique_name, 'ops': ops}, 0)

def _tf_device(device_option):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._tf_device', '_tf_device(device_option)', {'caffe2_pb2': caffe2_pb2, 'device_option': device_option}, 1)

def _add_tf_shape(m, ints):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._add_tf_shape', '_add_tf_shape(m, ints)', {'tensor_shape_pb2': tensor_shape_pb2, 'm': m, 'ints': ints}, 0)

def _set_tf_attr(m, arg):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._set_tf_attr', '_set_tf_attr(m, arg)', {'_add_tf_shape': _add_tf_shape, 'm': m, 'arg': arg}, 1)

def _operator_to_node(shapes, op):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._operator_to_node', '_operator_to_node(shapes, op)', {'NodeDef': NodeDef, '_tf_device': _tf_device, '_add_tf_shape': _add_tf_shape, '_set_tf_attr': _set_tf_attr, 'shapes': shapes, 'op': op}, 1)

def _blob_to_node(producing_ops, shapes, name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._blob_to_node', '_blob_to_node(producing_ops, shapes, name)', {'NodeDef': NodeDef, '_tf_device': _tf_device, '_add_tf_shape': _add_tf_shape, 'producing_ops': producing_ops, 'shapes': shapes, 'name': name}, 1)

def _operators_to_graph_def(shapes, ops, replace_colons='$', with_ssa=True, with_gradient_scope=True, track_blob_names=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._operators_to_graph_def', "_operators_to_graph_def(shapes, ops, replace_colons='$', with_ssa=True, with_gradient_scope=True, track_blob_names=None)", {'_get_blob_names': _get_blob_names, '_replace_colons': _replace_colons, '_convert_to_ssa': _convert_to_ssa, '_add_gradient_scope': _add_gradient_scope, '_fill_missing_operator_names': _fill_missing_operator_names, 'GraphDef': GraphDef, '_operator_to_node': _operator_to_node, '_blob_to_node': _blob_to_node, 'shapes': shapes, 'ops': ops, 'replace_colons': replace_colons, 'with_ssa': with_ssa, 'with_gradient_scope': with_gradient_scope, 'track_blob_names': track_blob_names}, 1)

def _propagate_device_option(net):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._propagate_device_option', '_propagate_device_option(net)', {'net': net}, 1)

def _try_get_shapes(nets):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter._try_get_shapes', '_try_get_shapes(nets)', {'workspace': workspace, 'logging': logging, 'nets': nets}, 1)

def nets_to_graph_def(nets, shapes=None, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter.nets_to_graph_def', 'nets_to_graph_def(nets, shapes=None, **kwargs)', {'_try_get_shapes': _try_get_shapes, 'copy': copy, '_propagate_device_option': _propagate_device_option, '_operators_to_graph_def': _operators_to_graph_def, 'nets': nets, 'shapes': shapes, 'kwargs': kwargs}, 1)

def cnn_to_graph_def(cnn, **kwargs):
    return nets_to_graph_def([cnn.param_init_net, cnn.net], **kwargs)

def ops_to_graph_def(ops, shapes=None, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard_exporter.ops_to_graph_def', 'ops_to_graph_def(ops, shapes=None, **kwargs)', {'copy': copy, '_operators_to_graph_def': _operators_to_graph_def, 'ops': ops, 'shapes': shapes, 'kwargs': kwargs}, 1)

