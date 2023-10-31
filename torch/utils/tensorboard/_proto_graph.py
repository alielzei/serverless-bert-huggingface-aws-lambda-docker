from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto

def attr_value_proto(dtype, shape, s):
    """Creates a dict of objects matching
    https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/attr_value.proto
    specifically designed for a NodeDef. The values have been
    reverse engineered from standard TensorBoard logged data.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._proto_graph.attr_value_proto', 'attr_value_proto(dtype, shape, s)', {'AttrValue': AttrValue, 'tensor_shape_proto': tensor_shape_proto, 'dtype': dtype, 'shape': shape, 's': s}, 1)

def tensor_shape_proto(outputsize):
    """Creates an object matching
    https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/tensor_shape.proto
    """
    return TensorShapeProto(dim=[TensorShapeProto.Dim(size=d) for d in outputsize])

def node_proto(name, op='UnSpecified', input=None, dtype=None, shape=None, outputsize=None, attributes=''):
    """Creates an object matching
    https://github.com/tensorflow/tensorboard/blob/master/tensorboard/compat/proto/node_def.proto
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._proto_graph.node_proto', "node_proto(name, op='UnSpecified', input=None, dtype=None, shape=None, outputsize=None, attributes='')", {'NodeDef': NodeDef, 'attr_value_proto': attr_value_proto, 'name': name, 'op': op, 'input': input, 'dtype': dtype, 'shape': shape, 'outputsize': outputsize, 'attributes': attributes}, 1)

