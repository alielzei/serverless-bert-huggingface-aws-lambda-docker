from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.node_def_pb2 import NodeDef
from tensorboard.compat.proto.versions_pb2 import VersionDef
from tensorboard.compat.proto.attr_value_pb2 import AttrValue
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto

def load_onnx_graph(fname):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._onnx_graph.load_onnx_graph', 'load_onnx_graph(fname)', {'parse': parse, 'fname': fname}, 1)

def parse(graph):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._onnx_graph.parse', 'parse(graph)', {'TensorShapeProto': TensorShapeProto, 'NodeDef': NodeDef, 'AttrValue': AttrValue, 'GraphDef': GraphDef, 'VersionDef': VersionDef, 'graph': graph}, 1)

