from collections import OrderedDict
from tensorboard.compat.proto.config_pb2 import RunMetadata
from tensorboard.compat.proto.graph_pb2 import GraphDef
from tensorboard.compat.proto.step_stats_pb2 import StepStats, DeviceStepStats
from tensorboard.compat.proto.versions_pb2 import VersionDef
import torch
from ._proto_graph import node_proto
methods_OP = ['attributeNames', 'hasMultipleOutputs', 'hasUses', 'inputs', 'kind', 'outputs', 'outputsSize', 'scopeName']
methods_IO = ['node', 'offset', 'debugName']
GETATTR_KIND = 'prim::GetAttr'
CLASSTYPE_KIND = 'ClassType'


class NodeBase(object):
    
    def __init__(self, debugName=None, inputs=None, scope=None, tensor_size=None, op_type='UnSpecified', attributes=''):
        self.debugName = debugName
        self.inputs = inputs
        self.tensor_size = tensor_size
        self.kind = op_type
        self.attributes = attributes
        self.scope = scope
    
    def __repr__(self):
        repr = []
        repr.append(str(type(self)))
        for m in dir(self):
            if '__' not in m:
                repr.append(m + ': ' + str(getattr(self, m)) + str(type(getattr(self, m))))
        return '\n'.join(repr) + '\n\n'



class NodePy(NodeBase):
    
    def __init__(self, node_cpp, valid_methods):
        super(NodePy, self).__init__(node_cpp)
        valid_methods = valid_methods[:]
        self.inputs = []
        for m in valid_methods:
            if (m == 'inputs' or m == 'outputs'):
                list_of_node = list(getattr(node_cpp, m)())
                io_unique_names = []
                io_tensor_sizes = []
                for n in list_of_node:
                    io_unique_names.append(n.debugName())
                    if n.isCompleteTensor():
                        io_tensor_sizes.append(n.type().sizes())
                    else:
                        io_tensor_sizes.append(None)
                setattr(self, m, io_unique_names)
                setattr(self, m + 'tensor_size', io_tensor_sizes)
            else:
                setattr(self, m, getattr(node_cpp, m)())



class NodePyIO(NodePy):
    
    def __init__(self, node_cpp, input_or_output=None):
        super(NodePyIO, self).__init__(node_cpp, methods_IO)
        try:
            tensor_size = node_cpp.type().sizes()
        except RuntimeError:
            tensor_size = [1]
        self.tensor_size = tensor_size
        self.kind = 'Parameter'
        if input_or_output:
            self.input_or_output = input_or_output
            self.kind = 'IO Node'



class NodePyOP(NodePy):
    
    def __init__(self, node_cpp):
        super(NodePyOP, self).__init__(node_cpp, methods_OP)
        self.attributes = str({k: node_cpp[k] for k in node_cpp.attributeNames()}).replace("'", ' ')
        self.kind = node_cpp.kind()



class GraphPy(object):
    """Helper class to convert torch.nn.Module to GraphDef proto and visualization
    with TensorBoard.

    GraphDef generation operates in two passes:

    In the first pass, all nodes are read and saved to two lists.
    One list is for input/output nodes (nodes_io), which only have inbound
    or outbound connections, but not both. Another list is for internal
    operator nodes (nodes_op). The first pass also saves all scope name
    appeared in the nodes in scope_name_appeared list for later processing.

    In the second pass, scope names are fully applied to all nodes.
    debugNameToScopedName is a mapping from a node's ID to its fully qualified
    scope name. e.g. Net1/Linear[0]/1. Unfortunately torch.jit doesn't have
    totally correct scope output, so this is nontrivial. The function
    populate_namespace_from_OP_to_IO and find_common_root are used to
    assign scope name to a node based on the connection between nodes
    in a heuristic kind of way. Bookkeeping is done with shallowest_scope_name
    and scope_name_appeared.
    """
    
    def __init__(self):
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = 'default'
        self.scope_name_appeared = []
    
    def append(self, x):
        if isinstance(x, NodePyIO):
            self.nodes_io[x.debugName] = x
        if isinstance(x, NodePyOP):
            self.nodes_op.append(x)
    
    def printall(self):
        print('all nodes')
        for node in self.nodes_op:
            print(node)
        for key in self.nodes_io:
            print(self.nodes_io[key])
    
    def find_common_root(self):
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split('/')[0]
    
    def populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_op:
            for (node_output, outputSize) in zip(node.outputs, node.outputstensor_size):
                self.scope_name_appeared.append(node.scopeName)
                self.nodes_io[node_output] = NodeBase(node_output, node.inputs, node.scopeName, outputSize, op_type=node.kind, attributes=node.attributes)
        self.find_common_root()
        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.unique_name_to_scoped_name[input_node_id] = node.scopeName + '/' + input_node_id
        for (key, node) in self.nodes_io.items():
            if type(node) == NodeBase:
                self.unique_name_to_scoped_name[key] = node.scope + '/' + node.debugName
            if hasattr(node, 'input_or_output'):
                self.unique_name_to_scoped_name[key] = node.input_or_output + '/' + node.debugName
            if (hasattr(node, 'scope') and node.scope is not None):
                self.unique_name_to_scoped_name[key] = node.scope + '/' + node.debugName
                if (node.scope == '' and self.shallowest_scope_name):
                    self.unique_name_to_scoped_name[node.debugName] = self.shallowest_scope_name + '/' + node.debugName
        for (key, node) in self.nodes_io.items():
            self.nodes_io[key].inputs = [self.unique_name_to_scoped_name[node_input_id] for node_input_id in node.inputs]
            if node.debugName in self.unique_name_to_scoped_name:
                self.nodes_io[key].debugName = self.unique_name_to_scoped_name[node.debugName]
    
    def to_proto(self):
        """
        Converts graph representation of GraphPy object to TensorBoard
        required format.
        """
        nodes = []
        for v in self.nodes_io.values():
            nodes.append(node_proto(v.debugName, input=v.inputs, outputsize=v.tensor_size, op=v.kind, attributes=v.attributes))
        return nodes


def parse(graph, trace, args=None, omit_useless_nodes=True):
    """This method parses an optimized PyTorch model graph and produces
    a list of nodes and node stats for eventual conversion to TensorBoard
    protobuf format.

    Args:
      graph (PyTorch module): The model graph to be parsed.
      trace (PyTorch JIT TracedModule): The model trace to be parsed.
      args (tuple): input tensor[s] for the model.
      omit_useless_nodes (boolean): Whether to remove nodes from the graph.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._pytorch_graph.parse', 'parse(graph, trace, args=None, omit_useless_nodes=True)', {'GraphPy': GraphPy, 'CLASSTYPE_KIND': CLASSTYPE_KIND, 'NodePyIO': NodePyIO, 'GETATTR_KIND': GETATTR_KIND, 'NodePyOP': NodePyOP, 'torch': torch, 'graph': graph, 'trace': trace, 'args': args, 'omit_useless_nodes': omit_useless_nodes}, 1)

def graph(model, args, verbose=False):
    """
    This method processes a PyTorch model and produces a `GraphDef` proto
    that can be logged to TensorBoard.

    Args:
      model (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      verbose (bool): Whether to print out verbose information while
        processing.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.tensorboard._pytorch_graph.graph', 'graph(model, args, verbose=False)', {'torch': torch, 'parse': parse, 'RunMetadata': RunMetadata, 'StepStats': StepStats, 'DeviceStepStats': DeviceStepStats, 'GraphDef': GraphDef, 'VersionDef': VersionDef, 'model': model, 'args': args, 'verbose': verbose}, 2)

