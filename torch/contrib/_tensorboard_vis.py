import time
from collections import defaultdict
from functools import partial
try:
    from tensorflow.core.util import event_pb2
    from tensorflow.core.framework import graph_pb2
    from tensorflow.python.summary.writer.writer import FileWriter
except ImportError:
    raise ImportError('TensorBoard visualization of GraphExecutors requires having TensorFlow installed')

def dump_tensorboard_summary(graph_executor, logdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.contrib._tensorboard_vis.dump_tensorboard_summary', 'dump_tensorboard_summary(graph_executor, logdir)', {'FileWriter': FileWriter, 'visualize': visualize, 'event_pb2': event_pb2, 'time': time, 'graph_executor': graph_executor, 'logdir': logdir}, 0)

def visualize(graph, name_prefix='', pb_graph=None, executors_it=None):
    """Visualizes an independent graph, or a graph executor."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.contrib._tensorboard_vis.visualize', "visualize(graph, name_prefix='', pb_graph=None, executors_it=None)", {'graph_pb2': graph_pb2, 'torch': torch, 'visualize_graph_executor': visualize_graph_executor, 'partial': partial, 'visualize': visualize, 'visualize_rec': visualize_rec, 'graph': graph, 'name_prefix': name_prefix, 'pb_graph': pb_graph, 'executors_it': executors_it}, 1)

def visualize_graph_executor(state, name_prefix, pb_graph, inline_graph):
    """Appends the state of a given GraphExecutor to the graph protobuf.

    Arguments:
        state (GraphExecutor or GraphExecutorState): GraphExecutor to display.
        name_prefix (str): Name prefix of the containing subgraph.
        pb_graph (GraphDef): graph to append to.
        inline_graph (callable): a function that handles setting up a value_map,
            so that some graphs in here can be inlined. This is necessary, because
            this will simply be `visualize` for the top-level GraphExecutor,
            or `inline_graph` for all nested ones.

            The signature should look like (Graph, name_prefix) -> ().
            It will be called exactly once.

    The strategy is to embed all different configurations as independent subgraphs,
    while inlining the original graph as the one that actually produces the values.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.contrib._tensorboard_vis.visualize_graph_executor', 'visualize_graph_executor(state, name_prefix, pb_graph, inline_graph)', {'visualize': visualize, 'state': state, 'name_prefix': name_prefix, 'pb_graph': pb_graph, 'inline_graph': inline_graph}, 1)

def visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it=None):
    """Recursive part of visualize (basically skips setting up the input and output nodes)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.contrib._tensorboard_vis.visualize_rec', 'visualize_rec(graph, value_map, name_prefix, pb_graph, executors_it=None)', {'visualize_rec': visualize_rec, 'defaultdict': defaultdict, 'visualize_graph_executor': visualize_graph_executor, 'partial': partial, 'graph': graph, 'value_map': value_map, 'name_prefix': name_prefix, 'pb_graph': pb_graph, 'executors_it': executors_it}, 2)

