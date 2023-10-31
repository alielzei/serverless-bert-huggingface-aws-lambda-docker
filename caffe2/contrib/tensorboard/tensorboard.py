from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import click
import collections
import logging
import numpy as np
import os
from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.contrib.tensorboard.tensorboard_exporter as tb_exporter
try:
    from tensorboard.compat.proto.summary_pb2 import Summary, HistogramProto
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.summary.writer.event_file_writer import EventFileWriter as FileWriter
except ImportError:
    from tensorflow.core.framework.summary_pb2 import Summary, HistogramProto
    from tensorflow.core.util.event_pb2 import Event
    try:
        from tensorflow.summary import FileWriter
    except ImportError:
        from tensorflow.train import SummaryWriter as FileWriter


class Config(object):
    HEIGHT = 600
    ASPECT_RATIO = 1.6

CODE_TEMPLATE = '\n<script>\n  function load() {{\n    document.getElementById("{id}").pbtxt = {data};\n  }}\n</script>\n<link rel="import"\n  href="https://tensorboard.appspot.com/tf-graph-basic.build.html"\n  onload=load()\n>\n<div style="height:{height}px">\n  <tf-graph-basic id="{id}"></tf-graph-basic>\n</div>\n'
IFRAME_TEMPLATE = '\n<iframe\n  seamless\n  style="width:{width}px;height:{height}px;border:0"\n  srcdoc="{code}">\n</iframe>\n'

def _show_graph(graph_def):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard._show_graph', '_show_graph(graph_def)', {'CODE_TEMPLATE': CODE_TEMPLATE, 'np': np, 'Config': Config, 'IFRAME_TEMPLATE': IFRAME_TEMPLATE, 'IPython': IPython, 'graph_def': graph_def}, 0)

def visualize_cnn(cnn, **kwargs):
    g = tb_exporter.cnn_to_graph_def(cnn, **kwargs)
    _show_graph(g)

def visualize_net(nets, **kwargs):
    g = tb_exporter.nets_to_graph_def(nets, **kwargs)
    _show_graph(g)

def visualize_ops(ops, **kwargs):
    g = tb_exporter.ops_to_graph_def(ops, **kwargs)
    _show_graph(g)

@click.group()
def cli():
    pass

def write_events(tf_dir, events):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard.write_events', 'write_events(tf_dir, events)', {'FileWriter': FileWriter, 'tf_dir': tf_dir, 'events': events}, 0)

def graph_def_to_event(step, graph_def):
    return Event(wall_time=step, step=step, graph_def=graph_def.SerializeToString())

@cli.command('tensorboard-graphs')
@click.option('--c2-netdef', type=click.Path(exists=True, dir_okay=False), multiple=True)
@click.option('--tf-dir', type=click.Path(exists=True))
def tensorboard_graphs(c2_netdef, tf_dir):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard.tensorboard_graphs', 'tensorboard_graphs(c2_netdef, tf_dir)', {'logging': logging, '__name__': __name__, 'caffe2_pb2': caffe2_pb2, 'google': google, 'core': core, 'tb_exporter': tb_exporter, 'graph_def_to_event': graph_def_to_event, 'write_events': write_events, 'cli': cli, 'click': click, 'c2_netdef': c2_netdef, 'tf_dir': tf_dir}, 1)

@cli.command('tensorboard-events')
@click.option('--c2-dir', type=click.Path(exists=True, file_okay=False), help='Root directory of the Caffe2 run')
@click.option('--tf-dir', type=click.Path(writable=True), help='Output path to the logdir used by TensorBoard')
def tensorboard_events(c2_dir, tf_dir):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.tensorboard.tensorboard.tensorboard_events', 'tensorboard_events(c2_dir, tf_dir)', {'np': np, 'logging': logging, '__name__': __name__, 'collections': collections, 'os': os, 'HistogramProto': HistogramProto, 'Summary': Summary, 'Event': Event, 'write_events': write_events, 'cli': cli, 'click': click, 'c2_dir': c2_dir, 'tf_dir': tf_dir}, 1)
if __name__ == '__main__':
    cli()

