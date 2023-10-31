from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import json
import logging
from collections import defaultdict
from caffe2.python import utils
from future.utils import viewitems
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
try:
    import pydot
except ImportError:
    logger.info('Cannot import pydot, which is required for drawing a network. This can usually be installed in python with "pip install pydot". Also, pydot requires graphviz to convert dot files to pdf: in ubuntu, this can usually be installed with "sudo apt-get install graphviz".')
    print('net_drawer will not run correctly. Please install the correct dependencies.')
    pydot = None
from caffe2.proto import caffe2_pb2
OP_STYLE = {'shape': 'box', 'color': '#0F9D58', 'style': 'filled', 'fontcolor': '#FFFFFF'}
BLOB_STYLE = {'shape': 'octagon'}

def _rectify_operator_and_name(operators_or_net, name):
    """Gets the operators and name for the pydot graph."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_drawer._rectify_operator_and_name', '_rectify_operator_and_name(operators_or_net, name)', {'caffe2_pb2': caffe2_pb2, 'operators_or_net': operators_or_net, 'name': name}, 2)

def _escape_label(name):
    return json.dumps(name)

def GetOpNodeProducer(append_output, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_drawer.GetOpNodeProducer', 'GetOpNodeProducer(append_output, **kwargs)', {'pydot': pydot, 'append_output': append_output, 'kwargs': kwargs}, 1)

def GetBlobNodeProducer(**kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_drawer.GetBlobNodeProducer', 'GetBlobNodeProducer(**kwargs)', {'pydot': pydot, 'kwargs': kwargs}, 1)

def GetPydotGraph(operators_or_net, name=None, rankdir='LR', op_node_producer=None, blob_node_producer=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_drawer.GetPydotGraph', "GetPydotGraph(operators_or_net, name=None, rankdir='LR', op_node_producer=None, blob_node_producer=None)", {'GetOpNodeProducer': GetOpNodeProducer, 'OP_STYLE': OP_STYLE, 'GetBlobNodeProducer': GetBlobNodeProducer, 'BLOB_STYLE': BLOB_STYLE, '_rectify_operator_and_name': _rectify_operator_and_name, 'pydot': pydot, 'defaultdict': defaultdict, '_escape_label': _escape_label, 'operators_or_net': operators_or_net, 'name': name, 'rankdir': rankdir, 'op_node_producer': op_node_producer, 'blob_node_producer': blob_node_producer}, 1)

def GetPydotGraphMinimal(operators_or_net, name=None, rankdir='LR', minimal_dependency=False, op_node_producer=None):
    """Different from GetPydotGraph, hide all blob nodes and only show op nodes.

    If minimal_dependency is set as well, for each op, we will only draw the
    edges to the minimal necessary ancestors. For example, if op c depends on
    op a and b, and op b depends on a, then only the edge b->c will be drawn
    because a->c will be implied.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_drawer.GetPydotGraphMinimal', "GetPydotGraphMinimal(operators_or_net, name=None, rankdir='LR', minimal_dependency=False, op_node_producer=None)", {'GetOpNodeProducer': GetOpNodeProducer, 'OP_STYLE': OP_STYLE, '_rectify_operator_and_name': _rectify_operator_and_name, 'pydot': pydot, 'defaultdict': defaultdict, 'operators_or_net': operators_or_net, 'name': name, 'rankdir': rankdir, 'minimal_dependency': minimal_dependency, 'op_node_producer': op_node_producer}, 1)

def GetOperatorMapForPlan(plan_def):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_drawer.GetOperatorMapForPlan', 'GetOperatorMapForPlan(plan_def)', {'plan_def': plan_def}, 1)

def _draw_nets(nets, g):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_drawer._draw_nets', '_draw_nets(nets, g)', {'pydot': pydot, '_escape_label': _escape_label, 'nets': nets, 'g': g}, 1)

def _draw_steps(steps, g, skip_step_edges=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_drawer._draw_steps', '_draw_steps(steps, g, skip_step_edges=False)', {'pydot': pydot, '_escape_label': _escape_label, 'OP_STYLE': OP_STYLE, '_draw_nets': _draw_nets, '_draw_steps': _draw_steps, 'steps': steps, 'g': g, 'skip_step_edges': skip_step_edges}, 1)

def GetPlanGraph(plan_def, name=None, rankdir='TB'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_drawer.GetPlanGraph', "GetPlanGraph(plan_def, name=None, rankdir='TB')", {'pydot': pydot, '_draw_steps': _draw_steps, 'plan_def': plan_def, 'name': name, 'rankdir': rankdir}, 1)

def GetGraphInJson(operators_or_net, output_filepath):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_drawer.GetGraphInJson', 'GetGraphInJson(operators_or_net, output_filepath)', {'_rectify_operator_and_name': _rectify_operator_and_name, 'defaultdict': defaultdict, '_escape_label': _escape_label, 'json': json, 'operators_or_net': operators_or_net, 'output_filepath': output_filepath}, 0)
_DummyPngImage = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x01\x00\x00\x00\x007n\xf9$\x00\x00\x00\nIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82'

def GetGraphPngSafe(func, *args, **kwargs):
    """
    Invokes `func` (e.g. GetPydotGraph) with args. If anything fails - returns
    and empty image instead of throwing Exception
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.net_drawer.GetGraphPngSafe', 'GetGraphPngSafe(func, *args, **kwargs)', {'pydot': pydot, 'logger': logger, '_DummyPngImage': _DummyPngImage, 'func': func, 'args': args, 'kwargs': kwargs}, 1)

def main():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.net_drawer.main', 'main()', {'argparse': argparse, 'utils': utils, 'caffe2_pb2': caffe2_pb2, 'GetOperatorMapForPlan': GetOperatorMapForPlan, 'viewitems': viewitems, 'GetPydotGraphMinimal': GetPydotGraphMinimal, 'GetOpNodeProducer': GetOpNodeProducer, 'OP_STYLE': OP_STYLE, 'GetPydotGraph': GetPydotGraph}, 0)
if __name__ == '__main__':
    main()

