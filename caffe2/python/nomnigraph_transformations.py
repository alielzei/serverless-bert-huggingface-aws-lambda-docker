from __future__ import absolute_import, division, print_function, unicode_literals
from collections import defaultdict
import caffe2.python.nomnigraph as ng
from caffe2.python import core, utils

def transpose_network(nn):
    """
    Convert all Convolutions operators which are in the NCHW order
    to NHWC order and also transform their inputs and outputs so that the
    rest of the graph is not affected.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.nomnigraph_transformations.transpose_network', 'transpose_network(nn)', {'defaultdict': defaultdict, 'utils': utils, 'ng': ng, 'core': core, 'nn': nn}, 0)

