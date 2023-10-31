from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import string
import argparse
import numpy as np
from caffe2.python.model_helper import ModelHelper
from caffe2.python.predictor import mobile_exporter
from caffe2.python import core, workspace, brew, utils

def parse_kwarg(kwarg_str):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.benchmark_generator.parse_kwarg', 'parse_kwarg(kwarg_str)', {'string': string, 'kwarg_str': kwarg_str}, 2)

def main(args):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.benchmark_generator.main', 'main(args)', {'ModelHelper': ModelHelper, 'brew': brew, 'workspace': workspace, 'core': core, 'utils': utils, 'np': np, 'mobile_exporter': mobile_exporter, 'args': args}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utilitity to generate Caffe2 benchmark models.')
    parser.add_argument('operator', help='Caffe2 operator to benchmark.')
    parser.add_argument('-b', '--blob', help='Instantiate a blob --blob name=dim1,dim2,dim3', action='append')
    parser.add_argument('--context', help='Context to run on.', default='CPU')
    parser.add_argument('--kwargs', help='kwargs to pass to operator.', nargs='*', type=parse_kwarg, default=[])
    parser.add_argument('--init_net', help='Output initialization net.', default='init_net.pb')
    parser.add_argument('--predict_net', help='Output prediction net.', default='predict_net.pb')
    parser.add_argument('--benchmark_name', help='Name of the benchmark network', default='benchmark')
    parser.add_argument('--input_name', help='Name of the input blob.', default='data')
    parser.add_argument('--output_name', help='Name of the output blob.', default='output')
    parser.add_argument('--iters', help='Number of iterations to run the operator.', default='1')
    parser.add_argument('-d', '--debug', help='Print debug information.', action='store_true')
    parser.add_argument('-c', '--chain', help='Chain ops together (create data dependencies)', action='store_true')
    args = parser.parse_args()
    main(args)

