from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import numpy as np
from caffe2.python import core, workspace

def benchmark_mul_gradient(args):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.operator_test.mul_gradient_benchmark.benchmark_mul_gradient', 'benchmark_mul_gradient(args)', {'workspace': workspace, 'np': np, 'core': core, 'args': args}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark for MulGradient.')
    parser.add_argument('-m', type=int, default=9508, help='The number of rows of A')
    parser.add_argument('-n', type=int, default=80, help='The number of columns of A')
    parser.add_argument('-i', '--iteration', type=int, default=100, help='The number of iterations.')
    (args, extra_args) = parser.parse_known_args()
    core.GlobalInit(['python'] + extra_args)
    benchmark_mul_gradient(args)

