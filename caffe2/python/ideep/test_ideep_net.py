from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.models.download import ModelDownloader
import numpy as np
import argparse
import time
import os.path

def GetArgumentParser():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.ideep.test_ideep_net.GetArgumentParser', 'GetArgumentParser()', {'argparse': argparse}, 1)

def benchmark(args):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.ideep.test_ideep_net.benchmark', 'benchmark(args)', {'ModelDownloader': ModelDownloader, 'np': np, 'core': core, 'caffe2_pb2': caffe2_pb2, 'workspace': workspace, 'time': time, 'args': args}, 0)
if __name__ == '__main__':
    (args, extra_args) = GetArgumentParser().parse_known_args()
    if (not args.batch_size or not args.model or not args.order):
        GetArgumentParser().print_help()
    benchmark(args)

