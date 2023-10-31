from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import copy
import json
import os.path
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, utils
import caffe2.python._import_c_extension as C

def pairwise(iterable):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.pairwise', 'pairwise(iterable)', {'iterable': iterable}, 1)

def last_producer(ops, blob):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.last_producer', 'last_producer(ops, blob)', {'ops': ops, 'blob': blob}, 1)

def blob_uses(net, blob):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.blob_uses', 'blob_uses(net, blob)', {'net': net, 'blob': blob}, 1)

def GetArgumentParser():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.GetArgumentParser', 'GetArgumentParser()', {'argparse': argparse}, 1)

def fuse_first_bn(net, params, removed_tensors):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.fuse_first_bn', 'fuse_first_bn(net, params, removed_tensors)', {'copy': copy, 'pairwise': pairwise, 'blob_uses': blob_uses, 'np': np, 'net': net, 'params': params, 'removed_tensors': removed_tensors}, 3)

def fuse_bn(net, params, ignore_failure):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.fuse_bn', 'fuse_bn(net, params, ignore_failure)', {'fuse_first_bn': fuse_first_bn, 'net': net, 'params': params, 'ignore_failure': ignore_failure}, 3)

def fuse_first_mul_add(net, params, removed_tensors):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.fuse_first_mul_add', 'fuse_first_mul_add(net, params, removed_tensors)', {'copy': copy, 'pairwise': pairwise, 'blob_uses': blob_uses, 'log': log, 'utils': utils, 'np': np, 'net': net, 'params': params, 'removed_tensors': removed_tensors}, 1)

def fuse_mul_add(net, params):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.fuse_mul_add', 'fuse_mul_add(net, params)', {'fuse_first_mul_add': fuse_first_mul_add, 'net': net, 'params': params}, 3)

def add_tensor(net, name, blob):
    """ Create an operator to store the tensor 'blob',
        run the operator to put the blob to workspace.
        uint8 is stored as an array of string with one element.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.add_tensor', 'add_tensor(net, name, blob)', {'np': np, 'core': core, 'utils': utils, 'net': net, 'name': name, 'blob': blob}, 0)

def gen_init_net_from_blobs(blobs):
    """ Generate an initialization net based on a blob dict """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.gen_init_net_from_blobs', 'gen_init_net_from_blobs(blobs)', {'caffe2_pb2': caffe2_pb2, 'add_tensor': add_tensor, 'blobs': blobs}, 1)

def fuse_conv_relu(net):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.fuse_conv_relu', 'fuse_conv_relu(net)', {'copy': copy, 'core': core, 'caffe2_pb2': caffe2_pb2, 'C': C, 'net': net}, 1)

def Optimize(args):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.ideep.transform_ideep_net.Optimize', 'Optimize(args)', {'caffe2_pb2': caffe2_pb2, 'workspace': workspace, 'json': json, 'np': np, 'fuse_mul_add': fuse_mul_add, 'fuse_bn': fuse_bn, 'fuse_conv_relu': fuse_conv_relu, 'core': core, 'gen_init_net_from_blobs': gen_init_net_from_blobs, 'args': args}, 0)
if __name__ == '__main__':
    args = GetArgumentParser().parse_args()
    Optimize(args)

