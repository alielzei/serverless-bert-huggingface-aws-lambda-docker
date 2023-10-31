from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import copy
from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.python._import_c_extension as C

def rewrite_init_net_simple(net):
    for op in net.op:
        op.device_option.device_type = caffe2_pb2.IDEEP

def last_producer(ops, blob):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph.last_producer', 'last_producer(ops, blob)', {'ops': ops, 'blob': blob}, 1)

def fix_BoxWithNMSLimit(net):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph.fix_BoxWithNMSLimit', 'fix_BoxWithNMSLimit(net)', {'caffe2_pb2': caffe2_pb2, 'net': net}, 0)

def rewrite_run_net_simple(net):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph.rewrite_run_net_simple', 'rewrite_run_net_simple(net)', {'core': core, 'last_producer': last_producer, 'caffe2_pb2': caffe2_pb2, 'fix_BoxWithNMSLimit': fix_BoxWithNMSLimit, 'net': net}, 1)

def rewrite_run_net_simple_xrayocr_lstm(net):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph.rewrite_run_net_simple_xrayocr_lstm', 'rewrite_run_net_simple_xrayocr_lstm(net)', {'core': core, 'last_producer': last_producer, 'caffe2_pb2': caffe2_pb2, 'fix_BoxWithNMSLimit': fix_BoxWithNMSLimit, 'net': net}, 1)

def rewrite_model_helper_simple(model):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph.rewrite_model_helper_simple', 'rewrite_model_helper_simple(model)', {'copy': copy, 'rewrite_init_net_simple': rewrite_init_net_simple, 'rewrite_run_net_simple': rewrite_run_net_simple, 'model': model}, 1)

