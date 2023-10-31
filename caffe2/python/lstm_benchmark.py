from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, utils, rnn_cell, model_helper
from caffe2.python import recurrent
import argparse
import numpy as np
import time
import logging
logging.basicConfig()
log = logging.getLogger('lstm_bench')
log.setLevel(logging.DEBUG)

def generate_data(T, shape, num_labels, fixed_shape):
    """
    Fill a queue with input data
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.lstm_benchmark.generate_data', 'generate_data(T, shape, num_labels, fixed_shape)', {'log': log, 'core': core, 'workspace': workspace, 'np': np, 'T': T, 'shape': shape, 'num_labels': num_labels, 'fixed_shape': fixed_shape}, 3)

def create_model(args, queue, label_queue, input_shape):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.lstm_benchmark.create_model', 'create_model(args, queue, label_queue, input_shape)', {'model_helper': model_helper, 'rnn_cell': rnn_cell, 'workspace': workspace, 'np': np, 'recurrent': recurrent, 'args': args, 'queue': queue, 'label_queue': label_queue, 'input_shape': input_shape}, 2)

def Caffe2LSTM(args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.lstm_benchmark.Caffe2LSTM', 'Caffe2LSTM(args)', {'generate_data': generate_data, 'workspace': workspace, 'np': np, 'create_model': create_model, 'time': time, 'log': log, 'utils': utils, 'args': args}, 1)

@utils.debug
def Benchmark(args):
    return Caffe2LSTM(args)

def GetArgumentParser():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.lstm_benchmark.GetArgumentParser', 'GetArgumentParser()', {'argparse': argparse}, 1)
if __name__ == '__main__':
    (args, extra_args) = GetArgumentParser().parse_known_args()
    rnn_executor_opt = (1 if args.rnn_executor else 0)
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0', '--caffe2_print_blob_sizes_at_exit=0', '--caffe2_rnn_executor={}'.format(rnn_executor_opt), '--caffe2_gpu_memory_tracking=1'] + extra_args)
    device = core.DeviceOption((workspace.GpuDeviceType if args.gpu else caffe2_pb2.CPU), 4)
    with core.DeviceScope(device):
        Benchmark(args)

