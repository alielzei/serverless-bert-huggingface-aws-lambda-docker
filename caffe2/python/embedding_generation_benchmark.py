from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, utils, model_helper
import argparse
import numpy as np
import time
import logging
logging.basicConfig()
log = logging.getLogger('embedding_generation_benchmark')
log.setLevel(logging.DEBUG)

def generate_data(T, batch_size, max_seq_length):
    """
    Fill a queue with input data
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.embedding_generation_benchmark.generate_data', 'generate_data(T, batch_size, max_seq_length)', {'log': log, 'core': core, 'workspace': workspace, 'np': np, 'T': T, 'batch_size': batch_size, 'max_seq_length': max_seq_length}, 1)

def generate_embedding_table(vocab_size, embedding_size):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.embedding_generation_benchmark.generate_embedding_table', 'generate_embedding_table(vocab_size, embedding_size)', {'log': log, 'core': core, 'workspace': workspace, 'vocab_size': vocab_size, 'embedding_size': embedding_size}, 1)

def create_model(args, queue, embedding_table, embedding_size):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.embedding_generation_benchmark.create_model', 'create_model(args, queue, embedding_table, embedding_size)', {'model_helper': model_helper, 'args': args, 'queue': queue, 'embedding_table': embedding_table, 'embedding_size': embedding_size}, 1)

def Caffe2EmbeddingGeneration(args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.embedding_generation_benchmark.Caffe2EmbeddingGeneration', 'Caffe2EmbeddingGeneration(args)', {'generate_data': generate_data, 'generate_embedding_table': generate_embedding_table, 'create_model': create_model, 'workspace': workspace, 'time': time, 'log': log, 'args': args}, 1)

@utils.debug
def Benchmark(args):
    return Caffe2EmbeddingGeneration(args)

def GetArgumentParser():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.embedding_generation_benchmark.GetArgumentParser', 'GetArgumentParser()', {'argparse': argparse}, 1)
if __name__ == '__main__':
    (args, extra_args) = GetArgumentParser().parse_known_args()
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0', '--caffe2_print_blob_sizes_at_exit=0'] + extra_args)
    device = core.DeviceOption(caffe2_pb2.CPU)
    with core.DeviceScope(device):
        Benchmark(args)

