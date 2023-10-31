from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, lstm_benchmark, utils
from copy import copy

@utils.debug
def Compare(args):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.rnn.lstm_comparison.Compare', 'Compare(args)', {'core': core, 'workspace': workspace, 'lstm_benchmark': lstm_benchmark, 'copy': copy, 'utils': utils, 'args': args}, 0)
if __name__ == '__main__':
    args = lstm_benchmark.GetArgumentParser().parse_args()
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0', '--caffe2_print_blob_sizes_at_exit=0', '--caffe2_gpu_memory_tracking=1'])
    Compare(args)

