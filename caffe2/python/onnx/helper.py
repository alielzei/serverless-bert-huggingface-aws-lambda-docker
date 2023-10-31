from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from onnx.backend.base import namedtupledict
from caffe2.python.onnx.workspace import Workspace
import caffe2.python._import_c_extension as C
import io
import logging
import time
log = logging.getLogger(__name__)

def c2_native_run_op(op_def, inputs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.onnx.helper.c2_native_run_op', 'c2_native_run_op(op_def, inputs)', {'Workspace': Workspace, 'namedtupledict': namedtupledict, 'op_def': op_def, 'inputs': inputs}, 2)

def c2_native_run_net(init_net, predict_net, inputs, debug_arg=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.onnx.helper.c2_native_run_net', 'c2_native_run_net(init_net, predict_net, inputs, debug_arg=None)', {'Workspace': Workspace, 'namedtupledict': namedtupledict, 'init_net': init_net, 'predict_net': predict_net, 'inputs': inputs, 'debug_arg': debug_arg}, 2)

def load_caffe2_net(file):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.onnx.helper.load_caffe2_net', 'load_caffe2_net(file)', {'caffe2_pb2': caffe2_pb2, 'file': file}, 1)

def save_caffe2_net(net, file, output_txt=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.onnx.helper.save_caffe2_net', 'save_caffe2_net(net, file, output_txt=False)', {'net': net, 'file': file, 'output_txt': output_txt}, 0)

def benchmark_caffe2_model(init_net, predict_net, warmup_iters=3, main_iters=10, layer_details=True):
    """
        Run the benchmark net on the target model.
        Return the execution time per iteration (millisecond).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.onnx.helper.benchmark_caffe2_model', 'benchmark_caffe2_model(init_net, predict_net, warmup_iters=3, main_iters=10, layer_details=True)', {'Workspace': Workspace, 'init_net': init_net, 'predict_net': predict_net, 'warmup_iters': warmup_iters, 'main_iters': main_iters, 'layer_details': layer_details}, 1)

def benchmark_pytorch_model(model, inputs, training=False, warmup_iters=3, main_iters=10, verbose=False):
    """
        Run the model several times, and measure the execution time.
        Return the execution time per iteration (millisecond).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.onnx.helper.benchmark_pytorch_model', 'benchmark_pytorch_model(model, inputs, training=False, warmup_iters=3, main_iters=10, verbose=False)', {'time': time, 'log': log, 'model': model, 'inputs': inputs, 'training': training, 'warmup_iters': warmup_iters, 'main_iters': main_iters, 'verbose': verbose}, 1)

