from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'\nBenchmark for common convnets.\n\n(NOTE: Numbers below prior with missing parameter=update step, TODO to update)\n\nSpeed on Titan X, with 10 warmup steps and 10 main steps and with different\nversions of cudnn, are as follows (time reported below is per-batch time,\nforward / forward+backward):\n\n                    CuDNN V3        CuDNN v4\n                    AlexNet         32.5 / 108.0    27.4 /  90.1\n                    OverFeat       113.0 / 342.3    91.7 / 276.5\n                    Inception      134.5 / 485.8   125.7 / 450.6\n                    VGG (batch 64) 200.8 / 650.0   164.1 / 551.7\n\nSpeed on Inception with varied batch sizes and CuDNN v4 is as follows:\n\nBatch Size   Speed per batch     Speed per image\n16             22.8 /  72.7         1.43 / 4.54\n32             38.0 / 127.5         1.19 / 3.98\n64             67.2 / 233.6         1.05 / 3.65\n128            125.7 / 450.6         0.98 / 3.52\n\nSpeed on Tesla M40, which 10 warmup steps and 10 main steps and with cudnn\nv4, is as follows:\n\nAlexNet         68.4 / 218.1\nOverFeat       210.5 / 630.3\nInception      300.2 / 1122.2\nVGG (batch 64) 405.8 / 1327.7\n\n(Note that these numbers involve a "full" backprop, i.e. the gradient\nwith respect to the input image is also computed.)\n\nTo get the numbers, simply run:\n\nfor MODEL in AlexNet OverFeat Inception; do\nPYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py     --batch_size 128 --model $MODEL --forward_only True\ndone\nfor MODEL in AlexNet OverFeat Inception; do\nPYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py     --batch_size 128 --model $MODEL\ndone\nPYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py     --batch_size 64 --model VGGA --forward_only True\nPYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py     --batch_size 64 --model VGGA\n\nfor BS in 16 32 64 128; do\nPYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py     --batch_size $BS --model Inception --forward_only True\nPYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py     --batch_size $BS --model Inception\ndone\n\nNote that VGG needs to be run at batch 64 due to memory limit on the backward\npass.\n'
import argparse
import time
from caffe2.python import cnn, workspace, core
import caffe2.python.SparseTransformer as SparseTransformer

def MLP(order):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.convnet_benchmarks.MLP', 'MLP(order)', {'cnn': cnn, 'order': order}, 2)

def AlexNet(order):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.convnet_benchmarks.AlexNet', 'AlexNet(order)', {'cnn': cnn, 'order': order}, 2)

def OverFeat(order):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.convnet_benchmarks.OverFeat', 'OverFeat(order)', {'cnn': cnn, 'order': order}, 2)

def VGGA(order):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.convnet_benchmarks.VGGA', 'VGGA(order)', {'cnn': cnn, 'order': order}, 2)

def net_DAG_Builder(model):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.convnet_benchmarks.net_DAG_Builder', 'net_DAG_Builder(model)', {'SparseTransformer': SparseTransformer, 'model': model}, 1)

def _InceptionModule(model, input_blob, input_depth, output_name, conv1_depth, conv3_depths, conv5_depths, pool_depth):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.convnet_benchmarks._InceptionModule', '_InceptionModule(model, input_blob, input_depth, output_name, conv1_depth, conv3_depths, conv5_depths, pool_depth)', {'model': model, 'input_blob': input_blob, 'input_depth': input_depth, 'output_name': output_name, 'conv1_depth': conv1_depth, 'conv3_depths': conv3_depths, 'conv5_depths': conv5_depths, 'pool_depth': pool_depth}, 1)

def Inception(order):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.convnet_benchmarks.Inception', 'Inception(order)', {'cnn': cnn, '_InceptionModule': _InceptionModule, 'order': order}, 2)

def AddInput(model, batch_size, db, db_type):
    """Adds the data input part."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.convnet_benchmarks.AddInput', 'AddInput(model, batch_size, db, db_type)', {'core': core, 'model': model, 'batch_size': batch_size, 'db': db, 'db_type': db_type}, 2)

def AddParameterUpdate(model):
    """ Simple plain SGD update -- not tuned to actually train the models """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.experiments.python.convnet_benchmarks.AddParameterUpdate', 'AddParameterUpdate(model)', {'model': model}, 0)

def Benchmark(model_gen, arg):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.experiments.python.convnet_benchmarks.Benchmark', 'Benchmark(model_gen, arg)', {'AddParameterUpdate': AddParameterUpdate, 'workspace': workspace, 'core': core, 'time': time, 'model_gen': model_gen, 'arg': arg}, 0)

def GetArgumentParser():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.convnet_benchmarks.GetArgumentParser', 'GetArgumentParser()', {'argparse': argparse}, 1)
if __name__ == '__main__':
    args = GetArgumentParser().parse_args()
    if (not args.batch_size or not args.model or not args.order or not args.cudnn_ws):
        GetArgumentParser().print_help()
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    model_map = {'AlexNet': AlexNet, 'OverFeat': OverFeat, 'VGGA': VGGA, 'Inception': Inception, 'MLP': MLP}
    Benchmark(model_map[args.model], args)

