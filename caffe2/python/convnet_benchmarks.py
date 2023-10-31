"""
Benchmark for common convnets.

Speed on Titan X, with 10 warmup steps and 10 main steps and with different
versions of cudnn, are as follows (time reported below is per-batch time,
forward / forward+backward):

                    CuDNN V3        CuDNN v4
AlexNet         32.5 / 108.0    27.4 /  90.1
OverFeat       113.0 / 342.3    91.7 / 276.5
Inception      134.5 / 485.8   125.7 / 450.6
VGG (batch 64) 200.8 / 650.0   164.1 / 551.7

Speed on Inception with varied batch sizes and CuDNN v4 is as follows:

Batch Size   Speed per batch     Speed per image
 16             22.8 /  72.7         1.43 / 4.54
 32             38.0 / 127.5         1.19 / 3.98
 64             67.2 / 233.6         1.05 / 3.65
128            125.7 / 450.6         0.98 / 3.52

Speed on Tesla M40, which 10 warmup steps and 10 main steps and with cudnn
v4, is as follows:

AlexNet         68.4 / 218.1
OverFeat       210.5 / 630.3
Inception      300.2 / 1122.2
VGG (batch 64) 405.8 / 1327.7

(Note that these numbers involve a "full" backprop, i.e. the gradient
with respect to the input image is also computed.)

To get the numbers, simply run:

for MODEL in AlexNet OverFeat Inception; do
  PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py     --batch_size 128 --model $MODEL --forward_only True
done
for MODEL in AlexNet OverFeat Inception; do
  PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py     --batch_size 128 --model $MODEL
done
PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py   --batch_size 64 --model VGGA --forward_only True
PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py   --batch_size 64 --model VGGA

for BS in 16 32 64 128; do
  PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py     --batch_size $BS --model Inception --forward_only True
  PYTHONPATH=../gen:$PYTHONPATH python convnet_benchmarks.py     --batch_size $BS --model Inception
done

Note that VGG needs to be run at batch 64 due to memory limit on the backward
pass.
"""

import argparse
from caffe2.python import workspace, brew, model_helper

def MLP(order, cudnn_ws):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.convnet_benchmarks.MLP', 'MLP(order, cudnn_ws)', {'model_helper': model_helper, 'brew': brew, 'order': order, 'cudnn_ws': cudnn_ws}, 2)

def AlexNet(order, cudnn_ws):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.convnet_benchmarks.AlexNet', 'AlexNet(order, cudnn_ws)', {'model_helper': model_helper, 'brew': brew, 'order': order, 'cudnn_ws': cudnn_ws}, 2)

def OverFeat(order, cudnn_ws):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.convnet_benchmarks.OverFeat', 'OverFeat(order, cudnn_ws)', {'model_helper': model_helper, 'brew': brew, 'order': order, 'cudnn_ws': cudnn_ws}, 2)

def VGGA(order, cudnn_ws):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.convnet_benchmarks.VGGA', 'VGGA(order, cudnn_ws)', {'model_helper': model_helper, 'brew': brew, 'order': order, 'cudnn_ws': cudnn_ws}, 2)

def _InceptionModule(model, input_blob, input_depth, output_name, conv1_depth, conv3_depths, conv5_depths, pool_depth):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.convnet_benchmarks._InceptionModule', '_InceptionModule(model, input_blob, input_depth, output_name, conv1_depth, conv3_depths, conv5_depths, pool_depth)', {'brew': brew, 'model': model, 'input_blob': input_blob, 'input_depth': input_depth, 'output_name': output_name, 'conv1_depth': conv1_depth, 'conv3_depths': conv3_depths, 'conv5_depths': conv5_depths, 'pool_depth': pool_depth}, 1)

def Inception(order, cudnn_ws):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.convnet_benchmarks.Inception', 'Inception(order, cudnn_ws)', {'model_helper': model_helper, 'brew': brew, '_InceptionModule': _InceptionModule, 'order': order, 'cudnn_ws': cudnn_ws}, 2)

def AddParameterUpdate(model):
    """ Simple plain SGD update -- not tuned to actually train the models """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.convnet_benchmarks.AddParameterUpdate', 'AddParameterUpdate(model)', {'brew': brew, 'model': model}, 0)

def Benchmark(model_gen, arg):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.convnet_benchmarks.Benchmark', 'Benchmark(model_gen, arg)', {'AddParameterUpdate': AddParameterUpdate, 'workspace': workspace, 'model_gen': model_gen, 'arg': arg}, 0)

def GetArgumentParser():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.convnet_benchmarks.GetArgumentParser', 'GetArgumentParser()', {'argparse': argparse}, 1)
if __name__ == '__main__':
    (args, extra_args) = GetArgumentParser().parse_known_args()
    if (not args.batch_size or not args.model or not args.order):
        GetArgumentParser().print_help()
    else:
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'] + extra_args + ((['--caffe2_use_nvtx'] if args.use_nvtx else [])) + ((['--caffe2_htrace_span_log_path=' + args.htrace_span_log_path] if args.htrace_span_log_path else [])))
        model_map = {'AlexNet': AlexNet, 'OverFeat': OverFeat, 'VGGA': VGGA, 'Inception': Inception, 'MLP': MLP}
        Benchmark(model_map[args.model], args)

