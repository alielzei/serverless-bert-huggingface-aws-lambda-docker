from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
from caffe2.proto import caffe2_pb2
import click
import numpy as np
from onnx import checker, ModelProto
from caffe2.python.onnx.backend import Caffe2Backend as c2
import caffe2.python.onnx.frontend as c2_onnx

@click.command(help='convert caffe2 net to onnx model', context_settings={'help_option_names': ['-h', '--help']})
@click.argument('caffe2_net', type=click.File('rb'))
@click.option('--caffe2-net-name', type=str, help='Name of the caffe2 net')
@click.option('--caffe2-init-net', type=click.File('rb'), help='Path of the caffe2 init net pb file')
@click.option('--value-info', type=str, help='A json string providing the type and shape information of the inputs')
@click.option('-o', '--output', required=True, type=click.File('wb'), help='Output path for the onnx model pb file')
def caffe2_to_onnx(caffe2_net, caffe2_net_name, caffe2_init_net, value_info, output):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.onnx.bin.conversion.caffe2_to_onnx', 'caffe2_to_onnx(caffe2_net, caffe2_net_name, caffe2_init_net, value_info, output)', {'caffe2_pb2': caffe2_pb2, 'json': json, 'c2_onnx': c2_onnx, 'click': click, 'str': str, 'caffe2_net': caffe2_net, 'caffe2_net_name': caffe2_net_name, 'caffe2_init_net': caffe2_init_net, 'value_info': value_info, 'output': output}, 0)

@click.command(help='convert onnx model to caffe2 net', context_settings={'help_option_names': ['-h', '--help']})
@click.argument('onnx_model', type=click.File('rb'))
@click.option('-o', '--output', required=True, type=click.File('wb'), help='Output path for the caffe2 net file')
@click.option('--init-net-output', required=True, type=click.File('wb'), help='Output path for the caffe2 init net file')
def onnx_to_caffe2(onnx_model, output, init_net_output):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.onnx.bin.conversion.onnx_to_caffe2', 'onnx_to_caffe2(onnx_model, output, init_net_output)', {'ModelProto': ModelProto, 'c2': c2, 'click': click, 'onnx_model': onnx_model, 'output': output, 'init_net_output': init_net_output}, 0)

