from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import onnx
import onnx.defs
from onnx.helper import make_node, make_graph, make_tensor, make_tensor_value_info, make_model
from onnx.backend.base import namedtupledict
from caffe2.python.models.download import ModelDownloader
import caffe2.python.onnx.backend as c2
from caffe2.python.onnx.workspace import Workspace
from caffe2.python.trt.transform import convert_onnx_model_to_trt_op, transform_caffe2_net
from caffe2.python.onnx.tests.test_utils import TestCase
import numpy as np
import os.path
import json
import time
import unittest
import tarfile
import tempfile
import shutil
from six.moves.urllib.request import urlretrieve

def _print_net(net):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.trt.test_trt._print_net', '_print_net(net)', {'net': net}, 0)

def _base_url(opset_version):
    return 'https://s3.amazonaws.com/download.onnx/models/opset_{}'.format(opset_version)

def _download_onnx_model(model_name, opset_version):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.trt.test_trt._download_onnx_model', '_download_onnx_model(model_name, opset_version)', {'os': os, 'shutil': shutil, '_base_url': _base_url, 'tempfile': tempfile, 'urlretrieve': urlretrieve, 'tarfile': tarfile, 'model_name': model_name, 'opset_version': opset_version}, 1)


class TensorRTOpTest(TestCase):
    
    def setUp(self):
        self.opset_version = onnx.defs.onnx_opset_version()
    
    def _test_relu_graph(self, X, batch_size, trt_max_batch_size):
        node_def = make_node('Relu', ['X'], ['Y'])
        Y_c2 = c2.run_node(node_def, {'X': X})
        graph_def = make_graph([node_def], name='test', inputs=[make_tensor_value_info('X', onnx.TensorProto.FLOAT, [batch_size, 1, 3, 2])], outputs=[make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [batch_size, 1, 3, 2])])
        model_def = make_model(graph_def, producer_name='relu-test')
        op_outputs = [x.name for x in model_def.graph.output]
        op = convert_onnx_model_to_trt_op(model_def, max_batch_size=trt_max_batch_size)
        device_option = core.DeviceOption(caffe2_pb2.CUDA, 0)
        op.device_option.CopyFrom(device_option)
        Y_trt = None
        ws = Workspace()
        with core.DeviceScope(device_option):
            ws.FeedBlob('X', X)
            ws.RunOperatorsOnce([op])
            output_values = [ws.FetchBlob(name) for name in op_outputs]
            Y_trt = namedtupledict('Outputs', op_outputs)(*output_values)
        np.testing.assert_almost_equal(Y_c2, Y_trt)
    
    @unittest.skipIf(not workspace.C.use_trt, 'No TensortRT support')
    def test_relu_graph_simple(self):
        X = np.random.randn(1, 1, 3, 2).astype(np.float32)
        self._test_relu_graph(X, 1, 50)
    
    @unittest.skipIf(not workspace.C.use_trt, 'No TensortRT support')
    def test_relu_graph_big_batch(self):
        X = np.random.randn(52, 1, 3, 2).astype(np.float32)
        self._test_relu_graph(X, 52, 50)
    
    def _test_onnx_importer(self, model_name, data_input_index, opset_version=onnx.defs.onnx_opset_version()):
        model_dir = _download_onnx_model(model_name, opset_version)
        model_def = onnx.load(os.path.join(model_dir, 'model.onnx'))
        input_blob_dims = [int(x.dim_value) for x in model_def.graph.input[data_input_index].type.tensor_type.shape.dim]
        op_inputs = [x.name for x in model_def.graph.input]
        op_outputs = [x.name for x in model_def.graph.output]
        print('{}'.format(op_inputs))
        data = np.random.randn(*input_blob_dims).astype(np.float32)
        Y_c2 = c2.run_model(model_def, {op_inputs[data_input_index]: data})
        op = convert_onnx_model_to_trt_op(model_def, verbosity=3)
        device_option = core.DeviceOption(caffe2_pb2.CUDA, 0)
        op.device_option.CopyFrom(device_option)
        Y_trt = None
        ws = Workspace()
        with core.DeviceScope(device_option):
            ws.FeedBlob(op_inputs[data_input_index], data)
            if opset_version >= 5:
                ws.FeedBlob('data_0', data)
            ws.RunOperatorsOnce([op])
            output_values = [ws.FetchBlob(name) for name in op_outputs]
            Y_trt = namedtupledict('Outputs', op_outputs)(*output_values)
        np.testing.assert_allclose(Y_c2, Y_trt, rtol=0.001)
    
    @unittest.skipIf(not workspace.C.use_trt, 'No TensortRT support')
    def test_resnet50(self):
        self._test_onnx_importer('resnet50', 0, 9)
    
    @unittest.skipIf(not workspace.C.use_trt, 'No TensortRT support')
    def test_bvlc_alexnet(self):
        self._test_onnx_importer('bvlc_alexnet', 0, 9)
    
    @unittest.skip('Until fixing Unsqueeze op')
    def test_densenet121(self):
        self._test_onnx_importer('densenet121', -1, 3)
    
    @unittest.skipIf(not workspace.C.use_trt, 'No TensortRT support')
    def test_inception_v1(self):
        self._test_onnx_importer('inception_v1', -3, 9)
    
    @unittest.skip('Until fixing Unsqueeze op')
    def test_inception_v2(self):
        self._test_onnx_importer('inception_v2', 0, 9)
    
    @unittest.skip('Need to revisit our ChannelShuffle exporter to avoid generating 5D tensor')
    def test_shufflenet(self):
        self._test_onnx_importer('shufflenet', 0)
    
    @unittest.skipIf(not workspace.C.use_trt, 'No TensortRT support')
    def test_squeezenet(self):
        self._test_onnx_importer('squeezenet', -1, 9)
    
    @unittest.skipIf(not workspace.C.use_trt, 'No TensortRT support')
    def test_vgg16(self):
        self._test_onnx_importer('vgg16', 0, 9)
    
    @unittest.skipIf(not workspace.C.use_trt, 'No TensortRT support')
    def test_vgg19(self):
        self._test_onnx_importer('vgg19', -2, 9)



class TensorRTTransformTest(TestCase):
    
    def setUp(self):
        self.model_downloader = ModelDownloader()
    
    def _add_head_tail(self, pred_net, new_head, new_tail):
        orig_head = pred_net.external_input[0]
        orig_tail = pred_net.external_output[0]
        head = caffe2_pb2.OperatorDef()
        head.type = 'Copy'
        head.input.append(new_head)
        head.output.append(orig_head)
        dummy = caffe2_pb2.NetDef()
        dummy.op.extend(pred_net.op)
        del pred_net.op[:]
        pred_net.op.extend([head])
        pred_net.op.extend(dummy.op)
        pred_net.external_input[0] = new_head
        tail = caffe2_pb2.OperatorDef()
        tail.type = 'Copy'
        tail.input.append(orig_tail)
        tail.output.append(new_tail)
        pred_net.op.extend([tail])
        pred_net.external_output[0] = new_tail
    
    @unittest.skipIf(not workspace.C.use_trt, 'No TensortRT support')
    def test_resnet50_core(self):
        N = 2
        warmup = 20
        repeat = 100
        print('Batch size: {}, repeat inference {} times, warmup {} times'.format(N, repeat, warmup))
        (init_net, pred_net, _) = self.model_downloader.get_c2_model('resnet50')
        self._add_head_tail(pred_net, 'real_data', 'real_softmax')
        input_blob_dims = (N, 3, 224, 224)
        input_name = 'real_data'
        device_option = core.DeviceOption(caffe2_pb2.CUDA, 0)
        init_net.device_option.CopyFrom(device_option)
        pred_net.device_option.CopyFrom(device_option)
        for op in pred_net.op:
            op.device_option.CopyFrom(device_option)
            op.engine = 'CUDNN'
        net_outputs = pred_net.external_output
        Y_c2 = None
        data = np.random.randn(*input_blob_dims).astype(np.float32)
        c2_time = 1
        workspace.SwitchWorkspace('gpu_test', True)
        with core.DeviceScope(device_option):
            workspace.FeedBlob(input_name, data)
            workspace.RunNetOnce(init_net)
            workspace.CreateNet(pred_net)
            for _ in range(warmup):
                workspace.RunNet(pred_net.name)
            start = time.time()
            for _ in range(repeat):
                workspace.RunNet(pred_net.name)
            end = time.time()
            c2_time = end - start
            output_values = [workspace.FetchBlob(name) for name in net_outputs]
            Y_c2 = namedtupledict('Outputs', net_outputs)(*output_values)
        workspace.ResetWorkspace()
        with core.DeviceScope(device_option):
            workspace.RunNetOnce(init_net)
        start = time.time()
        pred_net_cut = transform_caffe2_net(pred_net, {input_name: input_blob_dims}, build_serializable_op=False)
        del init_net, pred_net
        pred_net_cut.device_option.CopyFrom(device_option)
        for op in pred_net_cut.op:
            op.device_option.CopyFrom(device_option)
        Y_trt = None
        input_name = pred_net_cut.external_input[0]
        print('C2 runtime: {}s'.format(c2_time))
        with core.DeviceScope(device_option):
            workspace.FeedBlob(input_name, data)
            workspace.CreateNet(pred_net_cut)
            end = time.time()
            print('Conversion time: {:.2f}s'.format(end - start))
            for _ in range(warmup):
                workspace.RunNet(pred_net_cut.name)
            start = time.time()
            for _ in range(repeat):
                workspace.RunNet(pred_net_cut.name)
            end = time.time()
            trt_time = end - start
            print('TRT runtime: {}s, improvement: {}%'.format(trt_time, (c2_time - trt_time) / c2_time * 100))
            output_values = [workspace.FetchBlob(name) for name in net_outputs]
            Y_trt = namedtupledict('Outputs', net_outputs)(*output_values)
        np.testing.assert_allclose(Y_c2, Y_trt, rtol=0.001)


