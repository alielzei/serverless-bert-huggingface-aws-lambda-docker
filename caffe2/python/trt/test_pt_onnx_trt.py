import os
import unittest
from typing import List, Any
from PIL import Image
import numpy as np
import torch
from torch.onnx import OperatorExportTypes
import torchvision.models as models
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def allocate_buffers(engine):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.trt.test_pt_onnx_trt.allocate_buffers', 'allocate_buffers(engine)', {'cuda': cuda, 'trt': trt, 'engine': engine}, 5)

def load_normalized_test_case(input_shape, test_image, pagelocked_buffer, normalization_hint):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.trt.test_pt_onnx_trt.load_normalized_test_case', 'load_normalized_test_case(input_shape, test_image, pagelocked_buffer, normalization_hint)', {'np': np, 'Image': Image, 'trt': trt, 'input_shape': input_shape, 'test_image': test_image, 'pagelocked_buffer': pagelocked_buffer, 'normalization_hint': normalization_hint}, 1)


class Test_PT_ONNX_TRT(unittest.TestCase):
    
    def __enter__(self):
        return self
    
    def setUp(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data')
        self.image_files = ['binoculars.jpeg', 'reflex_camera.jpeg', 'tabby_tiger_cat.jpg']
        for (index, f) in enumerate(self.image_files):
            self.image_files[index] = os.path.abspath(os.path.join(data_path, f))
            if not os.path.exists(self.image_files[index]):
                raise FileNotFoundError(self.image_files[index] + ' does not exist.')
        self.labels = open(os.path.abspath(os.path.join(data_path, 'class_labels.txt')), 'r').read().split('\n')
    
    def build_engine_onnx(self, model_file):
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=1) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
            builder.max_workspace_size = 1 << 33
            with open(model_file, 'rb') as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        self.fail('ERROR: {}'.format(parser.get_error(error)))
            return builder.build_cuda_engine(network)
    
    def _test_model(self, model_name, input_shape=(3, 224, 224), normalization_hint=0):
        model = getattr(models, model_name)(pretrained=True)
        shape = (1, ) + input_shape
        dummy_input = (torch.randn(shape), )
        onnx_name = model_name + '.onnx'
        torch.onnx.export(model, dummy_input, onnx_name, input_names=[], output_names=[], verbose=False, export_params=True, opset_version=9)
        with self.build_engine_onnx(onnx_name) as engine:
            (h_input, d_input, h_output, d_output, stream) = allocate_buffers(engine)
            with engine.create_execution_context() as context:
                err_count = 0
                for (index, f) in enumerate(self.image_files):
                    test_case = load_normalized_test_case(input_shape, f, h_input, normalization_hint)
                    cuda.memcpy_htod_async(d_input, h_input, stream)
                    context.execute_async_v2(bindings=[d_input, d_output], stream_handle=stream.handle)
                    cuda.memcpy_dtoh_async(h_output, d_output, stream)
                    stream.synchronize()
                    amax = np.argmax(h_output)
                    pred = self.labels[amax]
                    if '_'.join(pred.split()) not in os.path.splitext(os.path.basename(test_case))[0]:
                        err_count = err_count + 1
                self.assertLessEqual(err_count, 1, 'Too many recognition errors')
    
    def test_alexnet(self):
        self._test_model('alexnet', (3, 227, 227))
    
    def test_resnet18(self):
        self._test_model('resnet18')
    
    def test_resnet34(self):
        self._test_model('resnet34')
    
    def test_resnet50(self):
        self._test_model('resnet50')
    
    def test_resnet101(self):
        self._test_model('resnet101')
    
    @unittest.skip('Takes 2m')
    def test_resnet152(self):
        self._test_model('resnet152')
    
    def test_resnet50_2(self):
        self._test_model('wide_resnet50_2')
    
    @unittest.skip('Takes 2m')
    def test_resnet101_2(self):
        self._test_model('wide_resnet101_2')
    
    def test_squeezenet1_0(self):
        self._test_model('squeezenet1_0')
    
    def test_squeezenet1_1(self):
        self._test_model('squeezenet1_1')
    
    def test_googlenet(self):
        self._test_model('googlenet')
    
    def test_inception_v3(self):
        self._test_model('inception_v3')
    
    def test_mnasnet0_5(self):
        self._test_model('mnasnet0_5', normalization_hint=1)
    
    def test_mnasnet1_0(self):
        self._test_model('mnasnet1_0', normalization_hint=1)
    
    def test_mobilenet_v2(self):
        self._test_model('mobilenet_v2', normalization_hint=1)
    
    def test_shufflenet_v2_x0_5(self):
        self._test_model('shufflenet_v2_x0_5')
    
    def test_shufflenet_v2_x1_0(self):
        self._test_model('shufflenet_v2_x1_0')
    
    def test_vgg11(self):
        self._test_model('vgg11')
    
    def test_vgg11_bn(self):
        self._test_model('vgg11_bn')
    
    def test_vgg13(self):
        self._test_model('vgg13')
    
    def test_vgg13_bn(self):
        self._test_model('vgg13_bn')
    
    def test_vgg16(self):
        self._test_model('vgg16')
    
    def test_vgg16_bn(self):
        self._test_model('vgg16_bn')
    
    def test_vgg19(self):
        self._test_model('vgg19')
    
    def test_vgg19_bn(self):
        self._test_model('vgg19_bn')
    
    @unittest.skip('Takes 13m')
    def test_densenet121(self):
        self._test_model('densenet121')
    
    @unittest.skip('Takes 25m')
    def test_densenet161(self):
        self._test_model('densenet161')
    
    @unittest.skip('Takes 27m')
    def test_densenet169(self):
        self._test_model('densenet169')
    
    @unittest.skip('Takes 44m')
    def test_densenet201(self):
        self._test_model('densenet201')

if __name__ == '__main__':
    unittest.main()

