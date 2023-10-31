from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest
import numpy as np
import copy
from hypothesis import given
import hypothesis.strategies as st
from caffe2.python.model_helper import ModelHelper
from caffe2.python.models import resnet
from caffe2.python import workspace, brew
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl.rewrite_graph as rewrite_graph

def deterministic_io(model):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph_test.deterministic_io', 'deterministic_io(model)', {'copy': copy, 'model': model}, 1)

def simple_fc():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph_test.simple_fc', 'simple_fc()', {'ModelHelper': ModelHelper, 'brew': brew}, 2)

def double_matmul():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph_test.double_matmul', 'double_matmul()', {'ModelHelper': ModelHelper, 'brew': brew}, 2)

def simple_relu():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph_test.simple_relu', 'simple_relu()', {'ModelHelper': ModelHelper, 'brew': brew}, 2)

def simple_mlp():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph_test.simple_mlp', 'simple_mlp()', {'ModelHelper': ModelHelper, 'brew': brew}, 2)

def simple_cnn():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph_test.simple_cnn', 'simple_cnn()', {'ModelHelper': ModelHelper, 'brew': brew}, 2)

def alexnet():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph_test.alexnet', 'alexnet()', {'ModelHelper': ModelHelper, 'brew': brew}, 2)

def simple_resnet():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph_test.simple_resnet', 'simple_resnet()', {'ModelHelper': ModelHelper, 'resnet': resnet}, 2)

def complex_resnet():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mkl.rewrite_graph_test.complex_resnet', 'complex_resnet()', {'ModelHelper': ModelHelper, 'resnet': resnet}, 2)


@unittest.skipIf(not workspace.C.use_mkldnn, 'No MKLDNN support.')
class MKLRewriteTest(hu.HypothesisTestCase):
    
    @given(gen=st.sampled_from([simple_relu, simple_fc, simple_mlp, simple_cnn]))
    def test_mkl_simple_rewrite(self, gen):
        (cpu_model, (shape, )) = gen()
        cpu_model = deterministic_io(cpu_model)
        mkl_model = rewrite_graph.rewrite_model_helper_simple(cpu_model)
        X = np.random.randn(*shape).astype(np.float32)
        
        def run(model):
            self.ws.run(model.InitProto())
            self.ws.create_blob(model.Proto().external_input[0]).feed(X)
            self.ws.run(model.Proto())
            return self.ws.blobs[model.Proto().external_output[0]].fetch()
        np.testing.assert_allclose(run(cpu_model), run(mkl_model), atol=0.0001, rtol=0.0001)
    
    def test_mkl_resnet_rewrite(self):
        (cpu_model, (shape, )) = complex_resnet()
        cpu_model = deterministic_io(cpu_model)
        mkl_model = rewrite_graph.rewrite_model_helper_simple(cpu_model)
        np.random.seed(1701)
        X = np.random.randn(*shape).astype(np.float32)
        
        def run(model):
            self.ws.run(model.InitProto())
            self.ws.create_blob(model.Proto().external_input[0]).feed(X)
            self.ws.run(model.Proto())
            return self.ws.blobs[model.Proto().external_output[0]].fetch()
        np.testing.assert_allclose(run(cpu_model), run(mkl_model), atol=0.0001, rtol=0.0001)
    
    def test_mkl_multi_output_rewrite(self):
        (cpu_model, shapes) = double_matmul()
        cpu_model = deterministic_io(cpu_model)
        mkl_model = rewrite_graph.rewrite_model_helper_simple(cpu_model)
        np.random.seed(1701)
        Xs = [np.random.randn(*shape).astype(np.float32) for shape in shapes]
        
        def run(model):
            self.ws.run(model.InitProto())
            for (name, X) in zip(model.Proto().external_input, Xs):
                self.ws.create_blob(name).feed(X)
            print(model.Proto())
            self.ws.run(model.Proto())
            return [self.ws.blobs[name].fetch() for name in model.Proto().external_output]
        run(mkl_model)
        np.testing.assert_allclose(run(cpu_model), run(mkl_model), atol=0.0001, rtol=0.0001)
    
    def test_mkl_alexnet_rewrite(self):
        (cpu_model, (shape, )) = alexnet()
        cpu_model = deterministic_io(cpu_model)
        mkl_model = rewrite_graph.rewrite_model_helper_simple(cpu_model)
        np.random.seed(1701)
        X = np.random.randn(*shape).astype(np.float32)
        
        def run(model):
            self.ws.run(model.InitProto())
            self.ws.create_blob(model.Proto().external_input[0]).feed(X)
            self.ws.run(model.Proto())
            return self.ws.blobs[model.Proto().external_output[0]].fetch()
        np.testing.assert_allclose(run(cpu_model), run(mkl_model), atol=0.0001, rtol=0.0001)

if __name__ == '__main__':
    import unittest
    unittest.main()

