from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import unittest
from caffe2.python import brew, model_helper, workspace
from caffe2.python.modeling.initializers import Initializer, PseudoFP16Initializer


class InitializerTest(unittest.TestCase):
    
    def test_fc_initializer(self):
        model = model_helper.ModelHelper(name='test')
        data = model.net.AddExternalInput('data')
        fc1 = brew.fc(model, data, 'fc1', dim_in=1, dim_out=1)
        fc2 = brew.fc(model, fc1, 'fc2', dim_in=1, dim_out=1, WeightInitializer=Initializer)
        fc3 = brew.fc(model, fc2, 'fc3', dim_in=1, dim_out=1, WeightInitializer=Initializer, weight_init=('ConstantFill', {}))
        fc4 = brew.fc(model, fc3, 'fc4', dim_in=1, dim_out=1, WeightInitializer=None, weight_init=('ConstantFill', {}))
    
    @unittest.skipIf(not workspace.has_gpu_support, 'No GPU support')
    def test_fc_fp16_initializer(self):
        model = model_helper.ModelHelper(name='test')
        data = model.net.AddExternalInput('data')
        fc1 = brew.fc(model, data, 'fc1', dim_in=1, dim_out=1)
        fc2 = brew.fc(model, fc1, 'fc2', dim_in=1, dim_out=1, WeightInitializer=PseudoFP16Initializer)
        fc3 = brew.fc(model, fc2, 'fc3', dim_in=1, dim_out=1, weight_init=('ConstantFill', {}), WeightInitializer=PseudoFP16Initializer)
    
    def test_fc_external_initializer(self):
        model = model_helper.ModelHelper(name='test', init_params=False)
        data = model.net.AddExternalInput('data')
        fc1 = brew.fc(model, data, 'fc1', dim_in=1, dim_out=1)
        self.assertEqual(len(model.net.Proto().op), 1)
        self.assertEqual(len(model.param_init_net.Proto().op), 0)


