from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python.test_util import TestCase
from caffe2.proto import caffe2_pb2
import unittest
import numpy as np
from caffe2.python import core, workspace


@unittest.skipIf(not workspace.C.use_mkldnn, 'No MKLDNN support.')
class TestReShapeOps(TestCase):
    
    def test_reshape_ops(self):
        device_opt = core.DeviceOption(caffe2_pb2.IDEEP, 0)
        with core.DeviceScope(device_opt):
            workspace.FeedBlob('res', np.array([[0, 0, 0, 0]], dtype=np.float32))
            workspace.FeedBlob('shape', np.array([1, 4], dtype=np.int32), core.DeviceOption(caffe2_pb2.CPU, 0))
            workspace.FeedBlob('input', np.zeros((2, 2), dtype=np.float32))
            workspace.RunOperatorOnce(core.CreateOperator('Reshape', ['input', 'shape'], ['output', 'old_shape']))
            assert (workspace.FetchBlob('output') == workspace.FetchBlob('res')).all()
    
    def test_basic_reshape(self):
        _test_reshape(old_shape=(4, 2, 1), new_shape=(2, 4))
        _test_reshape(old_shape=(4, 2, 1), new_shape=(2, 4), arg_shape=False)
    
    def test_int64_reshape_input(self):
        _test_reshape(old_shape=(4, 2, 1), new_shape=(2, 4), arg_shape=False, shape_dtype=np.int64)
    
    def test_missing_dim(self):
        _test_reshape(old_shape=(4, 2, 1), new_shape=(-1, 8))
        _test_reshape(old_shape=(4, 2, 1), new_shape=(-1, 8), arg_shape=False)
    
    def test_in_place(self):
        _test_reshape(old_shape=(4, 2, 1), new_shape=(-1, 8), in_place=True)
        _test_reshape(old_shape=(4, 2, 1), new_shape=(-1, 8), in_place=True, arg_shape=False)
    
    def test_zero_dim(self):
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, 0, 0), expected_shape=(4, 2, 1))
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, 0, 0), expected_shape=(4, 2, 1), arg_shape=False)
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, 2, 1), expected_shape=(4, 2, 1))
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, 2, 1), expected_shape=(4, 2, 1), arg_shape=False)
    
    def test_zero_dim_and_missing_dim(self):
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, -1, 0), expected_shape=(4, 2, 1))
        _test_reshape(old_shape=(4, 2, 1), new_shape=(0, -1, 0), expected_shape=(4, 2, 1), arg_shape=False)
        _test_reshape(old_shape=(4, 3, 2), new_shape=(-1, 0), expected_shape=(8, 3))
        _test_reshape(old_shape=(4, 3, 2), new_shape=(-1, 0), expected_shape=(8, 3), arg_shape=False)
    
    def test_backprop(self):
        device_opt = core.DeviceOption(caffe2_pb2.IDEEP, 0)
        with core.DeviceScope(device_opt):
            old_shape = (4, 2, 1)
            new_shape = (1, 8)
            X = np.random.rand(*old_shape).astype(np.float32)
            Y = np.random.rand(*new_shape).astype(np.float32)
            net = core.Net('net')
            net.GivenTensorFill([], 'X', shape=old_shape, values=X.flatten())
            net.GivenTensorFill([], 'Y', shape=new_shape, values=Y.flatten())
            net.Reshape(['X'], ['X_out', 'old_shape'], shape=new_shape)
            net.Mul(['X_out', 'Y'], 'Z')
            net.AddGradientOperators(['Z'])
            workspace.RunNetOnce(net)
            Z = workspace.FetchBlob('Z')
            X_grad = workspace.FetchBlob('X_grad')
            np.testing.assert_allclose(Z.squeeze(), (X.reshape(new_shape) * Y).squeeze(), rtol=1e-05)
            np.testing.assert_array_equal(X_grad.shape, X.shape)
            np.testing.assert_allclose(X_grad, Y.reshape(old_shape), rtol=1e-05)
    
    def test_input_shape_changes(self):
        device_opt = core.DeviceOption(caffe2_pb2.IDEEP, 0)
        with core.DeviceScope(device_opt):
            workspace.FeedBlob('input_blob', np.array(np.random.rand(10, 20, 10), dtype=np.float32))
            net = core.Net('mynet')
            (z, _) = net.Reshape('input_blob', ['z_reshape', 'dummy_size'], shape=(-1, 10))
            workspace.CreateNet(net)
            workspace.RunNet(net)
            workspace.FeedBlob('input_blob', np.array(np.random.rand(10, 40, 10), dtype=np.float32))
            workspace.RunNet(net)


def _test_reshape(old_shape, new_shape, expected_shape=None, arg_shape=True, in_place=False, shape_dtype=np.int32):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.ideep.reshape_op_test._test_reshape', '_test_reshape(old_shape, new_shape, expected_shape=None, arg_shape=True, in_place=False, shape_dtype=np.int32)', {'core': core, 'caffe2_pb2': caffe2_pb2, 'np': np, 'workspace': workspace, 'old_shape': old_shape, 'new_shape': new_shape, 'expected_shape': expected_shape, 'arg_shape': arg_shape, 'in_place': in_place, 'shape_dtype': shape_dtype}, 0)
if __name__ == '__main__':
    unittest.main()

