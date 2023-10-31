from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import brew, core, workspace
from caffe2.python.model_helper import ModelHelper
from functools import partial
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np
import os
import torch
import unittest

def _layer_norm_ref(axis, epsilon, X):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.layer_norm_op_test._layer_norm_ref', '_layer_norm_ref(axis, epsilon, X)', {'np': np, 'axis': axis, 'epsilon': epsilon, 'X': X}, 3)

def _layer_norm_with_affine_ref(axis, epsilon, X, gamma, beta):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.layer_norm_op_test._layer_norm_with_affine_ref', '_layer_norm_with_affine_ref(axis, epsilon, X, gamma, beta)', {'_layer_norm_ref': _layer_norm_ref, 'axis': axis, 'epsilon': epsilon, 'X': X, 'gamma': gamma, 'beta': beta}, 3)

def _layer_norm_grad_ref(axis, gout_full, norm, mean_full, stdev_full, X_full):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.layer_norm_op_test._layer_norm_grad_ref', '_layer_norm_grad_ref(axis, gout_full, norm, mean_full, stdev_full, X_full)', {'np': np, 'axis': axis, 'gout_full': gout_full, 'norm': norm, 'mean_full': mean_full, 'stdev_full': stdev_full, 'X_full': X_full}, 1)


class TestLayerNormOp(serial.SerializedTestCase):
    
    @serial.given(X=hu.tensor(min_dim=2), **hu.gcs)
    def test_layer_norm_grad_op(self, X, gc, dc):
        axis = np.random.randint(0, len(X.shape))
        epsilon = 0.0001
        op = core.CreateOperator('LayerNormGradient', ['gout', 'out', 'mean', 'stdev', 'in'], ['gin'], axis=axis, epsilon=epsilon)
        (norm, mean, stdev) = _layer_norm_ref(axis, epsilon, X)
        gout = norm
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[gout, norm, mean, stdev, X], reference=partial(_layer_norm_grad_ref, axis))
        self.assertDeviceChecks(device_options=dc, op=op, inputs=[gout, norm, mean, stdev, X], outputs_to_check=[0])
    
    @given(X=hu.tensor(min_dim=2), eps=st.floats(1e-05, 0.001), elementwise_affine=st.booleans(), **hu.gcs)
    def test_layer_norm_op(self, X, eps, elementwise_affine, gc, dc):
        axis = np.random.randint(0, len(X.shape))
        op = core.CreateOperator('LayerNorm', (['X', 'gamma', 'beta'] if elementwise_affine else ['X']), ['Y', 'mean', 'std'], axis=axis, epsilon=eps, elementwise_affine=elementwise_affine)
        if elementwise_affine:
            ref = partial(_layer_norm_with_affine_ref, axis, eps)
        else:
            ref = partial(_layer_norm_ref, axis, eps)
        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            inputs = [X, gamma, beta]
        else:
            inputs = [X]
        self.assertReferenceChecks(device_option=gc, op=op, inputs=inputs, reference=ref)
        self.assertDeviceChecks(device_options=dc, op=op, inputs=inputs, outputs_to_check=[0, 1, 2])
    
    @given(M=st.integers(1, 10), N=st.integers(10, 20), axis=st.integers(0, 1), eps=st.floats(1e-05, 0.001), elementwise_affine=st.booleans(), **hu.gcs)
    def test_layer_norm_grad(self, M, N, axis, eps, elementwise_affine, gc, dc):
        op = core.CreateOperator('LayerNorm', (['X', 'gamma', 'beta'] if elementwise_affine else ['X']), ['Y', 'mean', 'std'], axis=axis, epsilon=eps, elementwise_affine=elementwise_affine)
        X = np.arange(M * N).astype(np.float32)
        np.random.shuffle(X)
        X = X.reshape((M, N))
        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            inputs = [X, gamma, beta]
        else:
            inputs = [X]
        for i in range(len(inputs)):
            self.assertGradientChecks(gc, op, inputs, i, [0])
    
    @unittest.skipIf(workspace.has_hip_support, "Operator cross-calling doesn't work with hip yet")
    @given(X=hu.tensor(min_dim=2), eps=st.floats(1e-05, 0.001), elementwise_affine=st.booleans(), **hu.gcs)
    def test_layer_norm_op_c10(self, X, eps, elementwise_affine, gc, dc):
        axis = np.random.randint(0, len(X.shape))
        op = core.CreateOperator('C10LayerNorm_DontUseThisOpYet', (['X', 'gamma', 'beta'] if elementwise_affine else ['X']), ['Y', 'mean', 'std'], axis=axis, epsilon=eps, elementwise_affine=elementwise_affine)
        if elementwise_affine:
            ref = partial(_layer_norm_with_affine_ref, axis, eps)
        else:
            ref = partial(_layer_norm_ref, axis, eps)
        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            inputs = [X, gamma, beta]
        else:
            inputs = [X]
        self.assertReferenceChecks(device_option=gc, op=op, inputs=inputs, reference=ref)
        self.assertDeviceChecks(device_options=dc, op=op, inputs=inputs, outputs_to_check=[0, 1, 2])
    
    @unittest.skipIf(workspace.has_hip_support, "Operator cross-calling doesn't work with hip yet")
    @given(X=hu.tensor(min_dim=2), eps=st.floats(1e-05, 0.001), elementwise_affine=st.booleans(), **hu.gcs)
    def test_layer_norm_op_c10_preallocated_outputs(self, X, eps, elementwise_affine, gc, dc):
        axis = np.random.randint(0, len(X.shape))
        self.ws.create_blob('X').feed(X)
        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            self.ws.create_blob('gamma').feed(gamma)
            self.ws.create_blob('beta').feed(beta)
        m = ModelHelper(name='test')
        m.net.C10LayerNorm_DontUseThisOpYet((['X', 'gamma', 'beta'] if elementwise_affine else ['X']), ['Y', 'mean', 'std'], axis=axis, epsilon=eps, elementwise_affine=elementwise_affine)
        self.ws.create_net(m.param_init_net).run()
        net = self.ws.create_net(m.net)
        net.run()
        net.run()
        if elementwise_affine:
            (expected_norm, expected_mean, expected_std) = _layer_norm_with_affine_ref(axis, eps, X, gamma, beta)
        else:
            (expected_norm, expected_mean, expected_std) = _layer_norm_ref(axis, eps, X)
        actual_norm = self.ws.fetch_blob('Y')
        actual_mean = self.ws.fetch_blob('mean')
        actual_std = self.ws.fetch_blob('std')
        torch.testing.assert_allclose(expected_norm, actual_norm, rtol=0.0001, atol=0.0001)
        torch.testing.assert_allclose(expected_mean, actual_mean)
        torch.testing.assert_allclose(expected_std, actual_std)
    
    @given(X=hu.tensor(min_dim=2), eps=st.floats(1e-05, 0.001), elementwise_affine=st.booleans(), **hu.gcs)
    def test_layer_norm_op_pytorch(self, X, eps, elementwise_affine, gc, dc):
        axis = np.random.randint(0, len(X.shape))
        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            (expected_norm, expected_mean, expected_std) = _layer_norm_with_affine_ref(axis, eps, X, gamma, beta)
            (actual_norm, actual_mean, actual_std) = torch.ops._caffe2.LayerNorm(torch.tensor(X), torch.tensor(gamma), torch.tensor(beta), axis, eps, True)
        else:
            (expected_norm, expected_mean, expected_std) = _layer_norm_ref(axis, eps, X)
            (actual_norm, actual_mean, actual_std) = torch.ops._caffe2.LayerNorm(torch.tensor(X), None, None, axis, eps)
        torch.testing.assert_allclose(expected_norm, actual_norm, rtol=0.0001, atol=0.0001)
        torch.testing.assert_allclose(expected_mean, actual_mean)
        torch.testing.assert_allclose(expected_std, actual_std)
    
    @unittest.skipIf(not workspace.has_cuda_support, 'No cuda support')
    @given(X=hu.tensor(min_dim=2), eps=st.floats(1e-05, 0.001), elementwise_affine=st.booleans())
    def test_layer_norm_op_pytorch_cuda(self, X, eps, elementwise_affine):
        axis = np.random.randint(0, len(X.shape))
        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            (expected_norm, expected_mean, expected_std) = _layer_norm_with_affine_ref(axis, eps, X, gamma, beta)
            (actual_norm, actual_mean, actual_std) = torch.ops._caffe2.LayerNorm(torch.tensor(X).cuda(), torch.tensor(gamma).cuda(), torch.tensor(beta).cuda(), axis, eps, True)
        else:
            (expected_norm, expected_mean, expected_std) = _layer_norm_ref(axis, eps, X)
            (actual_norm, actual_mean, actual_std) = torch.ops._caffe2.LayerNorm(torch.tensor(X).cuda(), None, None, axis, eps)
        torch.testing.assert_allclose(expected_norm, actual_norm.cpu(), rtol=0.0001, atol=0.0001)
        torch.testing.assert_allclose(expected_mean, actual_mean.cpu())
        torch.testing.assert_allclose(expected_std, actual_std.cpu())
    
    @given(X=hu.tensor(min_dim=2), eps=st.floats(1e-05, 0.001), elementwise_affine=st.booleans(), **hu.gcs)
    def test_layer_norm_op_jit(self, X, eps, elementwise_affine, gc, dc):
        
        @torch.jit.script
        def jit_layer_norm(X, gamma=None, beta=None, axis=1, eps=1e-05, elementwise_affine=False):
            return torch.ops._caffe2.LayerNorm(X, gamma, beta, axis, eps, elementwise_affine)
        axis = np.random.randint(0, len(X.shape))
        if elementwise_affine:
            gamma = np.random.randn(*X.shape[axis:]).astype(np.float32)
            beta = np.random.randn(*X.shape[axis:]).astype(np.float32)
            (expected_norm, expected_mean, expected_std) = _layer_norm_with_affine_ref(axis, eps, X, gamma, beta)
            (actual_norm, actual_mean, actual_std) = jit_layer_norm(torch.Tensor(X), torch.tensor(gamma), torch.tensor(beta), axis, eps, elementwise_affine)
        else:
            (expected_norm, expected_mean, expected_std) = _layer_norm_ref(axis, eps, X)
            (actual_norm, actual_mean, actual_std) = jit_layer_norm(torch.tensor(X), None, None, axis, eps, elementwise_affine)
        torch.testing.assert_allclose(expected_norm, actual_norm, rtol=0.0001, atol=0.0001)
        torch.testing.assert_allclose(expected_mean, actual_mean)
        torch.testing.assert_allclose(expected_std, actual_std)
    
    @given(X=hu.tensor(min_dim=2), **hu.gcs)
    def test_layer_norm_brew_wrapper(self, X, gc, dc):
        axis = np.random.randint(0, len(X.shape))
        scale_dim = [1] * np.ndim(X)
        scale_dim[axis] = X.shape[axis]
        self.ws.create_blob('input').feed(X)
        model = ModelHelper(name='test_layer_norm_brew_wrapper')
        brew.layer_norm(model, 'input', 'output', dim_in=X.shape[axis:], axis=axis, epsilon=0.0001)
        self.ws.create_net(model.param_init_net).run()
        self.ws.create_net(model.net).run()

if __name__ == '__main__':
    unittest.main()

