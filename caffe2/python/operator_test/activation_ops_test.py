from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from hypothesis import given, assume
import hypothesis.strategies as st
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.mkl_test_util as mu
import caffe2.python.serialized_test.serialized_test_util as serial
from scipy.stats import norm
import unittest


class TestActivations(serial.SerializedTestCase):
    
    @serial.given(X=hu.tensor(), in_place=st.booleans(), engine=st.sampled_from(['', 'CUDNN']), **mu.gcs)
    def test_relu(self, X, in_place, engine, gc, dc):
        if gc == mu.mkl_do:
            in_place = False
        op = core.CreateOperator('Relu', ['X'], (['X'] if in_place else ['Y']), engine=engine)
        
        def relu_ref(X):
            return [np.maximum(X, 0.0)]
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02
        self.assertReferenceChecks(gc, op, [X], relu_ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])
    
    @given(N=st.integers(1, 10), M=st.integers(1, 10), in_place=st.booleans(), **hu.gcs)
    def test_relu_empty_input(self, N, M, in_place, gc, dc):
        op = core.CreateOperator('Relu', ['X'], (['X'] if in_place else ['Y']))
        
        def relu_ref(X):
            return [np.maximum(X, 0.0)]
        X = np.random.randn(0, N, M).astype(np.float32)
        self.assertReferenceChecks(gc, op, [X], relu_ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])
    
    @unittest.skipIf(not workspace.has_gpu_support, 'Relu for float16 can only run on GPU now.')
    @given(X=hu.tensor(dtype=np.float16), in_place=st.booleans(), engine=st.sampled_from(['', 'CUDNN']), **hu.gcs)
    def test_relu_fp16(self, X, in_place, engine, gc, dc):
        assume(core.IsGPUDeviceType(gc.device_type))
        op = core.CreateOperator('Relu', ['X'], (['X'] if in_place else ['Y']), engine=engine)
        
        def relu_ref(X):
            return [np.maximum(X, 0.0)]
        
        def relu_grad_ref(g_out, outputs, fwd_inputs):
            dY = g_out
            [Y] = outputs
            dX = dY
            dX[Y == 0] = 0
            return [dX]
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02
        self.assertReferenceChecks(gc, op, [X], relu_ref, output_to_grad=('X' if in_place else 'Y'), grad_reference=relu_grad_ref)
    
    @serial.given(X=hu.tensor(elements=st.floats(-3.0, 3.0)), n=st.floats(min_value=0.5, max_value=2.0), in_place=st.booleans(), **hu.gcs)
    def test_relu_n(self, X, n, in_place, gc, dc):
        op = core.CreateOperator('ReluN', ['X'], (['X'] if in_place else ['Y']), n=n)
        
        def relu_n_ref(X):
            return [np.minimum(np.maximum(X, 0.0), n)]
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04
        X -= n
        X += 0.02 * np.sign(X)
        X[X == 0.0] -= 0.02
        X += n
        self.assertReferenceChecks(gc, op, [X], relu_n_ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=0.005)
    
    @serial.given(X=hu.tensor(), alpha=st.floats(min_value=0.1, max_value=2.0), in_place=st.booleans(), engine=st.sampled_from(['', 'CUDNN']), **hu.gcs)
    def test_elu(self, X, alpha, in_place, engine, gc, dc):
        op = core.CreateOperator('Elu', ['X'], (['X'] if in_place else ['Y']), alpha=alpha, engine=engine)
        
        def elu_ref(X):
            Y = X
            Y[X < 0] = alpha * (np.exp(X[X < 0]) - 1.0)
            return [Y]
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04
        self.assertReferenceChecks(gc, op, [X], elu_ref)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=0.01)
    
    @given(X=hu.tensor(min_dim=4, max_dim=4), alpha=st.floats(min_value=0.1, max_value=2.0), inplace=st.booleans(), shared=st.booleans(), order=st.sampled_from(['NCHW', 'NHWC']), seed=st.sampled_from([20, 100]), **hu.gcs)
    def test_prelu(self, X, alpha, inplace, shared, order, seed, gc, dc):
        np.random.seed(seed)
        W = np.random.randn((X.shape[1] if order == 'NCHW' else X.shape[3])).astype(np.float32)
        if shared:
            W = np.random.randn(1).astype(np.float32)
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04
        
        def prelu_ref(X, W):
            Y = X.copy()
            W = (W.reshape(1, -1, 1, 1) if order == 'NCHW' else W.reshape(1, 1, 1, -1))
            assert len(X.shape) == 4
            neg_indices = X <= 0
            assert len(neg_indices.shape) == 4
            assert X.shape == neg_indices.shape
            Y[neg_indices] = (Y * W)[neg_indices]
            return (Y, )
        op = core.CreateOperator('PRelu', ['X', 'W'], [('Y' if not inplace else 'X')], alpha=alpha, order=order)
        self.assertReferenceChecks(gc, op, [X, W], prelu_ref)
        self.assertDeviceChecks(dc, op, [X, W], [0])
        if not inplace:
            self.assertGradientChecks(gc, op, [X, W], 0, [0], stepsize=0.01)
            self.assertGradientChecks(gc, op, [X, W], 1, [0], stepsize=0.01)
    
    @serial.given(X=hu.tensor(), alpha=st.floats(min_value=0.1, max_value=2.0), inplace=st.booleans(), **hu.gcs)
    def test_leaky_relu(self, X, alpha, inplace, gc, dc):
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04
        
        def leaky_relu_ref(X):
            Y = X.copy()
            neg_indices = X <= 0
            Y[neg_indices] = Y[neg_indices] * alpha
            return (Y, )
        op = core.CreateOperator('LeakyRelu', ['X'], [('Y' if not inplace else 'X')], alpha=alpha)
        self.assertReferenceChecks(gc, op, [X], leaky_relu_ref)
        self.assertDeviceChecks(dc, op, [X], [0])
    
    @given(X=hu.tensor(), inplace=st.booleans(), **hu.gcs)
    def test_leaky_relu_default(self, X, inplace, gc, dc):
        X += 0.04 * np.sign(X)
        X[X == 0.0] += 0.04
        
        def leaky_relu_ref(X):
            Y = X.copy()
            neg_indices = X <= 0
            Y[neg_indices] = Y[neg_indices] * 0.01
            return (Y, )
        op = core.CreateOperator('LeakyRelu', ['X'], [('Y' if not inplace else 'X')])
        self.assertReferenceChecks(gc, op, [X], leaky_relu_ref)
        self.assertDeviceChecks(dc, op, [X], [0])
    
    @given(X=hu.tensor(), fast_gelu=st.booleans(), **hu.gcs)
    def test_gelu(self, X, fast_gelu, gc, dc):
        op = core.CreateOperator('Gelu', ['X'], ['Y'], fast_gelu=fast_gelu)
        
        def gelu_ref(X):
            return (X * norm.cdf(X), )
        tol = (0.001 if fast_gelu else 0.0001)
        self.assertReferenceChecks(gc, op, [X], gelu_ref, threshold=tol)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0])

if __name__ == '__main__':
    unittest.main()

