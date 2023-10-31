from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import functools
import logging
import hypothesis
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
import numpy as np
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
logger = logging.getLogger(__name__)

def ref_wngrad(param_in, seq_b_in, grad, lr, epsilon, output_effective_lr=False, output_effective_lr_and_update=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.wngrad_test.ref_wngrad', 'ref_wngrad(param_in, seq_b_in, grad, lr, epsilon, output_effective_lr=False, output_effective_lr_and_update=False)', {'np': np, 'param_in': param_in, 'seq_b_in': seq_b_in, 'grad': grad, 'lr': lr, 'epsilon': epsilon, 'output_effective_lr': output_effective_lr, 'output_effective_lr_and_update': output_effective_lr_and_update}, 4)

def wngrad_sparse_test_helper(parent_test, inputs, seq_b, lr, epsilon, engine, gc, dc):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.wngrad_test.wngrad_sparse_test_helper', 'wngrad_sparse_test_helper(parent_test, inputs, seq_b, lr, epsilon, engine, gc, dc)', {'np': np, 'core': core, 'logger': logger, 'parent_test': parent_test, 'inputs': inputs, 'seq_b': seq_b, 'lr': lr, 'epsilon': epsilon, 'engine': engine, 'gc': gc, 'dc': dc}, 2)


class TestWngrad(serial.SerializedTestCase):
    
    @serial.given(inputs=hu.tensors(n=2), seq_b=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), lr=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), epsilon=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), **hu.gcs_cpu_only)
    def test_wngrad_dense_base(self, inputs, seq_b, lr, epsilon, gc, dc):
        (param, grad) = inputs
        seq_b = np.array([seq_b], dtype=np.float32)
        lr = np.array([lr], dtype=np.float32)
        op = core.CreateOperator('Wngrad', ['param', 'seq_b', 'grad', 'lr'], ['param', 'seq_b'], epsilon=epsilon, device_option=gc)
        self.assertReferenceChecks(gc, op, [param, seq_b, grad, lr], functools.partial(ref_wngrad, epsilon=epsilon))
    
    @given(inputs=hu.tensors(n=2), seq_b=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), lr=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), epsilon=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), **hu.gcs_cpu_only)
    def test_wngrad_dense_output_effective_lr(self, inputs, seq_b, lr, epsilon, gc, dc):
        (param, grad) = inputs
        seq_b = np.array([seq_b], dtype=np.float32)
        lr = np.array([lr], dtype=np.float32)
        op = core.CreateOperator('Wngrad', ['param', 'seq_b', 'grad', 'lr'], ['param', 'seq_b', 'effective_lr'], epsilon=epsilon, device_option=gc)
        self.assertReferenceChecks(gc, op, [param, seq_b, grad, lr], functools.partial(ref_wngrad, epsilon=epsilon, output_effective_lr=True))
    
    @given(inputs=hu.tensors(n=2), seq_b=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), lr=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), epsilon=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), **hu.gcs_cpu_only)
    def test_wngrad_dense_output_effective_lr_and_update(self, inputs, seq_b, lr, epsilon, gc, dc):
        (param, grad) = inputs
        seq_b = np.abs(np.array([seq_b], dtype=np.float32))
        lr = np.array([lr], dtype=np.float32)
        op = core.CreateOperator('Wngrad', ['param', 'seq_b', 'grad', 'lr'], ['param', 'seq_b', 'effective_lr', 'update'], epsilon=epsilon, device_option=gc)
        self.assertReferenceChecks(gc, op, [param, seq_b, grad, lr], functools.partial(ref_wngrad, epsilon=epsilon, output_effective_lr_and_update=True))
    
    @settings(suppress_health_check=[HealthCheck.filter_too_much])
    @given(inputs=hu.tensors(n=2), seq_b=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), lr=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), epsilon=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), **hu.gcs_cpu_only)
    def test_sparse_wngrad(self, inputs, seq_b, lr, epsilon, gc, dc):
        return wngrad_sparse_test_helper(self, inputs, seq_b, lr, epsilon, None, gc, dc)
    
    @serial.given(inputs=hu.tensors(n=1), lr=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), seq_b=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), epsilon=st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False), **hu.gcs_cpu_only)
    def test_sparse_wngrad_empty(self, inputs, seq_b, lr, epsilon, gc, dc):
        param = inputs[0]
        seq_b = np.array([seq_b], dtype=np.float32)
        lr = np.array([lr], dtype=np.float32)
        grad = np.empty(shape=(0, ) + param.shape[1:], dtype=np.float32)
        indices = np.empty(shape=(0, ), dtype=np.int64)
        hypothesis.note('indices.shape: %s' % str(indices.shape))
        op = core.CreateOperator('SparseWngrad', ['param', 'seq_b', 'indices', 'grad', 'lr'], ['param', 'seq_b'], epsilon=epsilon, device_option=gc)
        
        def ref_sparse(param, seq_b, indices, grad, lr):
            param_out = np.copy(param)
            seq_b_out = np.copy(seq_b)
            return (param_out, seq_b_out)
        print('test_sparse_adagrad_empty with full precision embedding')
        seq_b_i = seq_b.astype(np.float32)
        param_i = param.astype(np.float32)
        self.assertReferenceChecks(gc, op, [param_i, seq_b_i, indices, grad, lr], ref_sparse)


