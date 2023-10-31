from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np

@st.composite
def id_list_batch():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.dense_vector_to_id_list_op_test.id_list_batch', 'id_list_batch()', {'draw': draw, 'np': np, 'hnp': hnp, 'st': st}, 1)

def dense_vector_to_id_list_ref(*arg):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.dense_vector_to_id_list_op_test.dense_vector_to_id_list_ref', 'dense_vector_to_id_list_ref(*arg)', {'arg': arg}, 2)


class TestDenseVectorToIdList(hu.HypothesisTestCase):
    
    def test_dense_vector_to_id_list_ref(self):
        dense_input = np.array([[1, 0, 0, 1, 0, 0, 0, 1], [1, 0, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1, 0, 1]], dtype=np.float32)
        (sparse_lengths, sparse_values) = dense_vector_to_id_list_ref(dense_input)
        expected_lengths = np.array([3, 3, 3], dtype=np.int32)
        expected_values = np.array([0, 3, 7, 0, 2, 7, 1, 5, 7], dtype=np.int64)
        np.testing.assert_array_equal(sparse_lengths, expected_lengths)
        np.testing.assert_array_equal(sparse_values, expected_values)
    
    @given(inputs=id_list_batch(), **hu.gcs_cpu_only)
    def test_dense_vector_to_id_list_op(self, inputs, gc, dc):
        op = core.CreateOperator('DenseVectorToIdList', ['values'], ['out_lengths', 'out_values'])
        self.assertDeviceChecks(dc, op, inputs, [0])
        self.assertReferenceChecks(gc, op, inputs, dense_vector_to_id_list_ref)


