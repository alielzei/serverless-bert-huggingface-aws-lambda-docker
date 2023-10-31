from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np

@st.composite
def id_list_batch():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.merge_id_lists_op_test.id_list_batch', 'id_list_batch()', {'draw': draw, 'np': np, 'hnp': hnp, 'hu': hu, 'st': st}, 1)

def merge_id_lists_ref(*args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.merge_id_lists_op_test.merge_id_lists_ref', 'merge_id_lists_ref(*args)', {'np': np, 'args': args}, 1)


class TestMergeIdListsOp(serial.SerializedTestCase):
    
    def test_merge_id_lists_ref(self):
        lengths_0 = np.array([3, 0, 4], dtype=np.int32)
        values_0 = np.array([1, 5, 6, 2, 4, 5, 6], dtype=np.int64)
        lengths_1 = np.array([3, 2, 1], dtype=np.int32)
        values_1 = np.array([5, 8, 9, 14, 9, 5], dtype=np.int64)
        (merged_lengths, merged_values) = merge_id_lists_ref(lengths_0, values_0, lengths_1, values_1)
        expected_lengths = np.array([5, 2, 4], dtype=np.int32)
        expected_values = np.array([1, 5, 6, 8, 9, 9, 14, 2, 4, 5, 6], dtype=np.int64)
        np.testing.assert_array_equal(merged_lengths, expected_lengths)
        np.testing.assert_array_equal(merged_values, expected_values)
    
    @serial.given(inputs=id_list_batch(), **hu.gcs_cpu_only)
    def test_merge_id_lists_op(self, inputs, gc, dc):
        num_inputs = int(len(inputs) / 2)
        op = core.CreateOperator('MergeIdLists', ['{prefix}_{i}'.format(prefix=p, i=i) for i in range(num_inputs) for p in ['lengths', 'values']], ['merged_lengths', 'merged_values'])
        self.assertDeviceChecks(dc, op, inputs, [0])
        self.assertReferenceChecks(gc, op, inputs, merge_id_lists_ref)


