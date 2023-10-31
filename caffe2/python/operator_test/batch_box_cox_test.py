from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np
TOLERANCE = 0.001

@st.composite
def _inputs():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.batch_box_cox_test._inputs', '_inputs()', {'draw': draw, 'TOLERANCE': TOLERANCE, 'st': st}, 5)


class TestBatchBoxCox(serial.SerializedTestCase):
    
    @serial.given(inputs=_inputs(), **hu.gcs_cpu_only)
    def test_batch_box_cox(self, inputs, gc, dc):
        self.batch_box_cox(inputs, gc, dc)
    
    @given(**hu.gcs_cpu_only)
    def test_lambda1_is_all_zero(self, gc, dc):
        inputs = (1, 1, [[2]], [0], [0])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (2, 1, [[2], [4]], [0], [0])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (1, 3, [[1, 2, 3]], [0, 0, 0], [0, 0, 0])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (2, 3, [[1, 2, 3], [4, 5, 6]], [0, 0, 0], [0, 0, 0])
        self.batch_box_cox(inputs, gc, dc)
    
    @given(**hu.gcs_cpu_only)
    def test_lambda1_is_partially_zero(self, gc, dc):
        inputs = (1, 5, [[1, 2, 3, 4, 5]], [0, -0.5, 0, 0.5, 0], [0.1, 0.2, 0.3, 0.4, 0.5])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (3, 5, [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [1, 2, 3, 4, 5]], [0, -0.5, 0, 0.5, 0], [0.1, 0.2, 0.3, 0.4, 0.5])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (2, 6, [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], [0, -0.5, 0, 0.5, 0, 1], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.batch_box_cox(inputs, gc, dc)
        inputs = (2, 7, [[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]], [0, -0.5, 0, 0.5, 0, 1, 0], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
        self.batch_box_cox(inputs, gc, dc)
    
    @given(**hu.gcs_cpu_only)
    def test_bound_base_away_from_zero(self, gc, dc):
        inputs = (2, 3, [[1e-05, 1e-06, 1e-07], [1e-07, -1e-06, 1e-05]], [0, 0, 0], [0, 0, 1e-06])
        self.batch_box_cox(inputs, gc, dc)
    
    def batch_box_cox(self, inputs, gc, dc):
        (N, D, data, lambda1, lambda2) = inputs
        data = np.array(data, dtype=np.float32).reshape(N, D)
        lambda1 = np.array(lambda1, dtype=np.float32)
        lambda2 = np.array(lambda2, dtype=np.float32)
        base = data + lambda2
        data[(base > 1 - TOLERANCE) & (base < 1 + TOLERANCE)] += 2 * TOLERANCE
        
        def ref(data, lambda1, lambda2):
            dim_1 = data.shape[1]
            output = np.copy(data)
            if data.size <= 0:
                return [output]
            for i in range(dim_1):
                output[:, i] = data[:, i] + lambda2[i]
                output[:, i] = np.maximum(output[:, i], 1e-06)
                if lambda1[i] == 0:
                    output[:, i] = np.log(output[:, i])
                else:
                    output[:, i] = (np.power(output[:, i], lambda1[i]) - 1) / lambda1[i]
            return [output]
        for naive in [False, True]:
            op = core.CreateOperator('BatchBoxCox', ['data', 'lambda1', 'lambda2'], ['output'], naive=naive, min_block_size=(0 if naive else 6))
            self.assertReferenceChecks(gc, op, [data, lambda1, lambda2], ref)

if __name__ == '__main__':
    import unittest
    unittest.main()

