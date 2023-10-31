from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from hypothesis import assume, given, settings, HealthCheck
import hypothesis.strategies as st
import numpy as np
import unittest

@st.composite
def _glu_old_input():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.glu_op_test._glu_old_input', '_glu_old_input()', {'draw': draw, 'hu': hu, 'np': np, 'st': st}, 2)


class TestGlu(serial.SerializedTestCase):
    
    @serial.given(X_axis=_glu_old_input(), **hu.gcs)
    def test_glu_old(self, X_axis, gc, dc):
        (X, axis) = X_axis
        
        def glu_ref(X):
            (x1, x2) = np.split(X, [X.shape[axis] // 2], axis=axis)
            Y = x1 * (1.0 / (1.0 + np.exp(-x2)))
            return [Y]
        op = core.CreateOperator('Glu', ['X'], ['Y'], dim=axis)
        self.assertReferenceChecks(gc, op, [X], glu_ref)

if __name__ == '__main__':
    unittest.main()

