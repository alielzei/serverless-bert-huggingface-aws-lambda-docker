from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import hypothesis.strategies as st
import numpy as np
from functools import partial
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial

def _unique_ref(x, return_inverse):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.unique_ops_test._unique_ref', '_unique_ref(x, return_inverse)', {'np': np, 'x': x, 'return_inverse': return_inverse}, 1)


class TestUniqueOps(serial.SerializedTestCase):
    
    @serial.given(X=hu.tensor1d(min_len=0, dtype=np.int32, elements=st.integers(min_value=-10, max_value=10)), return_remapping=st.booleans(), **hu.gcs)
    def test_unique_op(self, X, return_remapping, gc, dc):
        X = np.sort(X)
        op = core.CreateOperator('Unique', ['X'], (['U', 'remap'] if return_remapping else ['U']))
        self.assertDeviceChecks(device_options=dc, op=op, inputs=[X], outputs_to_check=([0, 1] if return_remapping else [0]))
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[X], reference=partial(_unique_ref, return_inverse=return_remapping))

if __name__ == '__main__':
    import unittest
    unittest.main()

