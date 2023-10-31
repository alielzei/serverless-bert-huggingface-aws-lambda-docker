from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from hypothesis import assume, given
import hypothesis.strategies as st
import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import core
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial


class TestDropout(serial.SerializedTestCase):
    
    @serial.given(X=hu.tensor(), in_place=st.booleans(), ratio=st.floats(0, 0.999), engine=st.sampled_from(['', 'CUDNN']), **hu.gcs)
    def test_dropout_is_test(self, X, in_place, ratio, engine, gc, dc):
        """Test with is_test=True for a deterministic reference impl."""
        if in_place:
            assume(not ((gc.device_type in {caffe2_pb2.CUDA, caffe2_pb2.HIP} and engine == '')))
            dc = dc[:1]
        op = core.CreateOperator('Dropout', ['X'], [('X' if in_place else 'Y')], ratio=ratio, engine=engine, is_test=True)
        self.assertDeviceChecks(dc, op, [X], [0])
        
        def reference_dropout_test(x):
            return (x, np.ones(x.shape, dtype=np.bool))
        self.assertReferenceChecks(gc, op, [X], reference_dropout_test, outputs_to_check=[0])
    
    @given(X=hu.tensor(), in_place=st.booleans(), output_mask=st.booleans(), engine=st.sampled_from(['', 'CUDNN']), **hu.gcs)
    def test_dropout_ratio0(self, X, in_place, output_mask, engine, gc, dc):
        """Test with ratio=0 for a deterministic reference impl."""
        if in_place:
            assume(gc.device_type not in {caffe2_pb2.CUDA, caffe2_pb2.HIP})
            dc = dc[:1]
        is_test = not output_mask
        op = core.CreateOperator('Dropout', ['X'], [('X' if in_place else 'Y')] + ((['mask'] if output_mask else [])), ratio=0.0, engine=engine, is_test=is_test)
        self.assertDeviceChecks(dc, op, [X], [0])
        if not is_test:
            self.assertGradientChecks(gc, op, [X], 0, [0])
        
        def reference_dropout_ratio0(x):
            return ((x, ) if is_test else (x, np.ones(x.shape, dtype=np.bool)))
        self.assertReferenceChecks(gc, op, [X], reference_dropout_ratio0, outputs_to_check=(None if engine != 'CUDNN' else [0]))


