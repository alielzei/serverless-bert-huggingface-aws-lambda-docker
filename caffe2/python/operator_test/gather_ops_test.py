from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

def ref_gather_axis0():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gather_ops_test.ref_gather_axis0', 'ref_gather_axis0()', {'np': np}, 1)

def ref_gather(axis):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gather_ops_test.ref_gather', 'ref_gather(axis)', {'np': np, 'axis': axis}, 1)

def ref_gather_match_outer(axis=1):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gather_ops_test.ref_gather_match_outer', 'ref_gather_match_outer(axis=1)', {'np': np, 'axis': axis}, 1)


class TestGatherOps(serial.SerializedTestCase):
    
    @given(rows_num=st.integers(0, 10000), index_num=st.integers(0, 5000), **hu.gcs)
    def test_gather_ops(self, rows_num, index_num, gc, dc):
        data = np.random.random((rows_num, 10, 20)).astype(np.float32)
        if rows_num > 0:
            ind = np.random.randint(rows_num, size=(index_num, )).astype('int32')
        else:
            ind = np.random.randint(10, size=(index_num, )).astype('int32')
        op = core.CreateOperator('Gather', ['data', 'ind'], ['output'])
        self.assertReferenceChecks(gc, op, [data, ind], ref_gather_axis0())
        self.assertDeviceChecks(dc, op, [data, ind], [0])
        return
    
    @given(batch_num=st.integers(1, 4000), rows_num=st.integers(1, 6), index_num=st.integers(1, 20), **hu.gcs)
    def test_gather_ops_axis2(self, batch_num, rows_num, index_num, gc, dc):
        data = np.random.random((batch_num, rows_num, 5)).astype(np.float32)
        ind = np.random.randint(5, size=(index_num, )).astype('int32')
        op = core.CreateOperator('Gather', ['data', 'ind'], ['output'], axis=2)
        self.assertReferenceChecks(gc, op, [data, ind], ref_gather(axis=2))
        self.assertDeviceChecks(dc, op, [data, ind], [0])
        return
    
    @given(batch_num=st.integers(1, 40), rows_num=st.integers(1, 6), index_num=st.integers(1, 20), **hu.gcs_cpu_only)
    def test_gather_ops_match_outer(self, batch_num, rows_num, index_num, gc, dc):
        data = np.random.random((batch_num, rows_num, 5)).astype(np.float32)
        ind = np.random.randint(rows_num, size=(batch_num, index_num)).astype('int32')
        op = core.CreateOperator('Gather', ['data', 'ind'], ['output'], axis=1, match_outer=True)
        self.assertReferenceChecks(gc, op, [data, ind], ref_gather_match_outer())
        self.assertDeviceChecks(dc, op, [data, ind], [0])
        self.assertGradientChecks(gc, op, [data, ind], 0, [0])
        return
    
    @given(batch_num=st.integers(1, 40), rows_num=st.integers(1, 6), index_num=st.integers(1, 20), **hu.gcs_cpu_only)
    def test_batch_gather_op_match_outer(self, batch_num, rows_num, index_num, gc, dc):
        data = np.random.random((batch_num, rows_num, 5)).astype(np.float32)
        ind = np.random.randint(rows_num, size=(batch_num, index_num)).astype('int32')
        op = core.CreateOperator('BatchGather', ['data', 'ind'], ['output'], match_outer=True)
        self.assertReferenceChecks(gc, op, [data, ind], ref_gather_match_outer())
        self.assertDeviceChecks(dc, op, [data, ind], [0])
        self.assertGradientChecks(gc, op, [data, ind], 0, [0])
        return
    
    @given(batch_num=st.integers(1, 30), rows_num=st.integers(1, 6), index_num=st.integers(1, 10), index_num2=st.integers(1, 10), axis2_num=st.integers(1, 10), **hu.gcs_cpu_only)
    def test_gather_op_match_outer_axis2_data4D_ind4D(self, batch_num, rows_num, axis2_num, index_num, index_num2, gc, dc):
        data = np.random.random((batch_num, rows_num, axis2_num, 5)).astype(np.float32)
        ind = np.random.randint(axis2_num, size=(batch_num, rows_num, index_num, index_num2)).astype('int32')
        op = core.CreateOperator('Gather', ['data', 'ind'], ['output'], axis=2, match_outer=True)
        self.assertReferenceChecks(gc, op, [data, ind], ref_gather_match_outer(axis=2))
        self.assertDeviceChecks(dc, op, [data, ind], [0])
        self.assertGradientChecks(gc, op, [data, ind], 0, [0], threshold=0.02)
        return


@st.composite
def _inputs():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gather_ops_test._inputs', '_inputs()', {'draw': draw, 'hnp': hnp, 'np': np, 'st': st}, 2)


class TestBatchGatherOps(hu.HypothesisTestCase):
    
    @given(inputs=_inputs(), **hu.gcs)
    def test_batch_gather_ops(self, inputs, gc, dc):
        (data, ind) = inputs
        op = core.CreateOperator('BatchGather', ['data', 'ind'], ['output'])
        self.assertReferenceChecks(gc, op, [data, ind], ref_gather(axis=1))
        self.assertGradientChecks(gc, op, [data, ind], 0, [0])



class TestGatherFused8BitRowwise(hu.HypothesisTestCase):
    
    @given(rows_num=st.integers(1, 10000), cols_num=st.integers(1, 128), index_num=st.integers(0, 5000), **hu.gcs)
    def test_batch_gather_ops(self, rows_num, cols_num, index_num, gc, dc):
        data = np.random.random((rows_num, cols_num)).astype(np.float32)
        ind = np.random.randint(rows_num, size=(index_num, )).astype('int32')
        net = core.Net('bench')
        quantized_data = net.FloatToFused8BitRowwiseQuantized('data', 'quantized_data')
        dequantized_data = net.Fused8BitRowwiseQuantizedToFloat(quantized_data, 'dequantized_data')
        net.Gather([dequantized_data, 'ind'], 'gather_reference')
        net.GatherFused8BitRowwise([quantized_data, 'ind'], 'gather_quantized')
        workspace.FeedBlob('data', data)
        workspace.FeedBlob('ind', ind)
        workspace.CreateNet(net)
        workspace.RunNetOnce(net)
        gather_reference = workspace.FetchBlob('gather_reference')
        gather_quantized = workspace.FetchBlob('gather_quantized')
        np.testing.assert_array_almost_equal(gather_reference, gather_quantized)

if __name__ == '__main__':
    import unittest
    unittest.main()

