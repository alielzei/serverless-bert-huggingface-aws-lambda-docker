from __future__ import absolute_import, division, print_function, unicode_literals
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import numpy as np
from caffe2.python import core, workspace
from hypothesis import given, strategies as st

def batched_boarders_and_data(data_min_size=5, data_max_size=10, examples_min_number=1, examples_max_number=4, example_min_size=1, example_max_size=3, dtype=np.float32, elements=None):
    dims_ = st.tuples(st.integers(min_value=data_min_size, max_value=data_max_size), st.integers(min_value=examples_min_number, max_value=examples_max_number), st.integers(min_value=example_min_size, max_value=example_max_size))
    return dims_.flatmap(lambda dims: st.tuples(hu.arrays([dims[1], dims[2], 2], dtype=np.int32, elements=st.integers(min_value=0, max_value=dims[0])), hu.arrays([dims[0]], dtype, elements)))

@st.composite
def _tensor_splits():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gather_ranges_op_test._tensor_splits', '_tensor_splits()', {'draw': draw, 'np': np, 'st': st}, 4)

@st.composite
def _bad_tensor_splits():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gather_ranges_op_test._bad_tensor_splits', '_bad_tensor_splits()', {'draw': draw, 'np': np, 'st': st}, 4)

def gather_ranges(data, ranges):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gather_ranges_op_test.gather_ranges', 'gather_ranges(data, ranges)', {'data': data, 'ranges': ranges}, 2)

def gather_ranges_to_dense(data, ranges, lengths):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gather_ranges_op_test.gather_ranges_to_dense', 'gather_ranges_to_dense(data, ranges, lengths)', {'np': np, 'data': data, 'ranges': ranges, 'lengths': lengths}, 1)

def gather_ranges_to_dense_with_key(data, ranges, key, lengths):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gather_ranges_op_test.gather_ranges_to_dense_with_key', 'gather_ranges_to_dense_with_key(data, ranges, key, lengths)', {'np': np, 'data': data, 'ranges': ranges, 'key': key, 'lengths': lengths}, 1)


class TestGatherRanges(serial.SerializedTestCase):
    
    @serial.given(boarders_and_data=batched_boarders_and_data(), **hu.gcs_cpu_only)
    def test_gather_ranges(self, boarders_and_data, gc, dc):
        (boarders, data) = boarders_and_data
        
        def boarders_to_range(boarders):
            assert len(boarders) == 2
            boarders = sorted(boarders)
            return [boarders[0], boarders[1] - boarders[0]]
        ranges = np.apply_along_axis(boarders_to_range, 2, boarders)
        self.assertReferenceChecks(device_option=gc, op=core.CreateOperator('GatherRanges', ['data', 'ranges'], ['output', 'lengths']), inputs=[data, ranges], reference=gather_ranges)
    
    @serial.given(tensor_splits=_tensor_splits(), **hu.gcs_cpu_only)
    def test_gather_ranges_split(self, tensor_splits, gc, dc):
        (data, ranges, lengths, _) = tensor_splits
        self.assertReferenceChecks(device_option=gc, op=core.CreateOperator('GatherRangesToDense', ['data', 'ranges'], ['X_{}'.format(i) for i in range(len(lengths))], lengths=lengths), inputs=[data, ranges, lengths], reference=gather_ranges_to_dense)
    
    @given(tensor_splits=_tensor_splits(), **hu.gcs_cpu_only)
    def test_gather_ranges_with_key_split(self, tensor_splits, gc, dc):
        (data, ranges, lengths, key) = tensor_splits
        self.assertReferenceChecks(device_option=gc, op=core.CreateOperator('GatherRangesToDense', ['data', 'ranges', 'key'], ['X_{}'.format(i) for i in range(len(lengths))], lengths=lengths), inputs=[data, ranges, key, lengths], reference=gather_ranges_to_dense_with_key)
    
    def test_shape_and_type_inference(self):
        with hu.temp_workspace('shape_type_inf_int32'):
            net = core.Net('test_net')
            net.ConstantFill([], 'ranges', shape=[3, 5, 2], dtype=core.DataType.INT32)
            net.ConstantFill([], 'values', shape=[64], dtype=core.DataType.INT64)
            net.GatherRanges(['values', 'ranges'], ['values_output', 'lengths_output'])
            (shapes, types) = workspace.InferShapesAndTypes([net], {})
            self.assertEqual(shapes['values_output'], [64])
            self.assertEqual(types['values_output'], core.DataType.INT64)
            self.assertEqual(shapes['lengths_output'], [3])
            self.assertEqual(types['lengths_output'], core.DataType.INT32)
    
    @given(tensor_splits=_bad_tensor_splits(), **hu.gcs_cpu_only)
    def test_empty_range_check(self, tensor_splits, gc, dc):
        (data, ranges, lengths, key) = tensor_splits
        workspace.FeedBlob('data', data)
        workspace.FeedBlob('ranges', ranges)
        workspace.FeedBlob('key', key)
        
        def getOpWithThreshold(min_observation=2, max_mismatched_ratio=0.5, max_empty_ratio=None):
            return core.CreateOperator('GatherRangesToDense', ['data', 'ranges', 'key'], ['X_{}'.format(i) for i in range(len(lengths))], lengths=lengths, min_observation=min_observation, max_mismatched_ratio=max_mismatched_ratio, max_empty_ratio=max_empty_ratio)
        workspace.RunOperatorOnce(getOpWithThreshold())
        workspace.RunOperatorOnce(getOpWithThreshold(max_mismatched_ratio=0.3, min_observation=50))
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(getOpWithThreshold(max_mismatched_ratio=0.3, min_observation=5))
        with self.assertRaises(RuntimeError):
            workspace.RunOperatorOnce(getOpWithThreshold(min_observation=50, max_empty_ratio=0.01))

if __name__ == '__main__':
    import unittest
    unittest.main()

