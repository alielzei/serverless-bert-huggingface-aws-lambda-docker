from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import numpy as np

def FakeQuantization8BitsRowwise(data):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.lengths_reducer_rowwise_8bit_ops_test.FakeQuantization8BitsRowwise', 'FakeQuantization8BitsRowwise(data)', {'np': np, 'data': data}, 1)


class TestQuantize8bits(hu.HypothesisTestCase):
    
    def test_quantize_op(self):
        op = core.CreateOperator('FloatToRowwiseQuantized8Bits', ['input_data'], ['quantized_input', 'scale_bias'])
        input_data = np.float32(np.asarray([[801.0, 786, 235.2, 2353.3434], [5.0, 11.0, 9.0, -2.0]]))
        workspace.FeedBlob('input_data', input_data)
        workspace.RunOperatorOnce(op)
        op1 = core.CreateOperator('Rowwise8BitQuantizedToFloat', ['quantized_input', 'scale_bias'], ['dequantized_input'])
        workspace.RunOperatorOnce(op1)
        result = workspace.FetchBlob('dequantized_input')
        ground_truth = FakeQuantization8BitsRowwise(input_data)
        np.testing.assert_array_almost_equal(result, ground_truth)
    
    def test_quantize_tensor_with_const_row_op(self):
        op = core.CreateOperator('FloatToRowwiseQuantized8Bits', ['input_data'], ['quantized_input', 'scale_bias'])
        input_data = np.float32(np.asarray([[801.0, 786, 235.2, 2353.3434], [9.0, 9.0, 9.0, 9.0]]))
        workspace.FeedBlob('input_data', input_data)
        workspace.RunOperatorOnce(op)
        op1 = core.CreateOperator('Rowwise8BitQuantizedToFloat', ['quantized_input', 'scale_bias'], ['dequantized_input'])
        workspace.RunOperatorOnce(op1)
        result = workspace.FetchBlob('dequantized_input')
        ground_truth = FakeQuantization8BitsRowwise(input_data)
        ground_truth[1, :] = 9.0
        np.testing.assert_array_almost_equal(result, ground_truth)
    
    def test_SparseSegmentUint8(self):
        init_net = core.Net('init')
        net = core.Net('bench')
        size = 10**3
        isize = 10**2
        d = init_net.UniformFill([], shape=[size, 32])
        w = init_net.UniformFill([], shape=[isize])
        i = init_net.UniformIntFill([], shape=[isize], max=size - 1)
        i = init_net.Cast([i], to=core.DataType.INT64)
        l = init_net.ConstantFill([], ['l'], shape=[isize // 10], value=10, dtype=core.DataType.INT32)
        net.FloatToRowwiseQuantized8Bits([d], ['quantized_data', 'scale_bias'])
        net.Rowwise8BitQuantizedToFloat(['quantized_data', 'scale_bias'], ['dequantized_data'])
        net.SparseLengthsWeightedSum(['dequantized_data', w, i, l], ['PositionWeighted_0'], engine='fp16')
        net.SparseLengthsWeightedSum8BitsRowwise(['quantized_data', w, i, l, 'scale_bias'], ['PositionWeighted_1'])
        net.SparseLengthsSum(['dequantized_data', i, l], ['Sum_0'], engine='fp16')
        net.SparseLengthsSum8BitsRowwise(['quantized_data', i, l, 'scale_bias'], ['Sum_1'])
        net.SparseLengthsMean(['dequantized_data', i, l], ['Mean_0'], engine='fp16')
        net.SparseLengthsMean8BitsRowwise(['quantized_data', i, l, 'scale_bias'], ['Mean_1'])
        gathered_w = net.Gather(['quantized_data', i], engine='fp16')
        gathered_scale_bias = net.Gather(['scale_bias', i], engine='fp16')
        net.Rowwise8BitQuantizedToFloat([gathered_w, gathered_scale_bias], 'Gathered_1')
        net.Gather(['dequantized_data', i], 'Gathered_0')
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        workspace.RunNetOnce(init_net)
        workspace.CreateNet(net)
        workspace.RunNetOnce(net)
        PositionWeighted_1 = workspace.FetchBlob('PositionWeighted_1')
        ground_truth_posw = workspace.FetchBlob('PositionWeighted_0')
        np.testing.assert_array_almost_equal(PositionWeighted_1, ground_truth_posw, decimal=5)
        Sum_1 = workspace.FetchBlob('Sum_1')
        ground_truth_sum = workspace.FetchBlob('Sum_0')
        np.testing.assert_array_almost_equal(Sum_1, ground_truth_sum, decimal=5)
        Mean_1 = workspace.FetchBlob('Mean_1')
        ground_truth_mean = workspace.FetchBlob('Mean_0')
        np.testing.assert_array_almost_equal(Mean_1, ground_truth_mean, decimal=5)
        Gathered_1 = workspace.FetchBlob('Gathered_1')
        ground_truth_gathered = workspace.FetchBlob('Gathered_0')
        np.testing.assert_array_almost_equal(Gathered_1, ground_truth_gathered, decimal=5)


