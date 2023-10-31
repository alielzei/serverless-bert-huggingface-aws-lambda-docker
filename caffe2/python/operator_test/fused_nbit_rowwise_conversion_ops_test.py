from __future__ import absolute_import, division, print_function, unicode_literals
import struct
import caffe2.python.hypothesis_test_util as hu
import hypothesis.strategies as st
import numpy as np
from caffe2.python.operator_test.fused_nbit_rowwise_test_helper import _compress_uniform_simplified, param_search_greedy
from caffe2.python import core, dyndep, workspace
from hypothesis import assume, given
round_to_nearest = np.vectorize(round)

def bytes_to_half_floats(byte_matrix):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.fused_nbit_rowwise_conversion_ops_test.bytes_to_half_floats', 'bytes_to_half_floats(byte_matrix)', {'np': np, 'byte_matrix': byte_matrix}, 1)

def half_floats_to_bytes(floats):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.fused_nbit_rowwise_conversion_ops_test.half_floats_to_bytes', 'half_floats_to_bytes(floats)', {'np': np, 'floats': floats}, 1)

def int8_to_bytes(int8s):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.fused_nbit_rowwise_conversion_ops_test.int8_to_bytes', 'int8_to_bytes(int8s)', {'np': np, 'struct': struct, 'int8s': int8s}, 1)

def fused_rowwise_nbit_quantize_reference(data, bit):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.fused_nbit_rowwise_conversion_ops_test.fused_rowwise_nbit_quantize_reference', 'fused_rowwise_nbit_quantize_reference(data, bit)', {'np': np, 'half_floats_to_bytes': half_floats_to_bytes, 'data': data, 'bit': bit}, 1)

def fused_rowwise_nbit_quantize_dequantize_reference(data, bit):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.fused_nbit_rowwise_conversion_ops_test.fused_rowwise_nbit_quantize_dequantize_reference', 'fused_rowwise_nbit_quantize_dequantize_reference(data, bit)', {'fused_rowwise_nbit_quantize_reference': fused_rowwise_nbit_quantize_reference, 'bytes_to_half_floats': bytes_to_half_floats, 'np': np, 'data': data, 'bit': bit}, 1)


class TestFusedNBitRowwiseQuantizationConversion(hu.HypothesisTestCase):
    
    @given(input_data=hu.tensor(min_dim=2, max_dim=2), bit_rate=st.sampled_from([2, 4]))
    def test_quantize_op(self, input_data, bit_rate):
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        assume(input_data.shape[1] % num_elem_per_byte == 0)
        quantize = core.CreateOperator('FloatToFused' + str(bit_rate) + 'BitRowwiseQuantized', ['input_data'], ['quantized_data'])
        workspace.FeedBlob('input_data', input_data)
        workspace.RunOperatorOnce(quantize)
        quantized_data = workspace.FetchBlob('quantized_data')
        reference = fused_rowwise_nbit_quantize_reference(input_data.astype(np.float32), bit_rate)
        interleaved_dim = input_data.shape[1] // num_elem_per_byte
        np.testing.assert_array_equal(quantized_data[:, :interleaved_dim], reference[:, :interleaved_dim])
        np.testing.assert_array_almost_equal(bytes_to_half_floats(quantized_data[:, interleaved_dim:interleaved_dim + 2]), bytes_to_half_floats(reference[:, interleaved_dim:interleaved_dim + 2]))
        np.testing.assert_array_equal(quantized_data[:, interleaved_dim + 2], reference[:, interleaved_dim + 2])
    
    @given(input_data=hu.tensor(min_dim=2, max_dim=2), bit_rate=st.sampled_from([2, 4]))
    def test_quantize_and_dequantize_op(self, input_data, bit_rate):
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        assume(input_data.shape[1] % num_elem_per_byte == 0)
        quantize = core.CreateOperator('FloatToFused' + str(bit_rate) + 'BitRowwiseQuantized', ['input_data'], ['quantized_data'])
        workspace.FeedBlob('input_data', input_data)
        workspace.RunOperatorOnce(quantize)
        quantized_data = workspace.FetchBlob('quantized_data')
        dequantize = core.CreateOperator('Fused' + str(bit_rate) + 'BitRowwiseQuantizedToFloat', ['quantized_data'], ['dequantized_data'])
        workspace.FeedBlob('quantized_data', quantized_data)
        workspace.RunOperatorOnce(dequantize)
        dequantized_data = workspace.FetchBlob('dequantized_data')
        reference = fused_rowwise_nbit_quantize_dequantize_reference(input_data, bit_rate)
        np.testing.assert_array_almost_equal(dequantized_data, reference)


def ErrorThresholdRow(X, bit_rate):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.fused_nbit_rowwise_conversion_ops_test.ErrorThresholdRow', 'ErrorThresholdRow(X, bit_rate)', {'np': np, 'X': X, 'bit_rate': bit_rate}, 1)


class TestNBitFakeFused(hu.HypothesisTestCase):
    
    @given(bit_rate=st.sampled_from([2, 4]))
    def testNBit(self, bit_rate):
        net = core.Net('bench')
        batchsize = np.random.randint(2, 1000)
        blocksize = np.random.randint(2, 1000)
        input_data = np.random.rand(batchsize, blocksize).astype(np.float32)
        op = core.CreateOperator('FloatToFused' + str(bit_rate) + 'BitFakeRowwiseQuantized', 'input_data', 'minmax_quantized_data')
        net.Proto().op.extend([op])
        net.Fused8BitRowwiseQuantizedToFloat('minmax_quantized_data', 'minmax_dequantized_data')
        op = core.CreateOperator('FloatToFused' + str(bit_rate) + 'BitFakeRowwiseQuantized', 'input_data', 'greedy_quantized_data', engine='GREEDY')
        net.Proto().op.extend([op])
        net.Fused8BitRowwiseQuantizedToFloat('greedy_quantized_data', 'greedy_dequantized_data')
        workspace.FeedBlob('input_data', input_data)
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        workspace.RunNetOnce(net)
        minmax_dequantized_data = workspace.FetchBlob('minmax_dequantized_data')
        greedy_dequantized_data = workspace.FetchBlob('greedy_dequantized_data')
        err_thres = ErrorThresholdRow(input_data, bit_rate)
        diff_minmax = np.abs(input_data - minmax_dequantized_data)
        diff_greedy = np.abs(input_data - greedy_dequantized_data)
        for i in range(err_thres.size):
            assert np.sum(diff_minmax[i, :] > err_thres[i]) == 0, 'error at row {} too high (diff_minmax[i, :] {} diff_minmax[i, :] > err_thres[i] {} err_thres[i] {}'.format(i, diff_minmax[i, :], diff_minmax[i, :] > err_thres[i], err_thres[i])
            l2_minmax_err = np.linalg.norm(diff_minmax[i, :])
            l2_greedy_err = np.linalg.norm(diff_greedy[i, :])
            assert l2_greedy_err <= l2_minmax_err * 1.03, 'L2 quantization error using greedy algorithm {} at row {} is bigger than error using minmax {} (input_data[i,:] {} minmax_dequantized_data[i,:] {} greedy_dequantized_data[i,:] {}'.format(l2_greedy_err, i, l2_minmax_err, input_data[i, :], minmax_dequantized_data[i, :], greedy_dequantized_data[i, :])



class TestNBitGreedyFused(hu.HypothesisTestCase):
    
    @given(bit_rate=st.sampled_from([2, 4]))
    def testNBit(self, bit_rate):
        net = core.Net('bench')
        batchsize = np.random.randint(2, 1000)
        assert 8 % bit_rate == 0
        num_elem_per_byte = 8 // bit_rate
        blocksize = np.random.randint(2, 500) * num_elem_per_byte
        input_data = np.random.rand(batchsize, blocksize).astype(np.float32)
        op = core.CreateOperator('FloatToFused' + str(bit_rate) + 'BitRowwiseQuantized', 'input_data', 'minmax_quantized_data')
        net.Proto().op.extend([op])
        op = core.CreateOperator('Fused' + str(bit_rate) + 'BitRowwiseQuantizedToFloat', 'minmax_quantized_data', 'minmax_dequantized_data')
        net.Proto().op.extend([op])
        op = core.CreateOperator('FloatToFused' + str(bit_rate) + 'BitRowwiseQuantized', 'input_data', 'greedy_quantized_data', engine='GREEDY')
        net.Proto().op.extend([op])
        op = core.CreateOperator('Fused' + str(bit_rate) + 'BitRowwiseQuantizedToFloat', 'greedy_quantized_data', 'greedy_dequantized_data')
        net.Proto().op.extend([op])
        workspace.FeedBlob('input_data', input_data)
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
        workspace.RunNetOnce(net)
        minmax_dequantized_data = workspace.FetchBlob('minmax_dequantized_data')
        greedy_dequantized_data = workspace.FetchBlob('greedy_dequantized_data')
        diff_minmax = np.abs(input_data - minmax_dequantized_data)
        l2_minmax = np.linalg.norm(input_data - minmax_dequantized_data, axis=1)
        diff_greedy = np.abs(input_data - greedy_dequantized_data)
        l2_greedy = np.linalg.norm(input_data - greedy_dequantized_data, axis=1)
        for i in range(input_data.shape[0]):
            (xmin, xmax) = param_search_greedy(input_data[i, :], bit_rate, n_bins=200, ratio=0.16)
            (X_q_ref, l2_greedy_ref) = _compress_uniform_simplified(input_data[i, :], bit_rate, xmin, xmax, fp16_scale_bias=True)
            l2_discrepancy = np.abs(l2_greedy[i] - l2_greedy_ref) / input_data.shape[1]
            assert l2_discrepancy < 1e-05, 'l2_discrepancy between C++ and Python greedy algorithm {} at row {} is too high (actual l2 err {} ref l2 err {} actual {} ref {})'.format(l2_discrepancy, i, l2_greedy[i], l2_greedy_ref, greedy_dequantized_data[i, :], X_q_ref)
            assert l2_greedy[i] <= l2_minmax[i] * 1.03, 'L2 quantization error using greedy algorithm {} at row {} is bigger than error using minmax {}'.format(l2_greedy[i], i, l2_minmax[i])


