from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
import numpy as np
import struct
from hypothesis import given
round_to_nearest = np.vectorize(round)

def bytes_to_floats(byte_matrix):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.fused_8bit_rowwise_conversion_ops_test.bytes_to_floats', 'bytes_to_floats(byte_matrix)', {'np': np, 'struct': struct, 'byte_matrix': byte_matrix}, 1)

def floats_to_bytes(floats):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.fused_8bit_rowwise_conversion_ops_test.floats_to_bytes', 'floats_to_bytes(floats)', {'np': np, 'struct': struct, 'floats': floats}, 1)

def fused_rowwise_8bit_quantize_reference(data):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.fused_8bit_rowwise_conversion_ops_test.fused_rowwise_8bit_quantize_reference', 'fused_rowwise_8bit_quantize_reference(data)', {'np': np, 'round_to_nearest': round_to_nearest, 'floats_to_bytes': floats_to_bytes, 'data': data}, 1)

def fused_rowwise_8bit_quantize_dequantize_reference(data):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.fused_8bit_rowwise_conversion_ops_test.fused_rowwise_8bit_quantize_dequantize_reference', 'fused_rowwise_8bit_quantize_dequantize_reference(data)', {'fused_rowwise_8bit_quantize_reference': fused_rowwise_8bit_quantize_reference, 'bytes_to_floats': bytes_to_floats, 'np': np, 'data': data}, 1)


class TestFused8BitRowwiseQuantizationConversion(hu.HypothesisTestCase):
    
    @given(input_data=hu.tensor(min_dim=1, max_dim=3, max_value=33))
    def test_quantize_op(self, input_data):
        quantize = core.CreateOperator('FloatToFused8BitRowwiseQuantized', ['input_data'], ['quantized_data'])
        workspace.FeedBlob('input_data', input_data)
        workspace.RunOperatorOnce(quantize)
        quantized_data = workspace.FetchBlob('quantized_data')
        reference = fused_rowwise_8bit_quantize_reference(input_data.astype(np.float32))
        np.testing.assert_array_almost_equal(quantized_data, reference)
    
    @given(input_data=hu.tensor(min_dim=1, max_dim=3, max_value=33))
    def test_quantize_and_dequantize_op(self, input_data):
        quantize = core.CreateOperator('FloatToFused8BitRowwiseQuantized', ['input_data'], ['quantized_data'])
        workspace.FeedBlob('input_data', input_data)
        workspace.RunOperatorOnce(quantize)
        quantized_data = workspace.FetchBlob('quantized_data')
        dequantize = core.CreateOperator('Fused8BitRowwiseQuantizedToFloat', ['quantized_data'], ['dequantized_data'])
        workspace.FeedBlob('quantized_data', quantized_data)
        workspace.RunOperatorOnce(dequantize)
        dequantized_data = workspace.FetchBlob('dequantized_data')
        reference = fused_rowwise_8bit_quantize_dequantize_reference(input_data)
        np.testing.assert_array_almost_equal(dequantized_data, reference)


